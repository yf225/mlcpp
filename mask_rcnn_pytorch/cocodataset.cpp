#include "cocodataset.h"
#include "anchors.h"
#include "boxutils.h"
#include "datasetclasses.h"
#include "imageutils.h"
#include "nnutils.h"

namespace {
/* Given the anchors and GT boxes, compute overlaps and identify positive
 * anchors and deltas to refine them to match their corresponding GT boxes.
 * anchors: [num_anchors, (y1, x1, y2, x2)]
 * gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
 * Returns:
 * rpn_match: [N] (int32) matches between anchors and GT boxes.
 *            1 = positive anchor, -1 = negative anchor, 0 = neutral
 * rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
 */
std::tuple<at::Tensor, at::Tensor> BuildRpnTargets(at::Tensor anchors,
                                                   at::Tensor gt_boxes,
                                                   const Config& config) {
  // RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
  auto rpn_match = torch::zeros({anchors.size(0)}, at::dtype(at::kInt));
  // RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
  auto rpn_bbox = torch::zeros({config.rpn_train_anchors_per_image, 4});

  auto minus_one = torch::tensor(-1);
  auto one = torch::tensor(1);

  // Handle COCO crowds
  // A crowd box in COCO is a bounding box around several instances.
  // They are excluded on loading stage

  // Compute overlaps [num_anchors, num_gt_boxes]
  // use loops because anchors are too big
  auto overlaps = BBoxOverlapsLoops(anchors, gt_boxes);

  //  // Debug block
  //  {
  //    std::cerr << std::get<0>(overlaps.sort(0, true)).narrow(0, 0,
  //    10).squeeze(); auto max_overlaps =
  //        std::get<1>(overlaps.sort(0, true)).narrow(0, 0, 10).squeeze();
  //    auto max_anchors = anchors.index_select(0, max_overlaps);
  //    VisualizeRPNTrargets(config.image_shape[0], config.image_shape[1],
  //                         max_anchors, gt_boxes);
  //    exit(0);
  //  }

  // Match anchors to GT Boxes
  // If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
  // If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
  // Neutral anchors are those that don't match the conditions above,
  // and they don't influence the loss function.
  // However, don't keep any GT box unmatched (rare, but happens). Instead,
  // match it to the closest anchor (even if its max IoU is < 0.3).

  // 1. Set negative anchors first. They get overwritten below if a GT box is
  // matched to them. Skip boxes in crowd areas.
  auto anchor_iou_argmax = torch::argmax(overlaps, /*dim*/ 1);
  auto anchor_iou_max =
      overlaps.index({torch::arange(overlaps.size(0), at::dtype(at::kLong)),
                      anchor_iou_argmax});
  rpn_match = torch::where(anchor_iou_max < 0.5, minus_one, rpn_match);  // 0.3

  // 2. Set an anchor for each GT box (regardless of IoU value).
  // TODO: (Legacy)If multiple anchors have the same IoU match all of them
  auto gt_iou_argmax = torch::argmax(overlaps, /*dim*/ 0);
  rpn_match.index_fill_(0, gt_iou_argmax, 1);
  // 3. Set anchors with high overlap as positive.
  rpn_match = torch::where(anchor_iou_max >= config.anchor_iou_max_threshold,
                           one, rpn_match);

  // Subsample to balance positive and negative anchors
  // Don't let positives be more than half the anchors
  auto ids = (rpn_match == 1).nonzero().narrow(1, 0, 1);  // take first column
  auto extra = ids.size(0) - (config.rpn_train_anchors_per_image / 2);
  if (extra > 0) {
    // Reset the extra ones to neutral
    auto idx = torch::randperm(ids.size(0), at::dtype(at::kLong));
    idx = idx.narrow(0, 0, extra);
    ids = ids.take(idx);  // random::choice(ids, extra, replace=False)
    rpn_match.index_fill_(0, ids, 0);
  }
  // Same for negative proposals
  auto positives_num = torch::sum(rpn_match == 1).item<int32_t>();
  // or: auto positives_num = ids.size(0);
  ids = (rpn_match == -1).nonzero().narrow(1, 0, 1);
  extra = ids.size(0) - (config.rpn_train_anchors_per_image - positives_num);
  if (extra > 0) {
    // Rest the extra ones to neutral
    auto idx = torch::randperm(ids.size(0), at::dtype(at::kLong));
    idx = idx.narrow(0, 0, extra);
    ids = ids.take(idx);  // random::choice(ids, extra, replace=False)
    rpn_match.index_fill_(0, ids, 0);
  }

  // For positive anchors, compute shift and scale needed to transform them
  // to match the corresponding GT boxes.
  ids = (rpn_match == 1).nonzero().narrow(1, 0, 1).squeeze();
  auto gt = gt_boxes.index_select(0, anchor_iou_argmax.take(ids));
  auto a = anchors.index_select(0, ids);
  rpn_bbox.index_put_({torch::arange(0, ids.numel(), at::kLong)},
                      BoxRefinement(a, gt));

  // Normalize
  auto std_dev = torch::tensor(config.rpn_bbox_std_dev,
                               at::dtype(at::kFloat).requires_grad(false));
  rpn_bbox /= std_dev;

  return std::make_tuple(rpn_match, rpn_bbox);
}
}  // namespace

CocoDataset::CocoDataset(std::shared_ptr<CocoLoader> loader,
                         std::shared_ptr<const Config> config)
    : loader_(loader), config_(config) {
  loader_->LoadData(GetDatasetClasses());
  // train only on vehicles
  // loader_->LoadData(GetDatasetClasses(), {2, 3, 4, 6, 7});

  anchors_ = GeneratePyramidAnchors(
      config_->rpn_anchor_scales, config_->rpn_anchor_ratios,
      config_->backbone_shapes, config_->backbone_strides,
      config_->rpn_anchor_stride);
}

Sample CocoDataset::get(size_t index) {
  auto img_desc = loader_->GetImage(index);
  auto img_width = img_desc.image.cols;
  auto img_height = img_desc.image.rows;
  auto ret_tuple1 = ResizeImage(img_desc.image, config_->image_min_dim,
                  config_->image_max_dim, config_->image_padding);
  auto image = std::get<0>(ret_tuple1);
  auto window = std::get<1>(ret_tuple1);
  auto scale = std::get<2>(ret_tuple1);
  auto padding = std::get<3>(ret_tuple1);

  auto masks = ResizeMasks(img_desc.masks, scale, padding);

  // Resize and format boxes -> [y1,x1,y2,x2]
  std::vector<float> boxes;
  boxes.reserve(img_desc.boxes.size() * 4);
  for (auto bbox : img_desc.boxes) {
    boxes.push_back(padding.top_pad + std::ceil(bbox.y * scale));
    boxes.push_back(padding.left_pad + std::ceil(bbox.x * scale));
    boxes.push_back(padding.top_pad +
                    std::ceil((bbox.y + bbox.height) * scale));
    boxes.push_back(padding.left_pad +
                    std::ceil((bbox.x + bbox.width) * scale));
  }

  // Debug block
  //  if (!image.empty()) {
  //    cv::Mat img = image.clone();
  //    for (size_t i = 0; i < masks.size(); ++i) {
  //      cv::Point tl(static_cast<int32_t>(boxes[i * 4 + 1]),
  //                   static_cast<int32_t>(boxes[i * 4]));
  //      cv::Point br(static_cast<int32_t>(boxes[i * 4 + 3]),
  //                   static_cast<int32_t>(boxes[i * 4 + 2]));
  //      cv::rectangle(img, tl, br, cv::Scalar(255, 0, 0));

  //      cv::Mat mask_ch[3];
  //      mask_ch[2] = masks[i] * 255;
  //      mask_ch[0] = cv::Mat::zeros(img.size(), CV_8UC1);
  //      mask_ch[1] = cv::Mat::zeros(img.size(), CV_8UC1);
  //      cv::Mat mask;
  //      cv::merge(mask_ch, 3, mask);
  //      cv::addWeighted(img, 1, mask, 0.5, 0, img);
  //    }
  //    cv::imwrite("test_resize.png", img);
  //    exit(0);
  //  }

  // Resize masks to smaller size to reduce memory usage
  if (config_->use_mini_mask) {
    masks = MinimizeMasks(boxes, masks, config_->mini_mask_shape[0],
                          config_->mini_mask_shape[1]);
  }

  // Make training sample
  Sample result;
  img_desc.image = MoldImage(image, *config_);
  result.data.image = CvImageToTensor(img_desc.image);
  result.data.image_meta.image_id = static_cast<int32_t>(img_desc.id);
  result.data.image_meta.window = window;
  result.data.image_meta.image_width = img_width;
  result.data.image_meta.image_height = img_height;

  std::vector<at::Tensor> tmasks;
  for (auto& m : masks) {
    auto mask = CvImageToTensor(m) != 0;
    tmasks.push_back(mask.to(at::kFloat));
  }
  result.target.gt_masks = torch::stack(tmasks);

  int32_t annotations_num = static_cast<int32_t>(img_desc.boxes.size());
  result.target.gt_boxes = torch::tensor(boxes, at::dtype(at::kFloat))
                               .reshape({annotations_num, 4})
                               .clone();
  result.target.gt_class_ids =
      torch::tensor(img_desc.classes, at::dtype(at::kInt)).clone();

  // RPN Targets
  auto ret_tuple2 = BuildRpnTargets(anchors_, result.target.gt_boxes, *config_);
  auto rpn_match = std::get<0>(ret_tuple2);
  auto rpn_bbox = std::get<1>(ret_tuple2);
      
  // If more instances than fits in the array, sub-sample from them.
  if (result.target.gt_boxes.size(0) > config_->max_gt_instances) {
    auto ids = torch::randperm(result.target.gt_boxes.size(0), at::kLong);
    ids = ids.narrow(0, 0, config_->max_gt_instances);
    result.target.gt_class_ids =
        result.target.gt_class_ids.index_select(0, ids);
    result.target.gt_boxes = result.target.gt_boxes.index_select(0, ids);
    result.target.gt_masks = result.target.gt_masks.index_select(0, ids);
  }
  result.target.rpn_match = rpn_match.unsqueeze(1);
  result.target.rpn_bbox = rpn_bbox;

  // Add batch dim
  result.data.image = result.data.image.unsqueeze(0);
  result.target.rpn_match = result.target.rpn_match.unsqueeze(0);
  result.target.rpn_bbox = result.target.rpn_bbox.unsqueeze(0);
  result.target.gt_class_ids = result.target.gt_class_ids.unsqueeze(0);
  result.target.gt_boxes = result.target.gt_boxes.unsqueeze(0);
  result.target.gt_masks = result.target.gt_masks.unsqueeze(0);

  return result;
}

torch::optional<size_t> CocoDataset::size() const {
  return loader_->GetImagesCount();
}
