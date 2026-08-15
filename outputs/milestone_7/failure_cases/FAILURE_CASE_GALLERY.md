# Milestone 7 Failure-Case Gallery

Reproducible annotated examples of detector failures, selected by rule (not manually):
- missed pedestrian / cyclist: largest-area false negative
- small-object failure: largest-area small pedestrian false negative
- false positive / localization / confusion: highest-confidence occurrence

Annotation convention: green = all ground-truth boxes; red = missed object; blue = erroneous detection.

| Dataset | Detector | Error type | Image |
|---|---|---|---|
| kitti | faster_rcnn | missed_pedestrian | kitti_faster_rcnn_missed_pedestrian.png |
| kitti | faster_rcnn | missed_cyclist | kitti_faster_rcnn_missed_cyclist.png |
| kitti | faster_rcnn | small_object_failure | kitti_faster_rcnn_small_object_failure.png |
| kitti | retinanet | false_positive | kitti_retinanet_false_positive.png |
| kitti | retinanet | localization_error | kitti_retinanet_localization_error.png |
| kitti | retinanet | class_confusion | kitti_retinanet_class_confusion.png |
| waymo | rtdetr | missed_pedestrian | waymo_rtdetr_missed_pedestrian.png |
| waymo | rtdetr | missed_cyclist | waymo_rtdetr_missed_cyclist.png |
| waymo | yolo | small_object_failure | waymo_yolo_small_object_failure.png |
| waymo | retinanet | false_positive | waymo_retinanet_false_positive.png |
| waymo | retinanet | localization_error | waymo_retinanet_localization_error.png |
| waymo | faster_rcnn | class_confusion | waymo_faster_rcnn_class_confusion.png |
