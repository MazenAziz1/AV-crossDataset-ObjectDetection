# Milestone 5 Benchmark Comparison

## Final KITTI validation summary

| Detector | mAP50 | mAP50-95 | Mean inference ms/image |
|---|---:|---:|---:|
| yolo | 0.9206 | 0.6910 | 15.99 |
| rtdetr | 0.9213 | 0.6314 | 43.49 |
| faster_rcnn | 0.8609 | 0.5396 | 59.02 |
| retinanet | 0.8078 | 0.5081 | 54.77 |

## Key findings

- Best mAP50-95: **yolo** with **0.6910**.
- Best mAP50: **rtdetr** with **0.9213**.
- Fastest detector: **yolo** with **15.99 ms/image**.
- Best combined accuracy-speed score: **yolo**.

## Interpretation

YOLO is the strongest overall model because it achieves the highest mAP50-95 while also being the fastest detector.
RT-DETR achieves the highest mAP50 by a very small margin, but YOLO performs better under stricter IoU thresholds and has much lower inference latency.
Faster R-CNN performs better than RetinaNet in both mAP50 and mAP50-95, but it is also the slowest model in the local benchmark.

## Output files

- `tables/detector_ranking_full.csv`
- `tables/class_level_ap50_95_full.csv`
- `figures/map50_95_comparison.png`
- `figures/map50_comparison.png`
- `figures/inference_time_comparison.png`