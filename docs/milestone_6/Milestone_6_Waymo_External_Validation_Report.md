# Milestone 6 — Waymo External Validation and Generalization Analysis

## 1. Milestone 6 Objective

Evaluate how well the KITTI-trained detectors generalize to Waymo without retraining, by applying the four locked checkpoints directly to the frozen Waymo representative subset and measuring domain-shift degradation against the Milestone 4/5 KITTI validation baseline.

## 2. External Validation Methodology

The Waymo evaluation reuses the frozen Milestone 4/5 evaluation stack (adapters + pycocotools COCOeval). Input size 640x640, AP curve confidence 0.001, operating point confidence 0.25 / IoU 0.50, no threshold tuning on Waymo, and no DontCare-style ignore regions (Sign excluded).

## 3. Dataset and Waymo Subset Description

- Images: 996
- Target boxes: 24819
- Source: Waymo Open Dataset validation split, FRONT camera, every 5th frame, 25 segments.

## 4. No-Retraining Policy

No training/fine-tuning/hyperparameter/checkpoint/threshold selection uses Waymo data.

## 5. Model Checkpoints Used

- YOLOv8s: D:\MCs\advances object detection and recognition\project\AV-crossDataset-ObjectDetection\outputs\milestone_4\checkpoints\yolo\final\best.pt
- RT-DETR-L: D:\MCs\advances object detection and recognition\project\AV-crossDataset-ObjectDetection\outputs\milestone_4\checkpoints\rtdetr\final\best.pt
- RetinaNet: D:\MCs\advances object detection and recognition\project\AV-crossDataset-ObjectDetection\outputs\milestone_4\checkpoints\retinanet\final\best.pt
- Faster R-CNN: D:\MCs\advances object detection and recognition\project\AV-crossDataset-ObjectDetection\outputs\milestone_4\checkpoints\faster_rcnn\final\best.pt

## 6. Class Mapping

| COCO id | YOLO id | Torchvision label | Name |
|---|---|---|---|
| 1 | 0 | 1 | Vehicle |
| 2 | 1 | 2 | Pedestrian |
| 3 | 2 | 3 | Cyclist |

## 7. Evaluation Metrics

mAP@0.50, mAP@[0.50:0.95], per-class AP50/AP50-95, mean inference time.

## 8. KITTI Baseline Summary

| Detector | mAP@0.50 | mAP@0.50:0.95 | Vehicle AP | Pedestrian AP | Cyclist AP |
|---|---|---|---|---|---|
| YOLOv8s | 0.9248 | 0.6899 | 0.8489 | 0.5298 | 0.6911 |
| RT-DETR-L | 0.9058 | 0.6057 | 0.7304 | 0.4901 | 0.5967 |
| RetinaNet | 0.8390 | 0.5375 | 0.7032 | 0.3915 | 0.5180 |
| Faster R-CNN | 0.8790 | 0.5892 | 0.7414 | 0.4317 | 0.5944 |

## 9. Waymo External Validation Results

| Detector | mAP@0.50 | mAP@0.50:0.95 | Vehicle AP | Pedestrian AP | Cyclist AP | Mean inf. (ms) |
|---|---|---|---|---|---|---|
| YOLOv8s | 0.1420 | 0.0654 | 0.1056 | 0.0491 | 0.0416 | 34.21 |
| RT-DETR-L | 0.1509 | 0.0714 | 0.1156 | 0.0578 | 0.0409 | 62.57 |
| RetinaNet | 0.2145 | 0.0969 | 0.1439 | 0.0746 | 0.0721 | 77.50 |
| Faster R-CNN | 0.1748 | 0.0771 | 0.1226 | 0.0702 | 0.0384 | 92.07 |

## 10. KITTI vs Waymo Comparison

| Detector | KITTI mAP@0.50:0.95 | Waymo mAP@0.50:0.95 | Absolute drop | Drop % | Ratio | Rank |
|---|---|---|---|---|---|---|
| YOLOv8s | 0.6899 | 0.0654 | 0.6245 | 90.52 | 0.0948 | 4 |
| RT-DETR-L | 0.6057 | 0.0714 | 0.5343 | 88.21 | 0.1179 | 3 |
| RetinaNet | 0.5375 | 0.0969 | 0.4407 | 81.98 | 0.1802 | 1 |
| Faster R-CNN | 0.5892 | 0.0771 | 0.5121 | 86.92 | 0.1308 | 2 |

## 11. Generalization Ratio Analysis

- Best generalizing detector: **RetinaNet**
- Worst generalizing detector: **YOLOv8s**

## 12. Class-Wise Degradation Analysis

- Largest degraded class: **Vehicle**

## 13. Domain-Shift Discussion

All detectors degrade sharply (80-90% relative mAP@0.50:0.95 drop). The in-domain detector ranking inverts on Waymo: YOLOv8s is strongest on KITTI but least generalizing, while RetinaNet is weaker in-domain yet most stable out-of-domain, suggesting domain overfitting in the higher-capacity detectors.

## 14. Paper-Ready External Validation Subsection

Cross-dataset generalization was assessed by applying the four KITTI-trained detectors directly to a frozen Waymo representative subset (996 front-camera images) with no retraining, fine-tuning, or threshold tuning. All detectors degrade sharply on Waymo: mAP@[0.50:0.95] drops from 0.538-0.690 (KITTI) to 0.065-0.097 (Waymo), corresponding to generalization ratios of 0.095-0.180. RetinaNet is the most domain-stable detector (ratio 0.180) and YOLOv8s the least (ratio 0.095). Vehicle is the class with the largest degradation across detectors. These results indicate that the in-domain ranking does not transfer to the out-of-domain setting, motivating domain-robust training or adaptation strategies.

## 15. Limitations and Threats to Validity

- Waymo subset (996) is smaller than KITTI validation (1,496).
- Inference time includes first-image warmup and is not comparable to Milestone 4 dummy-input latency.
- Operating point uses the frozen 0.25 confidence threshold (not optimized on Waymo).
- Results are specific to the locked checkpoints and frozen subset.

## 16. Generated Artifacts

- Handoff validation: outputs/milestone_6/handoff_validation/
- Waymo metrics: outputs/milestone_6/waymo_external_validation/metrics/
- Comparison tables: outputs/milestone_6/generalization_analysis/tables/
- Figures: outputs/milestone_6/figures/
- Final audit: outputs/milestone_6/final_audit/
