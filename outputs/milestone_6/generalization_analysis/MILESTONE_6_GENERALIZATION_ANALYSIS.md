# Milestone 6 - KITTI vs Waymo Generalization Analysis

Created at: `2026-08-15T00:30:02`

## Objective

This analysis compares Milestone 5 KITTI in-domain validation results against Milestone 6 Waymo external-validation results using the same locked KITTI-trained checkpoints. No retraining or fine-tuning is used.

## Formula

`Generalization Ratio = Waymo metric / KITTI metric`

A ratio closer to `1.0` indicates better cross-dataset stability.

## Aggregate KITTI vs Waymo Comparison

| detector    |   rank_by_mAP50_95_generalization |   KITTI_num_images |   Waymo_num_images |   KITTI_mAP50 |   Waymo_mAP50 |   mAP50_absolute_drop |   mAP50_drop_percent |   mAP50_generalization_ratio |   KITTI_mAP50_95 |   Waymo_mAP50_95 |   mAP50_95_absolute_drop |   mAP50_95_drop_percent |   mAP50_95_generalization_ratio |   KITTI_mean_inference_ms |   Waymo_mean_inference_ms |   inference_ms_difference_Waymo_minus_KITTI |   Waymo_to_KITTI_inference_ratio |
|:------------|----------------------------------:|-------------------:|-------------------:|--------------:|--------------:|----------------------:|---------------------:|-----------------------------:|-----------------:|-----------------:|-------------------------:|------------------------:|--------------------------------:|--------------------------:|--------------------------:|--------------------------------------------:|---------------------------------:|
| retinanet   |                                 1 |               1496 |                996 |      0.807834 |      0.168049 |              0.639785 |              79.1975 |                     0.208025 |         0.508122 |         0.078764 |                 0.429358 |                 84.4989 |                        0.155011 |                   54.7681 |                   51.9363 |                                   -2.83183  |                         0.948294 |
| rtdetr      |                                 2 |               1496 |                996 |      0.921312 |      0.166143 |              0.755169 |              81.9667 |                     0.180333 |         0.631434 |         0.076835 |                 0.554599 |                 87.8316 |                        0.121684 |                   43.4884 |                   48.5091 |                                    5.0207   |                         1.11545  |
| faster_rcnn |                                 3 |               1496 |                996 |      0.860892 |      0.12197  |              0.738922 |              85.8321 |                     0.141679 |         0.539631 |         0.054493 |                 0.485138 |                 89.9018 |                        0.100982 |                   59.0178 |                   58.8757 |                                   -0.142026 |                         0.997593 |
| yolo        |                                 4 |               1496 |                996 |      0.920643 |      0.109444 |              0.8112   |              88.1122 |                     0.118878 |         0.691006 |         0.053924 |                 0.637082 |                 92.1964 |                        0.078036 |                   15.9886 |                   20.0261 |                                    4.03748  |                         1.25252  |

## Generalization Ratio Table

|   rank_by_mAP50_95_generalization | detector    |   KITTI_mAP50_95 |   Waymo_mAP50_95 |   mAP50_95_absolute_drop |   mAP50_95_drop_percent |   mAP50_95_generalization_ratio |   KITTI_mAP50 |   Waymo_mAP50 |   mAP50_absolute_drop |   mAP50_drop_percent |   mAP50_generalization_ratio |
|----------------------------------:|:------------|-----------------:|-----------------:|-------------------------:|------------------------:|--------------------------------:|--------------:|--------------:|----------------------:|---------------------:|-----------------------------:|
|                                 1 | retinanet   |         0.508122 |         0.078764 |                 0.429358 |                 84.4989 |                        0.155011 |      0.807834 |      0.168049 |              0.639785 |              79.1975 |                     0.208025 |
|                                 2 | rtdetr      |         0.631434 |         0.076835 |                 0.554599 |                 87.8316 |                        0.121684 |      0.921312 |      0.166143 |              0.755169 |              81.9667 |                     0.180333 |
|                                 3 | faster_rcnn |         0.539631 |         0.054493 |                 0.485138 |                 89.9018 |                        0.100982 |      0.860892 |      0.12197  |              0.738922 |              85.8321 |                     0.141679 |
|                                 4 | yolo        |         0.691006 |         0.053924 |                 0.637082 |                 92.1964 |                        0.078036 |      0.920643 |      0.109444 |              0.8112   |              88.1122 |                     0.118878 |

## Largest Class-Level Degradation by Detector

| detector    | largest_absolute_degradation_class   |   largest_absolute_degradation |   largest_absolute_degradation_percent | lowest_generalization_ratio_class   |   lowest_class_generalization_ratio |
|:------------|:-------------------------------------|-------------------------------:|---------------------------------------:|:------------------------------------|------------------------------------:|
| faster_rcnn | Vehicle                              |                       0.626474 |                                87.2464 | Cyclist                             |                            0.050803 |
| retinanet   | Vehicle                              |                       0.575204 |                                83.2078 | Cyclist                             |                            0.123671 |
| rtdetr      | Vehicle                              |                       0.645308 |                                84.1067 | Cyclist                             |                            0.067613 |
| yolo        | Vehicle                              |                       0.755319 |                                88.6996 | Cyclist                             |                            0.040838 |

## Key Findings

- Best generalizer by mAP50-95 ratio: `retinanet` with ratio `0.155011`.
- Best Waymo detector by mAP50-95: `retinanet` with Waymo mAP50-95 `0.078764`.
- Fastest Waymo detector: `yolo` with mean inference `20.026075` ms.

## Interpretation Draft

All detectors show substantial degradation when transferred from KITTI to the Waymo external subset. This confirms a significant domain shift between the in-domain KITTI validation split and the external Waymo subset. The observed drop should be discussed as the central cross-dataset generalization result of Milestone 6 rather than as a training failure, because the evaluation uses locked KITTI-trained checkpoints without retraining.
