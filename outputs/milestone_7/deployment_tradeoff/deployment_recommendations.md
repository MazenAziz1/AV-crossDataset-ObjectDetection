# Milestone 7 - Deployment Suitability Recommendations

These recommendations are derived from the frozen M5/M6 results and the Milestone 7
error index. They are stated cautiously; no detector is declared deployment-ready on
the basis of this analysis alone.

## Observations

- Fastest inference (Waymo): **YOLOv8s**.
- Highest KITTI accuracy: **YOLOv8s**.
- Highest Waymo generalization: **RetinaNet**.
- Lowest Waymo safety false-negative rate: **RetinaNet**.

## Interpretation

- The in-domain accuracy ranking does not transfer to the out-of-domain setting: the
  strongest KITTI detector generalizes worst to Waymo.
- All detectors miss the large majority of small Waymo objects (pedestrians/cyclists in
  particular), which is the dominant safety risk in cross-dataset deployment.
- No single detector simultaneously optimizes accuracy, generalization, latency, and
  safety; the choice of detector depends on the deployment priority.

## Deployment-table columns

- KITTI_mAP50_95 / Waymo_mAP50_95: COCO mAP@[0.50:0.95].
- waymo_mean_inference_ms: per-image latency measured on real Waymo images.
- safety_fn_rate: pedestrian+cyclist false-negative rate at the operating point.
- small_recall: recall on small objects (<32^2 px).
