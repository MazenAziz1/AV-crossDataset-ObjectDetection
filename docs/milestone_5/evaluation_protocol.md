# Milestone 4 + 5: KITTI Evaluation Protocol
**Status**: `FROZEN`
**Date Frozen**: 2026-06-24

---

## 1. Primary Metrics (COCO mAP)
To align with standard benchmark metrics, the primary evaluation metric is the COCO Average Precision (AP):
* **mAP@[0.50:0.95]**: The mean AP computed over 10 IoU thresholds from $0.50$ to $0.95$ with a step of $0.05$.
* **AP50**: Mean AP at $0.50$ IoU threshold.
* **AP75**: Mean AP at $0.57$ IoU threshold (strict matching).
* **Per-class AP**: Calculated for the three target classes: `Vehicle`, `Pedestrian`, and `Cyclist`.

---

## 2. Operating Point Metrics
While AP evaluates the model across all confidence levels, we select a fixed operating point to compute practical metrics for visual validation:
* **Operating Point**: Confidence Threshold $\ge 0.25$ and IoU $\ge 0.50$.
* **Metrics Computed**:
  - **Precision**: $\frac{TP}{TP + FP}$
  - **Recall**: $\frac{TP}{TP + FN}$
  - **F1-Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$
  - **Detections per Image**: Average number of predicted boxes per image.
  - **False Positives per Image**: Average number of false detections per image.

---

## 3. Scale-Specific Metrics
Objects are categorized into sizes based on their area in pixels, following the COCO definition:
* **Small**: Area $< 32^2$ pixels ($< 1024$ pixels).
* **Medium**: Area between $32^2$ and $96^2$ pixels ($1024$ to $9216$ pixels).
* **Large**: Area $> 96^2$ pixels ($> 9216$ pixels).

---

## 4. KITTI-Specific Ignore Rules & Suppression
In autonomous driving validation on the KITTI dataset, we must handle specific background regions and non-target classes to prevent penalizing the models:

### A. DontCare Ignore Regions
KITTI annotations contain `DontCare` bounding boxes. These mark regions containing targets that are too far away or heavily occluded.
* **Suppression Logic**: If a predicted box does not match any ground-truth target but has an IoU overlap $\ge 0.50$ with a `DontCare` region of the *same class family* (e.g. vehicle-like prediction overlapping with a vehicle DontCare), the prediction is **suppressed**.
* **Effect**: It is neither counted as a True Positive (TP) nor as a False Positive (FP). It is excluded from the denominator of Precision.

### B. Excluded Non-Target Classes (Tram, Misc)
KITTI contains objects labeled as `Tram` or `Misc`.
* **Policy**: These are **excluded** from the main targets. Detections matching these non-target classes are not evaluated, and detections matching target classes that overlap with them are **not** automatically suppressed (unless they fall within a registered `DontCare` region).
