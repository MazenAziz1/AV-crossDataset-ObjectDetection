# Milestone 1 — Research Definition and Main Contributions

## 1. Research Problem

Modern object detectors achieve strong performance on autonomous-driving benchmarks, but their results are often reported using in-domain evaluation, where the training and test data come from the same dataset distribution. This evaluation setting may hide sensitivity to dataset bias and may not reflect how reliably a detector performs when deployed in a different environment.

It remains unclear how representative detector families preserve their accuracy, class-level reliability, and computational suitability when transferred directly from one autonomous-driving dataset to another without retraining, fine-tuning, or target-domain adaptation. This uncertainty is particularly important for safety-critical classes such as pedestrians and cyclists, as well as for small, distant, occluded, and crowded objects.

This study therefore investigates the external generalization of object detectors trained on KITTI and evaluated on a representative Waymo front-camera subset. Four detector families are considered: YOLO, Faster R-CNN, RetinaNet, and RT-DETR. The evaluation combines in-domain accuracy, cross-dataset degradation, class-wise behavior, safety-oriented failure analysis, and computational efficiency under a harmonized experimental protocol.

---

## 2. Research Questions

### RQ1 — External Generalization

How effectively do object detectors trained on KITTI generalize to a representative Waymo front-camera subset without retraining, fine-tuning, or target-domain adaptation?

### RQ2 — In-Domain Performance

How do YOLO, Faster R-CNN, RetinaNet, and RT-DETR compare under a harmonized KITTI in-domain evaluation protocol?

### RQ3 — Cross-Dataset Degradation

Which detector family experiences the smallest absolute and relative performance degradation when transferred from KITTI to Waymo?

### RQ4 — Class-Wise Generalization

How does cross-dataset performance degradation differ among the Vehicle, Pedestrian, and Cyclist classes?

### RQ5 — Safety-Critical Conditions

How do the detectors perform on small, distant, occluded, and crowded objects under dataset shift?

### RQ6 — Computational Efficiency

How do the four detectors compare in terms of inference latency, throughput, parameter count, computational complexity, model size, and memory requirements?

### RQ7 — Deployment Trade-Off

Which detector provides the strongest overall balance among in-domain accuracy, external generalization, safety-oriented recall, and computational efficiency?

---

## 3. Research Hypotheses

### H1 — Cross-Dataset Performance Drop

All evaluated detectors will exhibit a statistically meaningful performance decrease when transferred from KITTI to Waymo without adaptation.

### H2 — Vulnerable Road User Degradation

Pedestrian and Cyclist detection will experience greater relative degradation than Vehicle detection because of smaller object sizes, lower class frequency, higher appearance variability, and stronger occlusion effects.

### H3 — Architecture-Dependent Generalization

The four detector families will exhibit significantly different cross-dataset generalization behavior under the same training and evaluation protocol.

### H4 — In-Domain Accuracy Is Not Sufficient

The detector with the highest KITTI validation accuracy will not necessarily achieve the highest Waymo accuracy, the best generalization ratio, or the smallest relative performance drop.

### H5 — Accuracy–Efficiency Trade-Off

No single detector will dominate simultaneously in detection accuracy, cross-dataset robustness, safety-oriented recall, and computational efficiency.

### H6 — Difficult-Object Sensitivity

Small, distant, and occluded objects will account for a disproportionately large share of false negatives during external validation.

---

## 4. Main Contributions

The main contributions of this study are as follows:

1. **Controlled Cross-Family Detector Comparison**  
   A harmonized benchmark is developed for four representative object-detection paradigms: a real-time one-stage CNN detector (Yolo), a two-stage region-proposal detector (Faster R-CNN), a dense focal-loss detector (RetinaNet), and a real-time transformer detector (RT-DETR).

2. **Target-Free External Validation**  
   Models are trained exclusively on KITTI and evaluated directly on a representative Waymo front-camera subset without retraining, fine-tuning, domain adaptation, or target-dependent hyperparameter selection.

3. **Dataset and Class Harmonization**  
   A transparent class-mapping procedure is established to align KITTI and Waymo annotations into the common classes Vehicle, Pedestrian, and Cyclist.

4. **Cross-Dataset Generalization Analysis**  
   Generalization is evaluated using in-domain and external mAP, absolute performance degradation, relative performance degradation, generalization ratio, class-wise AP changes, and confidence intervals where feasible.

5. **Safety-Oriented Failure Analysis**  
   The study examines false negatives, vulnerable road users, small and distant objects, occlusion, crowded scenes, classification errors, localization errors, and class confusion.

6. **Deployment-Oriented Efficiency Evaluation**  
   The practical suitability of each detector is assessed using inference latency, FPS, parameter count, FLOPs, model size, and memory consumption on common hardware.

7. **Reproducible Evaluation Pipeline**  
   The study provides a reproducible workflow covering dataset preprocessing, annotation conversion, training configuration, inference, and evaluation.

---

## 5. Traceability Between Gaps, Questions, and Contributions

| Research Gap | Related Research Question | Study Response | Main Contribution |
|---|---|---|---|
| Single-dataset evaluation may hide dataset bias | RQ1 | KITTI-trained models are tested directly on Waymo | Target-free external validation |
| Existing detector results are not directly comparable | RQ2 | All models follow a harmonized protocol | Controlled cross-family comparison |
| Architecture-level robustness is unclear | RQ3 | Performance drops and generalization ratios are compared | Cross-dataset generalization analysis |
| Class-specific degradation is underexplored | RQ4 | Vehicle, Pedestrian, and Cyclist are evaluated separately | Dataset and class harmonization |
| Aggregate mAP hides safety-relevant failures | RQ5 | False negatives and difficult-object cases are analyzed | Safety-oriented failure analysis |
| Accuracy and efficiency are often evaluated separately | RQ6 | Latency, FPS, FLOPs, parameters, size, and memory are measured | Deployment-oriented efficiency evaluation |
| Practical deployment recommendations are limited | RQ7 | Accuracy, robustness, safety, and efficiency are jointly compared | Deployment trade-off analysis |

---

## 6. Scope and Claim Boundaries

This study does **not** claim:

- A new object-detection architecture.
- A new loss function.
- A new domain-adaptation method.
- State-of-the-art performance unless demonstrated experimentally.
- Universal conclusions beyond the evaluated datasets, classes, models, and hardware.

The novelty of the work lies in the external-validation design, controlled cross-family comparison, safety-oriented failure analysis, and deployment-focused evaluation protocol.
