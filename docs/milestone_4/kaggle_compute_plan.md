\# Kaggle Multi-Session Compute Allocation Plan



\## Purpose



This document defines how Milestone 4 + 5 training will be distributed between the local machine and Kaggle GPU sessions.



The local machine remains the source of truth for code, configurations, packaging, imported outputs, final KITTI validation, documentation, and GitHub commits. Kaggle is used only as the GPU training environment.



\## Scope Boundary



| Item | Decision |

|---|---|

| Training location | Kaggle |

| Local machine role | Source of truth, packaging, import, evaluation, documentation, Git |

| Training data | KITTI train only |

| Internal validation data | KITTI validation only |

| External Waymo validation | Deferred to Milestone 6 |

| Local full training | Disabled by policy |

| Local final evaluation | Preferred |

| Kaggle final evaluation | Allowed only if local VRAM fails |



Waymo must not be uploaded to Kaggle or used during Milestone 4 + 5.



\## Estimated Training Times



| Detector | Estimated Full Training Time | Planned Slot |

|---|---:|---|

| YOLO | 3 hours | Slot A |

| Faster R-CNN | 5 hours | Slot A |

| RetinaNet | 4.5 hours | Slot A |

| RT-DETR | 13 hours | Slot B |



YOLO, Faster R-CNN, and RetinaNet are expected to finish in single sessions. RT-DETR is expected to require resume-based training across multiple Kaggle sessions.



\## Compute Slot Allocation



\### Slot A



Slot A is assigned to:



\- YOLO

\- Faster R-CNN

\- RetinaNet



Recommended order:



1\. YOLO

2\. Faster R-CNN

3\. RetinaNet



\### Slot B



Slot B is reserved for RT-DETR because the estimated full training time may exceed one Kaggle session.



\## Session Limit Policy



| Item | Value |

|---|---:|

| Assumed session limit | 12 hours |

| Planned safe stop | 10.5 hours |

| Checkpoint before shutdown | Yes |

| Package outputs before shutdown | Yes |

| Save session manifest | Yes |



The training runner should stop gracefully after the current epoch when the runtime guard is reached.



\## RT-DETR Resume Policy



| Item | Value |

|---|---:|

| Target epochs | 150 |

| Save checkpoint every | 5 epochs |

| Validate every | 5 epochs |

| Runtime guard | 10.5 hours |

| Resume mode | Latest checkpoint |

| Resume state file | Required |



Expected RT-DETR flow:



1\. Session 1 starts from epoch 0.

2\. Training continues until completion or runtime guard stop.

3\. Last checkpoint, best checkpoint, resume state, logs, reports, and manifests are packaged.

4\. The resume package is downloaded locally.

5\. The resume package is uploaded or attached to the next Kaggle session.

6\. Session 2 resumes from the latest checkpoint.

7\. The process repeats until the target epoch is reached.



\## Required Outputs from Kaggle



Each Kaggle slot must return:



\- checkpoints;

\- training histories;

\- training reports;

\- checkpoint manifests;

\- session manifests;

\- environment snapshot;

\- package manifest.



RT-DETR must additionally return:



\- resume state file;

\- complete resume-chain information;

\- final session-completion status.



\## Local Import Requirements



After downloading Kaggle outputs, the local machine must:



1\. Import Slot A outputs.

2\. Import Slot B outputs.

3\. Validate checkpoint hashes.

4\. Validate the RT-DETR resume chain.

5\. Lock final checkpoints.

6\. Run final KITTI validation.

7\. Run benchmarking.

8\. Generate comparison outputs.

9\. Create documentation.

10\. Commit safe files to GitHub.



\## Prohibited Actions



The following actions are not allowed in Milestone 4 + 5:



\- uploading Waymo data to Kaggle;

\- using Waymo for checkpoint selection;

\- using Waymo for hyperparameter tuning;

\- running Milestone 6 external validation before checkpoint locking;

\- committing model weights to Git;

\- committing Kaggle ZIP packages or downloaded checkpoints.



\## Completion Gate



This step is complete when:



\- `configs/models/milestone\_4/kaggle\_compute\_plan.yaml` exists;

\- `docs/milestone\_4/kaggle\_compute\_plan.md` exists;

\- both files define Slot A and Slot B;

\- RT-DETR resume policy is documented;

\- Waymo is explicitly excluded from Milestone 4 + 5;

\- local training is disabled by policy;

\- local final KITTI validation is preferred.

