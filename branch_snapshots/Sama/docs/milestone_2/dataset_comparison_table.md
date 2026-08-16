# KITTI–Waymo Dataset Comparison

KITTI provides training and in-domain validation. Waymo provides external validation without retraining.

| Metric | KITTI all | KITTI train | KITTI validation | Waymo external validation | Notes |
| --- | --- | --- | --- | --- | --- |
| Experimental role | Source labeled benchmark | Model training | In-domain validation | External validation only | Waymo is not used for training, fine-tuning, hyperparameter selection, or model selection. |
| Source split | Official labeled training set | Project train split | Project validation split | Official Waymo validation split | KITTI official testing labels are not publicly available. |
| Images | 7481 | 5985 | 1496 | 996 |  |
| Driving segments |  |  |  | 25 | KITTI object-detection images are treated as independent benchmark samples. |
| Target boxes | 39086 | 31294 | 7792 | 24819 | Target boxes include only Vehicle, Pedestrian, and Cyclist. |
| Vehicle boxes | 32750 | 26278 | 6472 | 16928 |  |
| Pedestrian boxes | 4709 | 3729 | 980 | 7127 |  |
| Cyclist boxes | 1627 | 1287 | 340 | 764 |  |
| Ignored boxes | 12779 | 10174 | 2605 |  | KITTI ignored boxes include Tram, Misc, and DontCare under the harmonized task. Waymo Sign annotations are excluded before the representative boxes table is created. |
| Images with no target boxes | 0 | 0 | 0 | 12 | Negative images remain available for false-positive analysis. |
| Images containing Vehicle | 6798 | 5438 | 1360 | 977 | Counts images containing at least one Vehicle target box. |
| Images containing Pedestrian | 1796 | 1437 | 359 | 705 | Counts images containing at least one Pedestrian target box. |
| Images containing Cyclist | 1141 | 913 | 228 | 456 | Counts images containing at least one Cyclist target box. |
| Camera | Left color camera (image_2) | Left color camera (image_2) | Left color camera (image_2) | FRONT | Both datasets use a forward-facing monocular RGB view. |
| Sampling rule | Use all official labeled images | Frozen stratified split | Frozen stratified split | every_5th_front_frame_starting_at_first | Waymo temporal redundancy is reduced through uniform frame sampling. |
| Random seed |  | 42 | 42 |  | The KITTI split was created once and frozen before model training. |
| Selected segments by time of day | Not provided as official image metadata |  |  | {"Dawn/Dusk": 6, "Day": 14, "Night": 5} | Waymo values are counts of selected driving segments, not image counts. |
| Selected segments by location | KITTI collection area; no comparable per-image location metadata used |  |  | {"location_other": 2, "location_phx": 11, "location_sf": 12} | Waymo values are counts of selected driving segments. |
| Selected segments by weather | Not provided as official image metadata |  |  | {"rain": 1, "sunny": 24} | Waymo values are segment counts. The available validation candidate pool contained only one rainy segment. |
