# Milestone 7 Config Manifest

Created at: `2026-08-15T12:25:58`

Status: **PASSED**

## Created Files

- `configs\analysis\milestone_7\failure_case_policy.yaml`
- `configs\analysis\milestone_7\safety_error_policy.yaml`
- `configs\analysis\milestone_7\object_size_bins.yaml`
- `configs\analysis\milestone_7\README.md`

## Key Analysis Rules

- Primary IoU threshold: `0.50`
- Localization error IoU range: `0.10 <= IoU < 0.50`
- Class confusion IoU minimum: `0.30`
- Priority safety classes: `Pedestrian`, `Cyclist`
- Method: `target_box_normalized_area_quantiles`
- Normalized area: `bbox_area / image_area`
- Small objects: `normalized_area <= 0.004157071390230814`
- Medium objects: `0.004157071390230814 < normalized_area <= 0.015589650670960799`
- Large objects: `normalized_area > 0.015589650670960799`
- Target boxes used to define bins: `39086`
