# Milestone 7 Analysis Configs

Frozen analysis policy for the robustness/failure-case/safety analysis.

- `failure_case_policy.yaml` - TP/FP/FN matching rules, failure-type definitions, thresholds.
- `safety_error_policy.yaml` - safety-priority classes (Pedestrian, Cyclist) and FN-rate rules.
- `object_size_bins.yaml` - object-size bins, reused verbatim from the frozen Milestone 5
  evaluation policy (`configs/evaluation/milestone_5/evaluation_policy.yaml` scale_ranges).

These policies are locked before any result is generated.
