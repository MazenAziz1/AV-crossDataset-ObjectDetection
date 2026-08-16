# Evidence Map — Raw Results to Paper Tables and Figures

This document records the reproducible dependency chain from validated project artifacts
to the tables and figures used in `main.tex`. Regenerating the paper after any result
change is a matter of re-running the two generation scripts listed below and recompiling
LaTeX; no value is copied by hand.

Generation scripts:

- `scripts/final_paper/generate_tables.py`  -> `docs/final_paper/tables/*.tex`
- `scripts/final_paper/generate_figures.py` -> `docs/final_paper/figures/*.pdf`

## Tables

| Paper table (`\label{}`) | Generated file | Source artifact(s) |
|---|---|---|
| `tab:dataset_comparison` | `tables/tab_dataset_comparison.tex` | `docs/milestone_2/dataset_comparison_table.csv` |
| `tab:class_mapping` | `tables/tab_class_mapping.tex` | `docs/milestone_2/class_mapping_table.csv` |
| `tab:kitti_results` | `tables/tab_kitti_results.tex` | `outputs/milestone_5/figures/accuracy_comparison.csv` |
| `tab:efficiency` | `tables/tab_efficiency.tex` | `outputs/milestone_5/figures/efficiency_comparison.csv` |
| `tab:operating_point` | `tables/tab_operating_point.tex` | `outputs/milestone_5/figures/operating_point_comparison.csv` |
| `tab:waymo_results` | `tables/tab_waymo_results.tex` | `outputs/milestone_6/waymo_external_validation/tables/waymo_external_summary.csv` |
| `tab:kitti_vs_waymo` | `tables/tab_kitti_vs_waymo.tex` | `outputs/milestone_6/generalization_analysis/tables/kitti_vs_waymo_comparison.csv` |
| `tab:class_wise_degradation` | `tables/tab_class_wise_degradation.tex` | `outputs/milestone_6/generalization_analysis/tables/class_wise_degradation.csv` |
| `tab:object_size_recall` | `tables/tab_object_size_recall.tex` | `outputs/milestone_7/object_size_analysis/object_size_summary.csv` |
| `tab:safety_fn` | `tables/tab_safety_fn.tex` | `outputs/milestone_7/safety_error_analysis/safety_false_negative_summary.csv` |
| `tab:failure_type` | `tables/tab_failure_type.tex` | `outputs/milestone_7/safety_error_analysis/failure_type_summary.csv` |
| `tab:deployment` | `tables/tab_deployment.tex` | `outputs/milestone_7/deployment_tradeoff/deployment_suitability_table.csv` |

### Derivation notes

- `tab:kitti_results`, `tab:efficiency`, `tab:operating_point` reuse the milestone-5
  comparison CSVs (`scripts/milestone_5/create_comparison_outputs.py`), which in turn are
  derived from the raw per-model metric JSONs under
  `outputs/milestone_5/metrics/kitti_validation/`.
- `tab:waymo_results`, `tab:kitti_vs_waymo`, `tab:class_wise_degradation` reuse the
  milestone-6 summary CSVs (`scripts/milestone_6/03_run_waymo_external_validation.py` and
  `scripts/milestone_6/04_create_generalization_analysis.py`), derived from
  `outputs/milestone_6/waymo_external_validation/metrics/*.json` and the milestone-5 KITTI
  baselines.
- `tab:object_size_recall` filters `object_size_summary.csv` to `class_name == "all"`.
- `tab:safety_fn` filters `safety_false_negative_summary.csv` to
  `class_name == "pedestrian+cyclist"` and aggregates `tp`/`fn` over size categories to
  compute the overall false-negative rate.
- `tab:failure_type` reports the milestone-7 failure-type decomposition verbatim.
- `tab:deployment` reports the milestone-7 deployment-suitability summary verbatim. Note
  that its `Waymo_small_recall` column is computed in the source script over the
  pedestrian/cyclist (vulnerable-road-user) subset only, so it is labeled ``Waymo VRU small
  recall`` in the paper to distinguish it from the class-agnostic small-object recall in
  `tab:object_size_recall`.

## Figures

| Paper figure (`\label{}`) | Generated file | Source artifact(s) |
|---|---|---|
| `fig:kitti_vs_waymo` | `figures/kitti_vs_waymo_map50_95.pdf` | `outputs/milestone_6/generalization_analysis/tables/kitti_vs_waymo_comparison.csv` |
| `fig:generalization_ratio` | `figures/generalization_ratio_map50_95.pdf` | same CSV |
| `fig:class_wise_degradation` | `figures/class_wise_degradation.pdf` | `outputs/milestone_6/generalization_analysis/tables/class_wise_degradation.csv` |
| `fig:object_size_recall` | `figures/object_size_recall.pdf` | `outputs/milestone_7/object_size_analysis/object_size_summary.csv` |
| `fig:pedestrian_cyclist_fn` | `figures/pedestrian_cyclist_fn_rate.pdf` | `outputs/milestone_7/safety_error_analysis/safety_false_negative_summary.csv` |
| `fig:deployment_tradeoff` | `figures/deployment_tradeoff.pdf` | `outputs/milestone_7/deployment_tradeoff/deployment_suitability_table.csv` |
| `fig:pipeline` | `figures/pipeline.png` | copied from `docs/milestone_3/Unified Preprocessing and Validation Pipeline remake.png` |
| `fig:partition_roles` | `figures/partition_roles.png` | copied from `docs/milestone_3/Experimental roles and leakage prevention remake.png` |

The data figures reuse the same plotting logic as the milestone figure scripts
(`scripts/milestone_6/05_create_waymo_generalization_figures.py` and
`scripts/milestone_7/08_create_milestone_7_figures.py`) but emit vector PDFs into
`docs/final_paper/figures/`. The two pipeline diagrams are pre-existing PNG artifacts
(no vector source exists in the repository); they are copied verbatim.

## Underlying raw results (leaf artifacts)

- KITTI in-domain metrics: `outputs/milestone_5/metrics/kitti_validation/{yolo,rtdetr,retinanet,faster_rcnn}_metrics.json`
- KITTI benchmarks: `outputs/milestone_5/benchmarks/*_benchmark.json`
- Waymo external metrics: `outputs/milestone_6/waymo_external_validation/metrics/*_waymo_metrics.json`
- Robustness/safety: `outputs/milestone_7/{object_size_analysis,safety_error_analysis,deployment_tradeoff}/*.csv`
- Checkpoint registry: `outputs/milestone_4/manifests/final_checkpoint_registry.csv`
- Dataset harmonization: `docs/milestone_2/{dataset_comparison_table,class_mapping_table}.csv`

## Regeneration workflow

```bash
# 1. Regenerate tables and figures from the current artifacts.
python scripts/final_paper/generate_tables.py
python scripts/final_paper/generate_figures.py

# 2. Recompile the paper.
cd docs/final_paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
