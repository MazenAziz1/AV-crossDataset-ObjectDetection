# Waymo Representative External-Validation Subset

This directory documents the complete, reproducible workflow used to build the **Waymo Open Dataset v2.0.1 representative validation subset** for this cross-dataset 2D object-detection study.

The models are trained and validated in-domain on KITTI. Waymo is used strictly as an external validation dataset:

- No Waymo data is used for training.
- No Waymo data is used for fine-tuning.
- No Waymo data is used for hyperparameter selection.
- The final Waymo segment list is frozen before model evaluation.
- Only the `FRONT` camera is used.
- The harmonized classes are `Vehicle`, `Pedestrian`, and `Cyclist`.
- Waymo `Sign` annotations are ignored.

---

## Final Result

| Item | Final count |
|---|---:|
| Waymo validation segments | 25 |
| FRONT-camera images | 996 |
| Total retained boxes | 24,819 |
| Vehicle boxes | 16,928 |
| Pedestrian boxes | 7,127 |
| Cyclist boxes | 764 |
| Negative images with no target boxes | 12 |
| Missing images | 0 |
| Unreadable images | 0 |
| Invalid boxes | 0 |
| Out-of-bounds boxes | 0 |
| Manifest/annotation mismatches | 0 |
| Visual verification samples | 8 |

Final diversity:

| Dimension | Distribution |
|---|---|
| Time of day | 14 Day, 6 Dawn/Dusk, 5 Night |
| Weather | 24 Sunny, 1 Rain |
| Location | 12 San Francisco, 11 Phoenix, 2 Other |
| Density | 10 High, 8 Low, 7 Medium |
| Segments containing pedestrians | 22 |
| Segments containing cyclists | 18 |

The final downloaded `camera_image` Parquet files occupy approximately **8.89 GB**.

---

## 1. Prerequisites

This workflow was run on Windows using:

- Python virtual environment: `AVenv`
- Google Cloud CLI
- Python 3.12
- `pandas`
- `numpy`
- `pyarrow`
- `Pillow`
- `scipy`

Activate the environment:

```cmd
cd <your project root>
AVenv\Scripts\activate
```

Install missing packages if needed:

```cmd
python -m pip install pandas numpy pyarrow pillow scipy
```
(ignore for now)
Verify Google Cloud CLI:

(ensure gcloud in available and running)
```cmd
gcloud --version
where gcloud
```

On Windows, the executable is commonly similar to:

```text
C:\Users\<USER>\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd
```

---

## 2. Obtain Waymo Access and Configure `gcloud`

1. Register for the Waymo Open Dataset.
2. Accept the dataset terms.
3. Wait for access to be granted.
4. Install the Google Cloud CLI.
5. Authenticate using the same Google account that has Waymo access.

Typical authentication commands:

```cmd
gcloud auth login
gcloud init
```

Confirm bucket access:

```cmd
gcloud storage ls gs://waymo_open_dataset_v_2_0_1/
```

Expected top-level entries:

```text
gs://waymo_open_dataset_v_2_0_1/training/
gs://waymo_open_dataset_v_2_0_1/validation/
gs://waymo_open_dataset_v_2_0_1/testing/
gs://waymo_open_dataset_v_2_0_1/testing_location/
```

Inspect validation components:

```cmd
gcloud storage ls gs://waymo_open_dataset_v_2_0_1/validation/
```

Components used here:

```text
validation/stats/
validation/camera_box/
validation/camera_image/
validation/camera_calibration/
```

LiDAR, segmentation, keypoint, projection, and pose components are not needed for this 2D camera-only study.

---

## 3. Initial Directory Structure

```text
data/
└── waymo/
    ├── README.md
    ├── raw/
    │   └── validation/
    │       ├── stats/
    │       ├── camera_box/
    │       │   └── candidates/
    │       ├── camera_image/
    │       │   └── final/
    │       └── camera_calibration/
    │           └── final/
    ├── selection/
    └── representative_subset/
        ├── images/
        │   └── front/
        ├── annotations/
        ├── metadata/
        └── visual_checks/
```

Create it:

```cmd
mkdir data\waymo\raw\validation\stats
mkdir data\waymo\raw\validation\camera_box\candidates
mkdir data\waymo\raw\validation\camera_image\final
mkdir data\waymo\raw\validation\camera_calibration\final
(already exists)
mkdir data\waymo\selection
mkdir data\waymo\representative_subset\images\front
mkdir data\waymo\representative_subset\annotations
mkdir data\waymo\representative_subset\metadata
mkdir data\waymo\representative_subset\visual_checks
```

Purpose:

| Folder | Purpose |
|---|---|
| `raw/` | Original downloaded Waymo Parquet files; never manually modified |
| `selection/` | Catalogs, schema reports, candidate lists, final selection, logs, summaries |
| `representative_subset/` | Final extracted FRONT-camera JPGs, labels, boxes, metadata, visual checks |

---

## 4. Workflow Overview

```text
Download all validation stats
        ↓
Inspect schema
        ↓
Build catalog of all 202 validation segments
        ↓
Select 70 diverse candidate segments
        ↓
Download camera boxes for those candidates
        ↓
Analyze real FRONT-camera object coverage
        ↓
Optimize and freeze 25 final segments
        ↓
Download images and calibration for final segments
        ↓
Inspect image schema
        ↓
Uniformly sample every fifth FRONT frame
        ↓
Extract matching Vehicle/Pedestrian/Cyclist boxes
        ↓
Validate all files and create visual checks
```

This order avoids downloading the entire Waymo image validation set and prevents selection based on model results.

---

# Reproduction Steps

## Step 1 — Download Validation Statistics (ignore)

```cmd
gcloud storage cp "gs://waymo_open_dataset_v_2_0_1/validation/stats/*.parquet" "data\waymo\raw\validation\stats\"
```

Verify:

```cmd
dir /b "data\waymo\raw\validation\stats\*.parquet" | find /c /v ""
```

Expected:

```text
202
```

The stats component includes frame-level context such as segment ID, timestamp, time of day, location, weather, and camera object counts.

---

## Step 2 — Inspect the Stats Schema (already exist)

Script:

```text
scripts/milestone_2/waymo/01_inspect_waymo_stats.py
```

Run:

```cmd
python scripts\milestone_2\waymo\01_inspect_waymo_stats.py
```

Output:

```text
data/waymo/selection/stats_schema.txt
```

Verified columns:

```text
key.segment_context_name
key.frame_timestamp_micros
[StatsComponent].time_of_day
[StatsComponent].location
[StatsComponent].weather
[StatsComponent].camera_object_counts.types
[StatsComponent].camera_object_counts.counts
```

---

## Step 3 — Build the Validation Segment Catalog (already exist)

Script:

```text
scripts/milestone_2/waymo/02_build_waymo_segment_catalog.py
```

Run:

```cmd
python scripts\milestone_2\waymo\02_build_waymo_segment_catalog.py
```

Output:

```text
data/waymo/selection/segment_catalog.csv
```

The script creates one summary row per segment containing:

- segment ID
- number of frames
- time of day
- location
- weather
- average/max object counts
- frame coverage for each class
- density group

Catalog results:

```text
Segments: 202

Time:
- Day: 160
- Dawn/Dusk: 23
- Night: 19

Weather:
- Sunny: 201
- Rain: 1

Location:
- Phoenix: 93
- San Francisco: 88
- Other: 21

Density:
- Low: 68
- Medium: 67
- High: 67
```

---

## Step 4 — Select 70 Diverse Candidate Segments (already exist)

Script:

```text
scripts/milestone_2/waymo/03_select_candidate_segments.py
```

Run:

```cmd
python scripts\milestone_2\waymo\03_select_candidate_segments.py
```

Output:

```text
data/waymo/selection/candidate_segments.csv
```

The reproducible selection uses seed `42` and combines:

- the only rainy segment
- all Night segments
- all Dawn/Dusk segments
- cyclist-rich segments
- pedestrian-rich segments
- daytime segments balanced by location and density

Candidate distribution:

```text
Total: 70

Time:
- Day: 28
- Dawn/Dusk: 23
- Night: 19

Weather:
- Sunny: 69
- Rain: 1

Location:
- San Francisco: 33
- Phoenix: 30
- Other: 7

Density:
- Low: 27
- High: 23
- Medium: 20
```

These are candidates, not the final evaluation set.

---

## Step 5 — Download Candidate Camera Boxes (already exist)

Script:

```text
scripts/milestone_2/waymo/04_download_candidate_camera_boxes.py
```

Run:

```cmd
python scripts\milestone_2\waymo\04_download_candidate_camera_boxes.py
```

Downloaded data:

```text
data/waymo/raw/validation/camera_box/candidates/
└── 70 Parquet files
```

Download log:

```text
data/waymo/selection/candidate_camera_box_download_report.csv
```

Successful result:

```text
Requested: 70
Successful or existing: 70
Failed: 0
Local Parquet files: 70
```

### Windows `gcloud` subprocess fix (potential error)

If Python cannot find `gcloud`, locate the Windows wrapper:

```python
gcloud_executable = (
    shutil.which("gcloud.cmd")
    or shutil.which("gcloud")
)
```

Run it through `cmd.exe`:

```python
command = [
    "cmd.exe",
    "/c",
    gcloud_executable,
    "storage",
    "cp",
    "--no-clobber",
    cloud_uri,
    str(output_directory),
]
```

The download scripts are resumable: completed files are skipped and zero-byte files are removed before retrying.

---

## Step 6 — Inspect the Camera-Box Schema (already exist)

Script:

```text
scripts/milestone_2/waymo/05_inspect_camera_box_schema.py
```

Run:

```cmd
python scripts\milestone_2\waymo\05_inspect_camera_box_schema.py
```

Output:

```text
data/waymo/selection/camera_box_schema.txt
```

Verified columns:

```text
key.segment_context_name
key.frame_timestamp_micros
key.camera_name
key.camera_object_id
[CameraBoxComponent].box.center.x
[CameraBoxComponent].box.center.y
[CameraBoxComponent].box.size.x
[CameraBoxComponent].box.size.y
[CameraBoxComponent].type
[CameraBoxComponent].difficulty_level.detection
[CameraBoxComponent].difficulty_level.tracking
```

Identifiers:

```text
Camera:
1 = FRONT

Object types:
1 = Vehicle
2 = Pedestrian
3 = Sign
4 = Cyclist
```

Missing difficulty values are recorded as `UNSPECIFIED`, not treated as easy.

---

## Step 7 — Analyze Candidate FRONT-Camera Boxes (already exist)

Script:

```text
scripts/milestone_2/waymo/06_analyze_candidate_front_boxes.py
```

Run:

```cmd
python scripts\milestone_2\waymo\06_analyze_candidate_front_boxes.py
```

Outputs:

```text
data/waymo/selection/candidate_front_camera_stats.csv
data/waymo/selection/front_box_size_thresholds.json
```

The script filters to:

```text
key.camera_name == 1
```

and analyzes only Vehicle, Pedestrian, and Cyclist.

Candidate results:

```text
Segments containing FRONT vehicles: 70
Segments containing FRONT pedestrians: 51
Segments containing FRONT cyclists: 20

Vehicle boxes: 208,703
Pedestrian boxes: 77,486
Cyclist boxes: 4,267
```

Candidate-relative size thresholds:

```text
Small: area <= 745.95 px²
Medium: 745.95 < area <= 3250.18 px²
Large: area > 3250.18 px²
```

These are project sampling categories, not official Waymo size definitions.

---

## Step 8 — Select and Freeze the Final 25 Segments (already exist)

Script:

```text
scripts/milestone_2/waymo/07_select_final_segments.py
```

Run:

```cmd
python scripts\milestone_2\waymo\07_select_final_segments.py
```

Outputs:

```text
data/waymo/selection/final_segments.csv
data/waymo/selection/selection_summary.json
```

The script uses binary mixed-integer optimization with SciPy `milp`. The preferred profile was feasible and solved optimally.

Final distribution:

```text
Segments: 25

Time:
- Day: 14
- Dawn/Dusk: 6
- Night: 5

Weather:
- Sunny: 24
- Rain: 1

Location:
- San Francisco: 12
- Phoenix: 11
- Other: 2

Density:
- High: 10
- Low: 8
- Medium: 7

Segments containing pedestrians: 22
Segments containing cyclists: 18
```

`final_segments.csv` is the frozen external-validation segment list. Do not change it after evaluation begins.

---

## Step 9 — Download Final Images and Calibration  (already exist)

Script:

```text
scripts/milestone_2/waymo/08_download_final_images_and_calibration.py
```

Run:

```cmd
python scripts\milestone_2\waymo\08_download_final_images_and_calibration.py
```

Downloaded:

```text
data/waymo/raw/validation/camera_image/final/
└── 25 Parquet files

data/waymo/raw/validation/camera_calibration/final/
└── 25 Parquet files
```

Log:

```text
data/waymo/selection/final_components_download_report.csv
```

Result:

```text
Expected component files: 50
Successful or existing: 50
Failed: 0
Camera-image files: 25
Camera-calibration files: 25
Camera-image size: 8.89 GB
Camera-calibration size: 0.21 MB
```

The final box files already existed inside the 70 candidate camera-box files.

Calibration is preserved for reproducibility even though existing 2D pixel-space boxes can be extracted without it.

---

## Step 10 — Inspect the Camera-Image Schema (already exist)

Script:

```text
scripts/milestone_2/waymo/09_inspect_camera_image_schema.py
```

Run:

```cmd
python scripts\milestone_2\waymo\09_inspect_camera_image_schema.py
```

Output:

```text
data/waymo/selection/camera_image_schema.txt
```

Key columns:

```text
key.segment_context_name
key.frame_timestamp_micros
key.camera_name
[CameraImageComponent].image
```

Images are stored as binary JPEG data.

---

## Step 11 — Extract the Representative Subset (already exist)

Script:

```text
scripts/milestone_2/waymo/10_extract_representative_subset.py
```

Run:

```cmd
python scripts\milestone_2\waymo\10_extract_representative_subset.py
```

Sampling protocol:

```text
Camera: FRONT
Camera ID: 1
Order: chronological
Sampling: every fifth FRONT frame
Start: first FRONT frame
Positions: 0, 5, 10, 15, ...
```

This avoids near-duplicate consecutive frames and prevents manual selection bias.

The script:

1. Reads each final image Parquet file.
2. Keeps FRONT-camera rows.
3. Sorts frames by timestamp.
4. Selects every fifth frame.
5. Saves original JPEG bytes.
6. Matches boxes using segment ID, timestamp, and camera ID.
7. Keeps Vehicle, Pedestrian, and Cyclist.
8. Ignores Sign.
9. Clips boxes to image boundaries.
10. Writes image-level and box-level metadata.

Outputs:

```text
data/waymo/representative_subset/images/front/<segment_id>/<timestamp>.jpg

data/waymo/representative_subset/annotations/boxes.csv
data/waymo/representative_subset/annotations/class_mapping.yaml

data/waymo/representative_subset/metadata/manifest.csv
data/waymo/representative_subset/metadata/subset_summary.json
```

Result:

```text
Segments represented: 25
Selected FRONT images: 996
Images without target boxes: 12
Total retained boxes: 24,819
Vehicle boxes: 16,928
Pedestrian boxes: 7,127
Cyclist boxes: 764
Invalid boxes skipped: 0
```

The 12 negative images are intentionally retained for false-positive evaluation.

---

## Step 12 — Validate and Visualize

Script:

```text
scripts/milestone_2/waymo/11_validate_and_visualize_subset.py
```

Run:

```cmd
python scripts\milestone_2\waymo\11_validate_and_visualize_subset.py
```

Validation output:

```text
data/waymo/representative_subset/metadata/subset_validation_report.json
```

Visual checks:

```text
data/waymo/representative_subset/visual_checks/
├── sample_vehicle.jpg
├── sample_pedestrian.jpg
├── sample_cyclist.jpg
├── sample_mixed_scene.jpg
├── sample_night_scene.jpg
├── sample_rain_scene.jpg
├── sample_negative_scene.jpg
└── sample_crowded_scene.jpg
```

Final validation:

```text
Manifest images: 996
JPG files found: 996
Segments represented: 25
Annotation rows: 24,819
Missing images: 0
Unreadable images: 0
Unknown annotation image IDs: 0
Invalid boxes: 0
Out-of-bounds boxes: 0
Manifest count mismatches: 0
Visual checks created: 8

Validation status: PASSED
```

---

## Final Directory Structure

```text
data/
└── waymo/
    ├── README.md
    ├── raw/
    │   └── validation/
    │       ├── stats/
    │       │   └── 202 Parquet files
    │       ├── camera_box/
    │       │   └── candidates/
    │       │       └── 70 Parquet files
    │       ├── camera_image/
    │       │   └── final/
    │       │       └── 25 Parquet files
    │       └── camera_calibration/
    │           └── final/
    │               └── 25 Parquet files
    ├── selection/
    │   ├── stats_schema.txt
    │   ├── segment_catalog.csv
    │   ├── candidate_segments.csv
    │   ├── candidate_camera_box_download_report.csv
    │   ├── camera_box_schema.txt
    │   ├── candidate_front_camera_stats.csv
    │   ├── front_box_size_thresholds.json
    │   ├── final_segments.csv
    │   ├── selection_summary.json
    │   ├── final_components_download_report.csv
    │   └── camera_image_schema.txt
    └── representative_subset/
        ├── images/
        │   └── front/
        │       └── <25 segment folders containing 996 JPGs>
        ├── annotations/
        │   ├── boxes.csv
        │   └── class_mapping.yaml
        ├── metadata/
        │   ├── manifest.csv
        │   ├── subset_summary.json
        │   └── subset_validation_report.json
        └── visual_checks/
            ├── sample_vehicle.jpg
            ├── sample_pedestrian.jpg
            ├── sample_cyclist.jpg
            ├── sample_mixed_scene.jpg
            ├── sample_night_scene.jpg
            ├── sample_rain_scene.jpg
            ├── sample_negative_scene.jpg
            └── sample_crowded_scene.jpg
```

Scripts:

```text
scripts/
└── milestone_2/
    └── waymo/
        ├── 01_inspect_waymo_stats.py
        ├── 02_build_waymo_segment_catalog.py
        ├── 03_select_candidate_segments.py
        ├── 04_download_candidate_camera_boxes.py
        ├── 05_inspect_camera_box_schema.py
        ├── 06_analyze_candidate_front_boxes.py
        ├── 07_select_final_segments.py
        ├── 08_download_final_images_and_calibration.py
        ├── 09_inspect_camera_image_schema.py
        ├── 10_extract_representative_subset.py
        └── 11_validate_and_visualize_subset.py
```

---

## Class Harmonization

| Final class ID | Final class | Waymo type |
|---:|---|---:|
| 0 | Vehicle | 1 |
| 1 | Pedestrian | 2 |
| 2 | Cyclist | 4 |

Ignored:

| Waymo type | Reason |
|---|---|
| Sign (`3`) | Not part of the shared KITTI–Waymo three-class task |

---

## Key Files for Later Milestones (very important)

Frozen selection:

```text
data/waymo/selection/final_segments.csv
data/waymo/selection/selection_summary.json
```

Images:

```text
data/waymo/representative_subset/images/front/
```

Annotations:

```text
data/waymo/representative_subset/annotations/boxes.csv
data/waymo/representative_subset/annotations/class_mapping.yaml
```

Metadata:

```text
data/waymo/representative_subset/metadata/manifest.csv
data/waymo/representative_subset/metadata/subset_summary.json
```

Validation evidence:

```text
data/waymo/representative_subset/metadata/subset_validation_report.json
data/waymo/representative_subset/visual_checks/
```

Milestone 3 will convert this common extracted representation into COCO and model-specific formats.

---

## Reproducibility Rules

1. Do not manually modify files under `raw/`.
2. Do not alter `final_segments.csv` after evaluation begins.
3. Use the same 996 images for every detector.
4. Do not use Waymo for training, tuning, or model selection.
5. Keep the 12 negative images.
6. Preserve the selection seed and optimization summary.
7. Preserve all CSV/JSON logs and schema reports.
8. Inspect visual checks before format conversion or evaluation.
9. Rerun validation after any preprocessing change.
10. Keep original Parquet files until Milestone 3 and evaluation are verified.

---

## Git and Storage Notes

Raw Waymo data and extracted JPG images are large and should normally not be committed.

Recommended `.gitignore` entries:

```gitignore
data/waymo/raw/
data/waymo/representative_subset/images/
```

Files normally suitable for tracking:

```text
data/waymo/README.md
data/waymo/selection/*.csv
data/waymo/selection/*.json
data/waymo/selection/*.txt
data/waymo/representative_subset/annotations/class_mapping.yaml
data/waymo/representative_subset/metadata/*.json
scripts/milestone_2/waymo/*.py
```

Whether `boxes.csv` and `manifest.csv` may be committed or redistributed depends on repository size and the applicable Waymo license and terms. Do not redistribute Waymo-derived data unless permitted.

---

## Troubleshooting

### `gcloud` works in Command Prompt but Python cannot find it

Resolve `gcloud.cmd` and use `cmd.exe /c`, as shown above.

### Download interrupted

Rerun the same downloader script. Completed files are skipped.

### Download appears inactive

The scripts use `capture_output=True`, which hides live `gcloud` progress. Inspect the destination in a second terminal:

```cmd
dir "data\waymo\raw\validation\camera_image\final"
```

### Validation fails

Inspect:

```text
data/waymo/representative_subset/metadata/subset_validation_report.json
```

Correct the issue and rerun:

```cmd
python scripts\milestone_2\waymo\11_validate_and_visualize_subset.py
```

---

## Completion Checklist

```text
[✓] Waymo access granted
[✓] Google Cloud CLI configured
[✓] 202 validation stats files downloaded
[✓] Stats schema inspected
[✓] Validation segment catalog created
[✓] 70 diverse candidate segments selected
[✓] 70 candidate camera-box files downloaded
[✓] Camera-box schema inspected
[✓] Candidate FRONT-camera labels analyzed
[✓] 25 final diverse segments optimized and frozen
[✓] 25 camera-image files downloaded
[✓] 25 calibration files downloaded
[✓] Camera-image schema inspected
[✓] 996 FRONT-camera images extracted
[✓] 24,819 target boxes extracted
[✓] Vehicle/Pedestrian/Cyclist mapping saved
[✓] Manifest and subset summary generated
[✓] Full validation passed
[✓] Eight visual checks generated
```
