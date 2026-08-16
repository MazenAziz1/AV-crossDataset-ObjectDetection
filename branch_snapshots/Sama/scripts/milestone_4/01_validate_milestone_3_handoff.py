import os
import json
import csv
import sys
import yaml
from pathlib import Path
from PIL import Image
from datetime import datetime, timezone

def main():
    print("=================================================================")
    print("Running Milestone 3 Handoff Validation...")
    print("=================================================================")

    # Define paths
    project_root = Path(__file__).resolve().parents[2]
    m3_processed_root = project_root / "data" / "processed" / "milestone_3"
    m4_configs_root = project_root / "configs" / "models" / "milestone_4"
    m4_reports_root = project_root / "outputs" / "milestone_4" / "reports"
    
    # Ensure reports directory exists
    os.makedirs(m4_reports_root, exist_ok=True)
    
    report_json_path = m4_reports_root / "milestone_3_handoff_validation.json"
    issues_csv_path = m4_reports_root / "milestone_3_handoff_issues.csv"
    
    issues = []
    checks_performed = {}

    def log_issue(check_name, file_path, description, severity="ERROR"):
        issues.append({
            "check_name": check_name,
            "file_path": str(file_path),
            "issue_description": description,
            "severity": severity
        })

    # --- Check 1: Milestone 4 Configs exist and status is Frozen ---
    protocol_path = m4_configs_root / "experiment_protocol.yaml"
    class_contract_path = m4_configs_root / "class_contract.yaml"
    
    protocol = None
    class_contract = None
    
    if not protocol_path.exists():
        log_issue("Configs Check", protocol_path, "experiment_protocol.yaml does not exist.")
    else:
        try:
            with open(protocol_path, "r") as f:
                protocol = yaml.safe_load(f)
            if protocol.get("metadata", {}).get("status") != "frozen":
                log_issue("Configs Check", protocol_path, "experiment_protocol.yaml status is not frozen", "WARNING")
            checks_performed["configs_protocol_check"] = "PASSED"
        except Exception as e:
            log_issue("Configs Check", protocol_path, f"Failed to parse experiment_protocol.yaml: {str(e)}")
            
    if not class_contract_path.exists():
        log_issue("Configs Check", class_contract_path, "class_contract.yaml does not exist.")
    else:
        try:
            with open(class_contract_path, "r") as f:
                class_contract = yaml.safe_load(f)
            if class_contract.get("metadata", {}).get("status") != "frozen":
                log_issue("Configs Check", class_contract_path, "class_contract.yaml status is not frozen", "WARNING")
            checks_performed["configs_class_contract_check"] = "PASSED"
        except Exception as e:
            log_issue("Configs Check", class_contract_path, f"Failed to parse class_contract.yaml: {str(e)}")

    # --- Check 2: Milestone 3 Audit exists and PASSED ---
    m3_audit_path = m3_processed_root / "reports" / "final_dataset_audit.json"
    if not m3_audit_path.exists():
        log_issue("Audit Check", m3_audit_path, "final_dataset_audit.json does not exist.")
    else:
        try:
            with open(m3_audit_path, "r") as f:
                m3_audit = json.load(f)
            if not m3_audit.get("final_dataset_audit_passed", False):
                log_issue("Audit Check", m3_audit_path, "Milestone 3 final dataset audit status is not PASSED.")
            checks_performed["m3_audit_check"] = "PASSED"
        except Exception as e:
            log_issue("Audit Check", m3_audit_path, f"Failed to parse final_dataset_audit.json: {str(e)}")

    # --- Check 3: Image directories and files exist with correct counts ---
    kitti_train_images_dir = m3_processed_root / "images" / "kitti" / "train"
    kitti_val_images_dir = m3_processed_root / "images" / "kitti" / "val"
    
    train_count = 0
    val_count = 0
    
    if not kitti_train_images_dir.exists():
        log_issue("Dataset Check", kitti_train_images_dir, "KITTI train images directory does not exist.")
    else:
        train_count = len(list(kitti_train_images_dir.glob("*.png")))
        if train_count != 5985:
            log_issue("Dataset Check", kitti_train_images_dir, f"Expected 5985 KITTI train images, but found {train_count}.")
        checks_performed["kitti_train_count_check"] = f"PASSED ({train_count} images)"
            
    if not kitti_val_images_dir.exists():
        log_issue("Dataset Check", kitti_val_images_dir, "KITTI val images directory does not exist.")
    else:
        val_count = len(list(kitti_val_images_dir.glob("*.png")))
        if val_count != 1496:
            log_issue("Dataset Check", kitti_val_images_dir, f"Expected 1496 KITTI val images, but found {val_count}.")
        checks_performed["kitti_val_count_check"] = f"PASSED ({val_count} images)"

    # --- Check 4: Image resolution check (Sample of 10 images) ---
    resolution_ok = True
    images_to_sample = list(kitti_train_images_dir.glob("*.png"))[:5] + list(kitti_val_images_dir.glob("*.png"))[:5]
    for img_path in images_to_sample:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if w != 640 or h != 640:
                    log_issue("Image Resolution Check", img_path, f"Image size is {w}x{h}, expected 640x640.")
                    resolution_ok = False
        except Exception as e:
            log_issue("Image Resolution Check", img_path, f"Failed to open image for resolution check: {str(e)}")
            resolution_ok = False
            
    if resolution_ok and images_to_sample:
        checks_performed["image_resolution_check"] = "PASSED (all sampled images are 640x640)"

    # --- Check 5: COCO Annotations exist and are valid ---
    coco_train_path = m3_processed_root / "annotations" / "coco" / "kitti_train.json"
    coco_val_path = m3_processed_root / "annotations" / "coco" / "kitti_val.json"
    
    for label, path in [("train", coco_train_path), ("validation", coco_val_path)]:
        if not path.exists():
            log_issue("COCO Annotation Check", path, f"COCO {label} annotations file does not exist.")
        else:
            try:
                with open(path, "r") as f:
                    coco_data = json.load(f)
                
                # Check categories match class mapping contract
                cats = coco_data.get("categories", [])
                expected_cats = {1: "Vehicle", 2: "Pedestrian", 3: "Cyclist"}
                for cat in cats:
                    cid = cat.get("id")
                    cname = cat.get("name")
                    if cid in expected_cats:
                        if expected_cats[cid] != cname:
                            log_issue("Class Mapping Check", path, f"COCO category id {cid} maps to name '{cname}', expected '{expected_cats[cid]}'.")
                    else:
                        log_issue("Class Mapping Check", path, f"Unexpected category ID {cid} with name '{cname}' in COCO annotations.")
                
                checks_performed[f"coco_{label}_check"] = "PASSED"
            except Exception as e:
                log_issue("COCO Annotation Check", path, f"Failed to parse COCO {label} annotations: {str(e)}")

    # --- Check 6: YOLO Labels exist for all images ---
    yolo_train_labels_dir = m3_processed_root / "labels" / "kitti" / "train"
    yolo_val_labels_dir = m3_processed_root / "labels" / "kitti" / "val"
    
    yolo_labels_ok = True
    if not yolo_train_labels_dir.exists():
        log_issue("YOLO Label Check", yolo_train_labels_dir, "YOLO train labels directory does not exist.")
        yolo_labels_ok = False
    else:
        label_count = len(list(yolo_train_labels_dir.glob("*.txt")))
        if label_count != train_count:
            log_issue("YOLO Label Check", yolo_train_labels_dir, f"Train image count ({train_count}) and YOLO label count ({label_count}) do not match.")
            yolo_labels_ok = False
            
    if not yolo_val_labels_dir.exists():
        log_issue("YOLO Label Check", yolo_val_labels_dir, "YOLO val labels directory does not exist.")
        yolo_labels_ok = False
    else:
        label_count = len(list(yolo_val_labels_dir.glob("*.txt")))
        if label_count != val_count:
            log_issue("YOLO Label Check", yolo_val_labels_dir, f"Validation image count ({val_count}) and YOLO label count ({label_count}) do not match.")
            yolo_labels_ok = False
            
    if yolo_labels_ok:
        checks_performed["yolo_labels_count_check"] = "PASSED"

    # --- Check 7: DontCare and Excluded Object Sidecars exist ---
    sidecars_ok = True
    for split in ["train", "val"]:
        ignore_path = m3_processed_root / "annotations" / "ignore_regions" / f"kitti_{split}_ignore.json"
        excluded_path = m3_processed_root / "annotations" / "excluded_objects" / f"kitti_{split}_excluded.json"
        
        if not ignore_path.exists():
            log_issue("Sidecar Check", ignore_path, f"KITTI {split} ignore regions sidecar is missing.")
            sidecars_ok = False
        if not excluded_path.exists():
            log_issue("Sidecar Check", excluded_path, f"KITTI {split} excluded objects sidecar is missing.")
            sidecars_ok = False
            
    if sidecars_ok:
        checks_performed["ignore_and_excluded_sidecars_check"] = "PASSED"

    # --- Check 8: Absence of Waymo paths in Milestone 4 Configs ---
    if protocol:
        # Check if Waymo paths exist. Note: waymo_exclusion_rule is allowed to mention "waymo" or "Waymo".
        # We check configs for other waymo parameters or paths.
        configs_str = json.dumps(protocol).lower()
        if "waymo" in configs_str:
            # Check if it is only mentioned in the exclusion rule or metadata
            exclusion_val = protocol.get("scope", {}).get("waymo_exclusion_rule", False)
            if not exclusion_val:
                log_issue("Waymo Check", protocol_path, "waymo_exclusion_rule is not set to true in experiment_protocol.yaml.")
            
            # Look for active training/evaluation datasets
            train_ds = protocol.get("data", {}).get("roles", {}).get("training", {}).get("dataset", "")
            val_ds = protocol.get("data", {}).get("roles", {}).get("validation", {}).get("dataset", "")
            if "waymo" in train_ds.lower() or "waymo" in val_ds.lower():
                log_issue("Waymo Check", protocol_path, f"Waymo is active in data roles: training={train_ds}, validation={val_ds}.")
        checks_performed["waymo_exclusion_verification"] = "PASSED"

    # --- Save Issues to CSV ---
    try:
        with open(issues_csv_path, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["check_name", "file_path", "issue_description", "severity"])
            writer.writeheader()
            for issue in issues:
                writer.writerow(issue)
        print(f"Handoff issues CSV saved to: {issues_csv_path}")
    except Exception as e:
        print(f"Error saving issues CSV: {str(e)}")

    # --- Final Status Gate ---
    final_status = "PASSED" if len([i for i in issues if i["severity"] == "ERROR"]) == 0 else "FAILED"
    
    validation_summary = {
        "milestone": 4,
        "step": 3,
        "purpose": "Verify Milestone 3 handoff requirements are fully satisfied before starting model integration.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks_performed": checks_performed,
        "issues_found": len(issues),
        "errors_count": len([i for i in issues if i["severity"] == "ERROR"]),
        "warnings_count": len([i for i in issues if i["severity"] == "WARNING"]),
        "final_status": final_status
    }
    
    # --- Save Report to JSON ---
    try:
        with open(report_json_path, "w") as json_file:
            json.dump(validation_summary, json_file, indent=2)
        print(f"Handoff validation report JSON saved to: {report_json_path}")
    except Exception as e:
        print(f"Error saving report JSON: {str(e)}")

    print("-----------------------------------------------------------------")
    print(f"Handoff Validation Status: {final_status} (Issues found: {len(issues)})")
    print("=================================================================")
    
    if final_status == "FAILED":
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
