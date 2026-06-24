import os
import sys
import json
import csv
import hashlib
import shutil
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
import torch
import torchvision
import ultralytics

def compute_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

def main():
    print("=================================================================")
    print("Validating Environment and Pretrained Weights...")
    print("=================================================================")

    # Define paths
    project_root = Path(__file__).resolve().parents[2]
    pretrained_dir = project_root / "outputs" / "milestone_4" / "pretrained"
    reports_dir = project_root / "outputs" / "milestone_4" / "reports"
    manifests_dir = project_root / "outputs" / "milestone_4" / "manifests"
    
    os.makedirs(pretrained_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)

    issues = []
    checks = {}
    weights_manifest = []

    def log_issue(check_name, file_path, description, severity="ERROR"):
        issues.append({
            "check_name": check_name,
            "file_path": str(file_path),
            "issue_description": description,
            "severity": severity
        })

    # --- Part 1: Verify System & CUDA ---
    checks["python_version"] = sys.version
    checks["pytorch_version"] = torch.__version__
    checks["torchvision_version"] = torchvision.__version__
    checks["ultralytics_version"] = ultralytics.__version__

    cuda_available = torch.cuda.is_available()
    checks["cuda_available"] = cuda_available
    
    if not cuda_available:
        log_issue("CUDA Check", "system", "CUDA is not available. GPU acceleration is required for training.")
        checks["gpu_name"] = "N/A"
        checks["gpu_memory_gb"] = 0.0
    else:
        gpu_name = torch.cuda.get_device_name(0)
        checks["gpu_name"] = gpu_name
        
        # Get memory info
        total_mem = torch.cuda.get_device_properties(0).total_memory
        total_mem_gb = round(total_mem / (1024 ** 3), 2)
        checks["gpu_memory_gb"] = total_mem_gb
        print(f"Detected GPU: {gpu_name} ({total_mem_gb} GB VRAM)")
        
        if total_mem_gb < 3.8:
            log_issue("GPU VRAM Check", "system", f"Detected GPU has only {total_mem_gb} GB VRAM. Extreme optimization is mandatory.", "WARNING")

        # Test AMP
        try:
            amp_supported = torch.cuda.is_bf16_supported() or torch.cuda.is_fp16_supported()
            checks["amp_supported"] = amp_supported
            if not amp_supported:
                log_issue("AMP Check", "system", "Automatic Mixed Precision (AMP) is not supported by the hardware/driver.", "WARNING")
        except Exception as e:
            log_issue("AMP Check", "system", f"Failed to verify AMP compatibility: {str(e)}")

    # --- Part 2: Verify Disk Space ---
    try:
        total, used, free = shutil.disk_usage(project_root)
        free_gb = round(free / (1024 ** 3), 2)
        checks["free_disk_space_gb"] = free_gb
        if free_gb < 5.0:
            log_issue("Disk Space Check", project_root, f"Only {free_gb} GB free disk space available. Recommended: at least 5.0 GB.", "WARNING")
    except Exception as e:
        log_issue("Disk Space Check", project_root, f"Failed to check disk space: {str(e)}")

    # --- Part 3: Validate / Download Pretrained Weights ---
    # We specify official download URLs for ultralytics assets
    weight_targets = {
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
        "rtdetr-l.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-l.pt"
    }

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    for name, url in weight_targets.items():
        dest_path = pretrained_dir / name
        if not dest_path.exists():
            print(f"Downloading {name} to {dest_path}...")
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                print(f"Downloaded {name} successfully.")
            except Exception as e:
                log_issue("Weights Download", dest_path, f"Failed to download weight from {url}: {str(e)}")
                continue
        
        # Verify size and hash
        if dest_path.exists():
            try:
                size_mb = round(os.path.getsize(dest_path) / (1024 ** 2), 2)
                sha256_hash = compute_sha256(dest_path)
                print(f"Validated {name}: {size_mb} MB (SHA256: {sha256_hash[:10]}...)")
                
                weights_manifest.append({
                    "model_name": name.split(".")[0],
                    "filename": name,
                    "file_path": str(dest_path),
                    "file_size_mb": size_mb,
                    "sha256_hash": sha256_hash,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                log_issue("Weights Validation", dest_path, f"Failed to calculate file metadata/hash: {str(e)}")

    # --- Part 4: Validate Torchvision Pretrained Weights Loading ---
    torchvision_weights = {
        "fasterrcnn_resnet50_fpn": torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        "retinanet_resnet50_fpn_v2": torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    }

    for name, weight_enum in torchvision_weights.items():
        print(f"Validating state dict load for {name}...")
        try:
            state_dict = weight_enum.get_state_dict(progress=True)
            num_keys = len(state_dict.keys())
            print(f"Successfully loaded {name} state dict with {num_keys} parameters.")
            weights_manifest.append({
                "model_name": name,
                "filename": "torchvision_internal_hub",
                "file_path": weight_enum.url,
                "file_size_mb": 0.0, # Checked dynamically by torch hub
                "sha256_hash": "torch_hub_verified",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            log_issue("Torchvision Weights Load", name, f"Failed to retrieve state dict for {name}: {str(e)}")

    # --- Part 5: Model Construction Verification ---
    print("Verifying model construction & GPU upload...")
    if cuda_available:
        device = torch.device("cuda")
        # 1. Test YOLOv8s
        try:
            from ultralytics import YOLO
            yolo_weight_path = pretrained_dir / "yolov8s.pt"
            if yolo_weight_path.exists():
                yolo_model = YOLO(str(yolo_weight_path))
                # Upload to GPU implicitly done by training/inference, let's run a small test
                dummy_tensor = torch.randn(1, 3, 640, 640).to(device)
                _ = yolo_model.model.to(device)(dummy_tensor)
                print("YOLOv8s construction and forward pass PASSED.")
                checks["yolov8s_construction"] = "PASSED"
            else:
                checks["yolov8s_construction"] = "FAILED (Weights missing)"
        except Exception as e:
            log_issue("Model Construction", "yolov8s", f"Failed to instantiate YOLOv8s: {str(e)}")
            checks["yolov8s_construction"] = "FAILED"

        # 2. Test RT-DETR-L
        try:
            from ultralytics import RTDETR
            rtdetr_weight_path = pretrained_dir / "rtdetr-l.pt"
            if rtdetr_weight_path.exists():
                rtdetr_model = RTDETR(str(rtdetr_weight_path))
                dummy_tensor = torch.randn(1, 3, 640, 640).to(device)
                _ = rtdetr_model.model.to(device)(dummy_tensor)
                print("RT-DETR-L construction and forward pass PASSED.")
                checks["rtdetr_construction"] = "PASSED"
            else:
                checks["rtdetr_construction"] = "FAILED (Weights missing)"
        except Exception as e:
            log_issue("Model Construction", "rtdetr_l", f"Failed to instantiate RT-DETR-L: {str(e)}")
            checks["rtdetr_construction"] = "FAILED"

        # 3. Test Faster R-CNN
        try:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            faster_rcnn_model = fasterrcnn_resnet50_fpn(weights=torchvision_weights["fasterrcnn_resnet50_fpn"])
            faster_rcnn_model.to(device)
            faster_rcnn_model.eval() # Eval mode for checking forward inference pass
            dummy_images = [torch.randn(3, 640, 640).to(device)]
            with torch.no_grad():
                _ = faster_rcnn_model(dummy_images)
            print("Faster R-CNN construction and forward pass PASSED.")
            checks["faster_rcnn_construction"] = "PASSED"
        except Exception as e:
            log_issue("Model Construction", "faster_rcnn", f"Failed to instantiate Faster R-CNN: {str(e)}")
            checks["faster_rcnn_construction"] = "FAILED"

        # 4. Test RetinaNet
        try:
            from torchvision.models.detection import retinanet_resnet50_fpn_v2
            retinanet_model = retinanet_resnet50_fpn_v2(weights=torchvision_weights["retinanet_resnet50_fpn_v2"])
            retinanet_model.to(device)
            retinanet_model.eval()
            dummy_images = [torch.randn(3, 640, 640).to(device)]
            with torch.no_grad():
                _ = retinanet_model(dummy_images)
            print("RetinaNet construction and forward pass PASSED.")
            checks["retinanet_construction"] = "PASSED"
        except Exception as e:
            log_issue("Model Construction", "retinanet", f"Failed to instantiate RetinaNet: {str(e)}")
            checks["retinanet_construction"] = "FAILED"
    else:
        print("Skipping GPU construction tests as CUDA is not available.")

    # --- Write Outputs ---
    # 1. Environment Snapshot
    env_snapshot_path = reports_dir / "environment_snapshot.json"
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "milestone": 4,
        "step": 6,
        "checks": checks,
        "errors_count": len([i for i in issues if i["severity"] == "ERROR"]),
        "warnings_count": len([i for i in issues if i["severity"] == "WARNING"])
    }
    with open(env_snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Environment snapshot saved to: {env_snapshot_path}")

    # 2. Pretrained Weights Manifest
    manifest_path = manifests_dir / "pretrained_weights_manifest.csv"
    with open(manifest_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "filename", "file_path", "file_size_mb", "sha256_hash", "timestamp"])
        writer.writeheader()
        for row in weights_manifest:
            writer.writerow(row)
    print(f"Pretrained weights manifest saved to: {manifest_path}")

    # 3. Issues CSV
    issues_path = reports_dir / "environment_issues.csv"
    with open(issues_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check_name", "file_path", "issue_description", "severity"])
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue)
    print(f"Environment validation issues saved to: {issues_path}")

    # --- Exit code ---
    final_status = "PASSED" if len([i for i in issues if i["severity"] == "ERROR"]) == 0 else "FAILED"
    print("-----------------------------------------------------------------")
    print(f"Environment & Weights Validation Status: {final_status} (Errors: {len([i for i in issues if i['severity'] == 'ERROR'])} / Warnings: {len([i for i in issues if i['severity'] == 'WARNING'])} )")
    print("=================================================================")
    
    if final_status == "FAILED":
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
