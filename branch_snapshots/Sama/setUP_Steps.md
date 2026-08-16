# Python + PyTorch GPU Setup Guide

## For Deep Learning, Generative AI, and Autonomous Driving Object Detection Projects

This guide explains how to set up a clean Python environment on Windows 11 for deep learning, PyTorch CUDA, Jupyter, and object detection research.

---

# 1. Regularly, each time you work on the existing project

Go to the project folder:

```bash
cd "C:\Users\Mazen\Desktop\AAST\Research\Autonomous research"
```

Activate the existing environment:

```bash
AVenv\Scripts\activate
```

Check that the environment is active:

```bash
python --version
```

Then you can run your scripts, notebooks, or JupyterLab:

```bash
jupyter lab
```

---

# 2. VS Code setup

Open the project folder in VS Code.

Then press:

```text
Ctrl + Shift + P
```

Choose:

```text
Python: Select Interpreter
```

Then select:

```text
AVenv\Scripts\python.exe
```

This makes VS Code use the correct project environment.

---
# very important first to ensure all other steps work seamlessly
## 3. Prerequisite checks before creating a new environment 

Check installed Python versions:

```bash
py -0p
```

Recommended Python version:

```text
Python 3.12
```

Check NVIDIA GPU and driver:

```bash
nvidia-smi
```

If `nvidia-smi` shows your NVIDIA GPU, then the machine is ready for PyTorch CUDA installation.

---

## 4. For starting any new Deep Learning or GenAI project

First, go to the folder where you want to create your new project.

Example:

```bash
cd "C:\Users\Mazen\Desktop\AAST\Research"
mkdir New_Project_Name
cd New_Project_Name
```

Create a new virtual environment using Python 3.12:

```bash
py -3.12 -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Check Python version:

```bash
python --version
```

Upgrade basic tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

# 5. Install PyTorch with CUDA

For NVIDIA GPU support, install PyTorch CUDA version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Test PyTorch GPU:

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Torch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"
```

Expected result:

```text
CUDA available: True
```

If it says `False`, then PyTorch is not using the GPU.

---

# 6. Install common Data Science and ML packages

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy tqdm pyyaml
```

---

# 7. Install Jupyter support (not necessary if you are using VS Code)

```bash
pip install jupyterlab ipykernel ipywidgets
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name New_Project_Name --display-name "Python 3.12 - New_Project_Name"
```

---

# 8. Install Generative AI packages

```bash
pip install transformers datasets accelerate peft
pip install diffusers safetensors
```

Optional for experiment tracking:

```bash
pip install tensorboard
```

---

# 9. Install object detection and computer vision packages

```bash
pip install ultralytics
pip install opencv-python pillow albumentations
pip install pycocotools torchmetrics
pip install torchinfo thop ptflops
pip install fiftyone supervision imagesize
```

Optional config and logging tools:

```bash
pip install hydra-core omegaconf python-dotenv loguru rich
```

---

# If you are not the first one setting up the project

After creating and setting up the environment. find the shared `requirements.txt`, you do **not** need to manually install every package one by one.

After creating and activating the virtual environment:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

Install all project packages from the shared requirements file:

```bash
pip install -r requirements.txt
```

Then check that everything is installed correctly:

```bash
python -m pip check
```

Test PyTorch GPU:

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Torch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"
```

Expected result:

```text
CUDA available: True
```

## What this replaces

This `pip install -r requirements.txt` step replaces the manual installation sections:

```text
Section 5: Install PyTorch with CUDA
Section 6: Install common Data Science and ML packages
Section 7: Install Jupyter support
Section 8: Install Generative AI packages
Section 9: Install object detection and computer vision packages
```

**So you should either:**

```text
Option A: Install packages manually using Sections 5–9
```

or:

```text
Option B: Use requirements.txt with pip install -r requirements.txt
```

Do not do both unless you know what you are changing.

# 10. Final health checks

Check package conflicts:

```bash
python -m pip check
```

Expected output:

```text
No broken requirements found.
```

Test core imports:

```bash
python -c "import torch, torchvision, ultralytics, cv2, pycocotools, torchmetrics; print('Core packages imported successfully')"
```

Save installed packages:

```bash
python -m pip freeze > requirements.txt
```

---

# 11. Important note about RTX 3060 Laptop GPU with 6GB VRAM

The limitation is usually GPU memory, not PyTorch.

With 6GB VRAM, you can comfortably do:

```text
✅ CNN training
✅ Small/medium vision models
✅ BERT-base fine-tuning
✅ DistilBERT / MiniLM / small Transformers
✅ Stable Diffusion 1.5 inference
✅ LoRA training on small models
✅ Small LLM inference with quantization
```

But these will be hard locally:

```text
⚠️ Training large LLMs from scratch
⚠️ Fine-tuning 7B+ models comfortably
⚠️ Running big diffusion/video-generation models
⚠️ Training SDXL with large batch sizes
⚠️ Multi-modal giant models
```

For larger experiments, use cloud/server GPUs:

```text
Google Colab
Kaggle GPU
RunPod
Vast.ai
Lambda Labs
Paperspace
University GPU server
```

---

# 12. For the autonomous driving object detection project

The recommended setup is:

```text
Python 3.12
PyTorch CUDA
Ultralytics for YOLO + RT-DETR
Torchvision for Faster R-CNN + RetinaNet
pycocotools / torchmetrics for mAP
OpenCV / Albumentations for preprocessing
TensorBoard for tracking
torchinfo / thop / ptflops for efficiency metrics
FiftyOne / supervision for visualization and failure analysis
```

This setup is suitable for:

```text
KITTI training
Waymo external validation
YOLO experiments
RT-DETR experiments
Faster R-CNN experiments
RetinaNet experiments
mAP evaluation
FPS and inference time analysis
FLOPs and parameter counting
Failure-case visualization
```
---

# 14. Main rule

Do not install packages globally.

For every new project:

```text
Create a new virtual environment
Activate it
Install packages inside it
Select the same environment in VS Code
Save requirements.txt
```

# whenever we make change on github
```text
git status
git add .
git commit -m "Add KITTI preprocessing script"
git push
```