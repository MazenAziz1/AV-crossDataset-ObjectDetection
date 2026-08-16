# Milestone 4 + 5: Model Selection Rationale
**Status**: `FROZEN`
**Date Frozen**: 2026-06-24
**Target GPU**: NVIDIA Tesla T4 (16 GB VRAM)

---

## 1. Context & Architecture Preservation
For research consistency and comparability, this study preserves the originally selected detector architectures:
1. **YOLOv8s** (Small CNN, Single-Stage, Anchor-Free)
2. **Faster R-CNN ResNet-50 FPN** (Large CNN, Two-Stage, Anchor-Based)
3. **RetinaNet ResNet-50 FPN V2** (Medium/Large CNN, Single-Stage, Anchor-Based, Optimized Head)
4. **RT-DETR-L** (Large Vision Transformer, Query-Based)

Training runs on Kaggle **NVIDIA Tesla T4 (16 GB VRAM)** slots. Under this hardware, three of the four models fit a full physical batch of 32; only Faster R-CNN and RetinaNet require gradient accumulation, and Faster R-CNN additionally uses gradient checkpointing for memory safety.

---

## 2. Training Configurations & Memory Footprints

| Model Architecture | Parameters | Input Size | Physical Batch | Grad. Accum. Factor | Effective Batch | Est. VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8s** | ~11.2M | 640x640 | 32 | 1 | 32 | ~5.5 GB |
| **Faster R-CNN ResNet-50** | ~41.8M | 640x640 | 4 | 8 | 32 | ~14 GB |
| **RetinaNet ResNet-50 V2** | ~34.0M | 640x640 | 4 | 8 | 32 | ~9.5 GB |
| **RT-DETR-L** | ~32.9M | 640x640 | 16 | 1 | 16 | ~12 GB |

---

## 3. VRAM Mitigation Strategies

### A. Automatic Mixed Precision (AMP)
All four detectors train with AMP (`amp: true`) to reduce memory footprint and speed up training on the T4.

### B. Gradient Accumulation
Faster R-CNN and RetinaNet use a physical batch size of 4 with a gradient accumulation factor of 8, yielding an effective batch size of 32 without exceeding VRAM.

### C. Gradient Checkpointing (Activation Checkpointing)
Faster R-CNN enables gradient checkpointing on the ResNet-50 backbone. Intermediate activations are discarded and recomputed during backpropagation, reducing activation memory.

### D. Full Backbone Training
Faster R-CNN and RetinaNet train the full FPN backbone (`trainable_backbone_layers: 5`). No backbone freezing is applied on the T4.

---

## 4. Technical Tradeoffs & Risks

1. **Training Time Overhead**:
   * **Gradient accumulation**: Faster R-CNN and RetinaNet process 4 images per optimizer step, increasing the number of forward/backward passes per epoch.
   * **Checkpointing overhead**: Faster R-CNN's gradient checkpointing re-computes forward passes, adding roughly ~25% time overhead.
2. **Transformer Attention Bottlenecks**:
   * RT-DETR-L uses self-attention blocks whose memory scales quadratically with sequence length. It is therefore trained at the reduced batch size of 16. If VRAM OOMs at 640x640 even at batch 16, a fallback clears the CUDA cache (`torch.cuda.empty_cache()`) before retrying.
