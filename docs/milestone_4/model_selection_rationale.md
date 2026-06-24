# Milestone 4 + 5: Model Selection Rationale
**Status**: `FROZEN`
**Date Frozen**: 2026-06-24
**Target GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)

---

## 1. Context & Architecture Preservation
For research consistency and comparability, this study preserves the originally selected detector architectures:
1. **YOLOv8s** (Small CNN, Single-Stage, Anchor-Free)
2. **Faster R-CNN ResNet-50 FPN** (Large CNN, Two-Stage, Anchor-Based)
3. **RetinaNet ResNet-50 FPN V2** (Medium/Large CNN, Single-Stage, Anchor-Based, Optimized Head)
4. **RT-DETR-L** (Large Vision Transformer, Query-Based)

To fit these networks within the 4 GB VRAM limitation of the RTX 3050 Laptop GPU (approximately **~3.3 GB of free CUDA memory** after OS context overhead), we adapt the training configs rather than downgrading the model architectures.

---

## 2. Adapted Training Configurations & Memory Footprints

The following table details how each model is configured for memory-efficient training:

| Model Architecture | Parameters | Input Size | Min Batch Size | Grad. Accum. Factor | Est. VRAM | Success Probability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8s** | ~11.2M | 640x640 | 2 | 8 (effective: 16) | ~2.3 GB | **99%** |
| **Faster R-CNN ResNet-50** | ~41.8M | 640x640 | 1 | 16 (effective: 16) | ~2.8 GB | **85%** |
| **RetinaNet ResNet-50 V2** | ~34.0M | 640x640 | 1 | 16 (effective: 16) | ~2.9 GB | **85%** |
| **RT-DETR-L** | ~32.9M | 640x640 | 1 | 16 (effective: 16) | ~3.2 GB | **75%** |

---

## 3. Core VRAM Adaptation Strategies

To achieve these low footprints, the training code implements several memory-management mechanisms:

### A. Batch Size 1/2 + Gradient Accumulation
We reduce the batch size to the absolute minimum (1 for heavy models, 2 for YOLOv8s). To prevent gradients from becoming unstable during training, we use **Gradient Accumulation**. Over a factor of $N$ steps, gradients are accumulated in memory, and optimizer weights are updated only once every $N$ steps. This simulates an effective batch size of 16 without physical VRAM usage.

### B. Gradient Checkpointing (Activation Checkpointing)
For Faster R-CNN and RetinaNet, activation layers consume the majority of VRAM during the forward pass to prepare for the backward pass. Enabling **Gradient Checkpointing** discards intermediate activations and recalculates them on-the-fly during backpropagation, reducing activation memory footprint by up to 60%.

### C. Backbone Stage Freezing & Eval-Mode BatchNorm
For ResNet-50 backbones, we freeze the early layers (stages 1 and 2, and optionally stage 3). By setting these layers to `requires_grad=False` and forcing Batch Normalization layers to eval mode:
1. We save optimizer states (which take $2 \times$ weight parameters memory in Adam).
2. We prevent activation storage for the frozen blocks.

### D. CUDA Allocation Optimization
We prevent memory fragmentation in PyTorch by setting:
```bash
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```
This forces CUDA to recycle allocated memory blocks and prevents fragmentation-related OOM errors when switching between training iterations and validation loops.

---

## 4. Technical Tradeoffs & Risks

1. **Training Time Overhead**:
   * **Loop overhead**: Processing 1 image at a time increases the frequency of CPU-to-GPU memory swaps, which can increase training times by 1.5x.
   * **Checkpointing overhead**: Gradient checkpointing requires re-calculating forward passes, which adds roughly ~25% time overhead.
2. **Transformer Attention Bottlenecks**:
   * RT-DETR-L utilizes self-attention blocks. Since attention memory scales quadratically with sequence length, if VRAM OOMs at 640x640 even with batch size 1, we will insert a fallback mechanism to clear the CUDA cache (`torch.cuda.empty_cache()`) or freeze the backbone completely.
