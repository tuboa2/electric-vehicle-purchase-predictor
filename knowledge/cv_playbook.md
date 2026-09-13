# Computer Vision (CV) Playbook

> **Strategic Memory (Layer 3)**

## 1. Backbone Architectures
- ConvNeXt-v2 (tiny, small, base).
- EVA-02 / Swin Transformer v2.
- EfficientNet-B0 to B4 for rapid prototyping.

## 2. Augmentations & Regularization
- Albumentations: RandomCrop, HorizontalFlip, ShiftScaleRotate, CoarseDropout.
- Mixup ($\alpha = 0.8$) and CutMix ($\alpha = 1.0$) for multi-class classification.
- Progressive resizing: train on $224 \times 224$, fine-tune on $384 \times 384$ or $512 \times 512$.
