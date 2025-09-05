# Vision Transformer (ViT) for CIFAR-10 Classification

## Overview
This project implements a Vision Transformer (ViT) model from scratch using PyTorch to classify images from the CIFAR-10 dataset. The ViT architecture adapts transformer models, originally designed for natural language processing, to computer vision tasks by splitting images into patches and processing them as sequences.

## Model Architecture
- **Patch Embedding**: Images are divided into patches (4x4 pixels) and projected into an embedding space using a convolutional layer.
- **Positional Encoding**: Learnable positional embeddings are added to preserve spatial information.
- **Transformer Encoder**: Consists of multiple layers of multi-head self-attention and MLP blocks with residual connections.
- **Classification Head**: Uses the [CLS] token representation for final classification.

## Key Features
- Custom implementation of ViT components (PatchEmbedding, TransformerEncoder, MLP)
- Data augmentation (random cropping, flipping, color jitter)
- Device-agnostic code (runs on CPU or GPU)
- Training and evaluation loops with accuracy tracking

## Hyperparameters
- Batch Size: 128
- Epochs: 10
- Learning Rate: 3e-4
- Patch Size: 4
- Embedding Dimension: 256
- Number of Heads: 8
- Transformer Depth: 6 layers
- MLP Dimension: 512
- Dropout Rate: 0.1

## Results
The model achieves:
- Training Accuracy: ~65% (after 10 epochs)
- Test Accuracy: ~60% (after 10 epochs)

Performance can be improved by:
- Training for more epochs (e.g., 30+)
- Using a larger embedding dimension
- Increasing model depth
- Adding more data augmentation

## Usage
1. Install dependencies:
```bash
pip install torch torchvision matplotlib numpy tqdm
```

2. Run the notebook/code:
- The code will automatically download CIFAR-10
- Train the ViT model
- Display training curves and sample predictions

## File Structure
```
├── vit_cifar10.ipynb          # Main implementation notebook
├── README.md                  # This file
└── data/                      # CIFAR-10 dataset (auto-downloaded)
```

## Dependencies
- Python 3.7+
- PyTorch 1.13+
- TorchVision
- Matplotlib
- NumPy
- tqdm

## References
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
