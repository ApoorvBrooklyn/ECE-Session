# Deep Learning — Hands-On Exercises
### Graduate Session: Unit V

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

---

## Exercise Overview

| # | Notebook | Topics Covered | Difficulty | Est. Time |
|---|----------|----------------|------------|-----------|
| 1 | `01_neural_networks_from_scratch/` | Neural Network, Backprop, Activations | ★★☆ | 60 min |
| 2 | `02_cnn_architectures/` | LeNet, AlexNet, VGGNet, Feature Maps | ★★★ | 90 min |
| 3 | `03_backpropagation_visualization/` | Autograd, Vanishing Gradients, BatchNorm | ★★★ | 75 min |
| 4 | `04_object_detection/` | R-CNN, Faster R-CNN, YOLO, IoU, NMS | ★★★ | 90 min |
| 5 | `05_rnn_lstm/` | RNN, LSTM, Text Generation | ★★★ | 90 min |
| 6 | `06_gans/` | DCGAN, Mode Collapse, Latent Space | ★★★ | 90 min |

---

## What You Build

### Exercise 1 — Neural Network from Scratch
Build a 3-layer network **using only NumPy** and classify a spiral dataset that defeats any linear classifier. Implement forward pass, cross-entropy loss, and backpropagation manually.

**The hook**: Watch the decision boundary evolve from random noise to perfectly spiraling regions.

### Exercise 2 — CNN Architecture Evolution
Implement LeNet (1998), AlexNet (2012), and VGGNet (2014) in PyTorch and race them on Fashion-MNIST. Then peer inside the network using feature map visualization to see what early vs. deep layers detect.

**The hook**: See how 16 years of architecture research is captured in 200 lines of PyTorch.

### Exercise 3 — Build Your Own Autograd Engine
Implement a `Value` class that tracks its own gradient — essentially building a toy version of PyTorch's autograd. Then measure gradient norms across a 20-layer network with sigmoid vs. ReLU to see vanishing gradients in real time.

**The hook**: Run `o.backward()` on your own engine and watch it agree with PyTorch to 5 decimal places.

### Exercise 4 — Object Detection: R-CNN to YOLO
Implement IoU and NMS from scratch, run a pretrained Faster R-CNN on real images, and (optionally) compare with YOLOv8. Visualize the speed-accuracy Pareto frontier across 8 different architectures.

**The hook**: R-CNN takes 47 seconds per image. YOLO takes 25ms. Same task. 2000× faster.

### Exercise 5 — LSTM Shakespeare Generator
Demonstrate the vanishing gradient problem in vanilla RNNs, implement an LSTM cell from scratch (verified against PyTorch), then train a character-level language model on Shakespeare. Visualize cell state activations to see what the LSTM is "thinking."

**The hook**: After 3000 training steps the model generates plausible Shakespearean dialogue.

### Exercise 6 — GANs: Two Networks at War
Train a Vanilla GAN, watch it suffer from mode collapse, then fix it with DCGAN. Explore latent space interpolation to watch one handwritten digit smoothly morph into another.

**The hook**: The latent space interpolation is genuinely magical — smooth semantic transitions through 100-dimensional noise space.

---

## Key Dependencies
- **PyTorch 2.0+** — primary framework
- **torchvision** — datasets and pretrained models
- **ultralytics** — YOLOv8 (Exercise 4, optional)
- **matplotlib** — all visualizations
- **scikit-learn** — spiral dataset (Exercise 1)
