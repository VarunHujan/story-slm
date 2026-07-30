# Phase 0: Environment & Math Refresher

## 1. Environment Setup & CUDA
Before we build AI, we need the right tools. (We actually already installed these behind the scenes!).
- **Python venv**: Isolates our project so we don't break other apps on your PC.
- **PyTorch (`torch`)**: The core Deep Learning library that handles all the complex calculus and matrix math for us.
- **CUDA Check**: PyTorch can use a dedicated GPU to run 100x faster using NVIDIA's CUDA technology. If you don't have an NVIDIA GPU, PyTorch will automatically fall back to using your CPU. (Since you have an Intel GPU, we will be using the CPU, which is perfect for learning!).

## 2. NumPy & Tensor Shapes
In Machine Learning, we don't deal with single numbers. We deal with massive grids of numbers called **Tensors** (or Arrays in NumPy). 

### What is a "Shape"?
The most important concept in AI programming is the **Shape** of your data. If you understand shapes, you can build any neural network.
- `[5]` -> A 1D list of 5 numbers (A Vector).
- `[3, 5]` -> A 2D grid: 3 rows, 5 columns (A Matrix).
- `[2, 3, 5]` -> A 3D block of numbers (A Tensor).

When building our SLM, you will constantly see shapes like `[Batch_Size, Sequence_Length, Embedding_Dimension]`. If the shapes don't align perfectly when you multiply them, the neural network will crash!

### Broadcasting
Broadcasting is a math trick. If you want to add `10` to a matrix of 1,000 numbers, you don't write a `for-loop`. You just write `matrix + 10`, and Python instantly "broadcasts" the addition to every single number at the exact same time. This is why AI code is so fast!
