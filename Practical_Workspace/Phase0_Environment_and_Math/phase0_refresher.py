import torch
import numpy as np

print("--- 1. ENVIRONMENT & CUDA CHECK ---")
print(f"PyTorch Version Installed: {torch.__version__}")
print(f"NumPy Version Installed: {np.__version__}")

# Check if an NVIDIA GPU is available
if torch.cuda.is_available():
    print("Hardware: GPU (CUDA) is AVAILABLE! Training will be fast.")
else:
    print("Hardware: GPU (CUDA) is NOT available. Using CPU. (Expected for Intel graphics).")


print("\n--- 2. NUMPY & SHAPES REFRESHER ---")

# 1. A Vector (1 Dimension)
vector = np.array([10, 20, 30, 40, 50])
print(f"\nVector Data: {vector}")
print(f"Vector Shape: {vector.shape} --> (1 Dimension, 5 elements)")

# 2. A Matrix (2 Dimensions)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(f"\nMatrix Data:\n{matrix}")
print(f"Matrix Shape: {matrix.shape} --> (2 Rows, 3 Columns)")

# 3. Broadcasting (No loops required!)
print("\n--- 3. BROADCASTING MAGIC ---")
print("Adding 100 to the entire matrix instantly without a for-loop:")
print(matrix + 100)

print("\nPhase 0 Complete! Your environment is perfectly configured.")
