import torch

print("--- TOPIC 4: TENSORS ---")

# 1. 0D Scalar
scalar = torch.tensor(5)
print(f"\nScalar: {scalar}")
print(f"Shape: {scalar.shape} | Dimensions: {scalar.ndim}")

# 2. 1D Vector
vector = torch.tensor([1, 2, 3, 4])
print(f"\nVector: {vector}")
print(f"Shape: {vector.shape} | Dimensions: {vector.ndim}")

# 3. 2D Matrix
matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
print(f"\nMatrix:\n{matrix}")
print(f"Shape: {matrix.shape} (2 rows, 3 columns) | Dimensions: {matrix.ndim}")

# 4. Moving Tensors to GPU (if available)
print("\n--- HARDWARE ---")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Your computer is using: {device.upper()}")

# Move the matrix to the device (It will go to CPU since you have an Intel GPU)
matrix_on_device = matrix.to(device)
print(f"Matrix is now physically stored on: {matrix_on_device.device}")

print("\n--- AUTOGRAD (Automatic Calculus) ---")
# Creating a tensor with "requires_grad=True" tells PyTorch to track its calculus
smart_tensor = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"Smart Tensor: {smart_tensor}")
print("Notice 'requires_grad=True'! PyTorch is now actively watching this tensor to do calculus on it later.")
