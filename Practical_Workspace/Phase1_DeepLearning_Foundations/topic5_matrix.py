import torch

print("--- TOPIC 5: MATRIX MULTIPLICATION (DOT PRODUCTS) ---")

# Let's create two Vectors
input_vector = torch.tensor([1, 2, 3])
weight_vector = torch.tensor([4, 5, 6])

print(f"Input Vector: {input_vector}")
print(f"Weight Vector: {weight_vector}")

# 1. The Dot Product
# Manual math: (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
dot_product = torch.dot(input_vector, weight_vector)
print(f"\nDot Product Result: {dot_product.item()}")

# 2. Matrix Multiplication (Rows x Columns)
matrix_a = torch.tensor([
    [1, 2],
    [3, 4]
]) # Shape: 2x2

matrix_b = torch.tensor([
    [5, 6],
    [7, 8]
]) # Shape: 2x2

# In PyTorch, we use torch.matmul() or the @ symbol for Matrix Multiplication!
result_matrix = matrix_a @ matrix_b

print("\n--- MATRIX MULTIPLICATION (using @ symbol) ---")
print(f"Matrix A:\n{matrix_a}")
print(f"Matrix B:\n{matrix_b}")
print(f"Result (A @ B):\n{result_matrix}")
