import torch
import torch.nn as nn

print("==================================================")
print("Masterclass: Linear Layers (Weights and Biases)")
print("==================================================\n")

# =====================================================================
# 1. THE MANUAL WAY (Under the hood math)
# =====================================================================
print("--- 1. Manual Implementation (The Math) ---")

# Let's say we have 1 pizza (Batch size 1) with 3 features:
# [Dough (grams), Cheese (grams), Pepperoni (slices)]
# We will scale the numbers down so they are easier to read.
x = torch.tensor([[1.0, 2.0, 3.0]]) # Shape: [1, 3]

# We want to predict 2 things: [Price, Calories]
# We need a Weights Matrix. 
# Shape needs to be [in_features, out_features] if we do x @ W
# Let's define manual weights.
W = torch.tensor([
    [0.5, 10.0],  # Weights for Dough (Price weight, Calorie weight)
    [0.8, 15.0],  # Weights for Cheese
    [1.5, 20.0]   # Weights for Pepperoni
]) # Shape: [3, 2]

# Bias vector (1 bias for Price, 1 for Calories)
# Let's say base price is $2.0, and base calories from the box is 0 (we hope!)
b = torch.tensor([2.0, 0.0]) # Shape: [2]

# The Linear Transformation Formula: y = (x @ W) + b
# '@' is the symbol for matrix multiplication in Python
y_manual = (x @ W) + b

print(f"Input (x) shape: {x.shape}")
print(f"Weights (W) shape: {W.shape}")
print(f"Bias (b) shape: {b.shape}")
print(f"Manual Output (y_manual): \n{y_manual}")
print(f"Manual Output shape: {y_manual.shape}\n")


# =====================================================================
# 2. THE PYTORCH WAY (nn.Linear)
# =====================================================================
print("--- 2. The PyTorch Way (nn.Linear) ---")

# In PyTorch, we don't write out the matrices by hand. 
# We use nn.Linear. Let's create one that takes 3 inputs and gives 2 outputs.
torch.manual_seed(42) # For reproducible random numbers
linear_layer = nn.Linear(in_features=3, out_features=2)

print("PyTorch automatically created random Weights and Biases for us:")
print(f"Layer Weights:\n{linear_layer.weight.data}")
# Notice the shape of PyTorch's weights is [out_features, in_features] which is [2, 3]
print(f"Layer Bias:\n{linear_layer.bias.data}\n")

# To pass data through, we just "call" the layer like a function!
y_pytorch = linear_layer(x)

print(f"PyTorch Output (y_pytorch): \n{y_pytorch}")

print("\n-------------------------------------------------")
print("Takeaway:")
print("nn.Linear(3, 2) is just a clean, optimized wrapper around:")
print("output = (input @ weights.T) + bias")
print("It's just matrix multiplication in disguise!")
print("-------------------------------------------------")
