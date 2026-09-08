import torch
import torch.nn.functional as F

print("==================================================")
print("Masterclass: Activation Functions (Non-Linearity)")
print("==================================================\n")

# Let's create a tensor of random outputs from a hypothetical Linear Layer.
# We will include large negative numbers, zero, and large positive numbers.
x = torch.tensor([-10.0, -2.0, 0.0, 2.0, 10.0])

print("--- 1. The Raw Input Data ---")
print(f"Raw Output from Linear Layer:\n{x}\n")

# =====================================================================
# 2. SIGMOID
# =====================================================================
print("--- 2. Sigmoid (The Squeezer) ---")
# Sigmoid squeezes everything between 0 and 1.
# Great for probabilities!
sigmoid_out = torch.sigmoid(x)
print(f"Sigmoid Output:\n{sigmoid_out}")
print("Notice: -10 became ~0.0, 0 became exactly 0.5, and 10 became ~1.0\n")

# =====================================================================
# 3. ReLU (Rectified Linear Unit)
# =====================================================================
print("--- 3. ReLU (The Negative Eraser) ---")
# ReLU simply turns all negative numbers to 0, and leaves positives alone.
relu_out = F.relu(x)
print(f"ReLU Output:\n{relu_out}")
print("Notice: All negatives are absolute 0. Positives are untouched.\n")


# =====================================================================
# 4. GELU (Gaussian Error Linear Unit)
# =====================================================================
print("--- 4. GELU (The Modern LLM Standard) ---")
# GELU is like ReLU but smoother. It allows a tiny dip into the negatives.
gelu_out = F.gelu(x)
print(f"GELU Output:\n{gelu_out}")
print("Notice: -2.0 didn't become absolute 0, it became -0.0454! This tiny smooth curve helps LLMs train better.\n")

print("-------------------------------------------------")
print("Takeaway:")
print("Activation functions decide what information 'fires' and ")
print("moves to the next layer. Without them, Neural Networks ")
print("would just be one giant, boring, straight line.")
print("-------------------------------------------------")
