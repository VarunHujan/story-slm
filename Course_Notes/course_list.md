# 🗺️ The Ultimate SLM / LLM Mastery Curriculum
*(Updated based on advanced AI peer-review!)*

This is the complete roadmap from absolute beginner concepts to state-of-the-art AI research topics. 

### Phase 0: Environment & Math Refresher
1. Environment Setup (venv, CUDA check, pip)
2. NumPy Refresher (Arrays, Shapes, Broadcasting)

### Phase 1: Deep Learning Foundations
3. What is a Neural Network?
4. Tensors: The Language of PyTorch
5. Matrix Multiplication (Dot Products)
6. Linear Layers (Weights and Biases)
7. Activation Functions (ReLU, GELU, Sigmoid)
8. Forward Propagation
9. The Softmax Function and Logits *(Moved here so Loss makes sense!)*
10. The Loss Function (Cross-Entropy Loss)
11. Calculus in ML: Gradients and Derivatives
12. Backpropagation (How the model learns)
13. Gradient Descent & Optimizers (AdamW)
14. Learning Rate Schedules (Warmup + Cosine Decay) & Weight Decay
15. Hyperparameters and Epochs
16. Overfitting vs Underfitting (Generalization)
17. Batches and Data Loaders
18. Experiment Tracking (Weights & Biases / TensorBoard)

### Phase 2: Natural Language Processing (NLP) Basics
19. Pretraining Data Curation (Filtering, Deduplication, MinHash)
20. What is Tokenization?
21. Character-level Tokenization
22. Sub-word Tokenization (BPE - Byte Pair Encoding)
23. Training Your Own Tokenizer (Fitting merges with SentencePiece)
24. Word2Vec and the concept of Semantic Space
25. Neural Embeddings (Mapping words to vectors)
26. Curse of Dimensionality & Clustering
27. Context Windows
28. Recurrent Neural Networks (RNNs) & LSTMs (The old way)

### Phase 3: The Transformer Architecture (The Core of LLMs)
29. The "Attention Is All You Need" Paper
30. Self-Attention (Queries, Keys, Values)
31. Masked Self-Attention (Preventing looking into the future)
32. Multi-Head Attention (Looking at multiple contexts at once)
33. Scaled Dot-Product Attention
34. Positional Encoding (Absolute vs Relative)
35. The Residual Connection (Skip connections)
36. Layer Normalization (LayerNorm/RMSNorm)
37. The Feed-Forward Network (MLP Block)
38. Transformer Encoders (BERT) vs Decoders (GPT)
39. Temperature, Top-K, and Top-P Sampling

### Phase 4: Building our SLM from Scratch (Hands-On)
40. Setting up the PyTorch `nn.Module`
41. Coding the Tokenizer and DataLoaders
42. Coding the Embedding Table
43. Coding the Attention Head
44. Coding Multi-Head Attention
45. Coding the Transformer Block
46. Assembling the GPT architecture
47. Writing the Training Loop
48. Writing the Generation (Inference) function
49. Saving and Loading `.pt` weights

### Phase 5: Modern Architecture Upgrades
50. RoPE (Rotary Positional Embeddings)
51. Long-Context Extension (YaRN, ALiBi, NTK-aware scaling)
52. SwiGLU Activation Functions
53. GQA (Grouped Query Attention) and MQA
54. Flash Attention (Optimizing GPU memory)
55. Mixture of Experts (MoE) - How GPT-4 works
56. KV Cache (Speeding up generation)
57. Speculative Decoding (Advanced Inference Speedup)
58. Sliding Window Attention

### Phase 6: Scaling Up (Training massive models)
59. The Chinchilla Scaling Laws (Data vs Parameters)
60. Distributed Data Parallel (DDP)
61. Fully Sharded Data Parallel (FSDP)
62. DeepSpeed and ZeRO Optimization
63. Mixed Precision Training (FP16, BF16)
64. Gradient Accumulation
65. Checkpointing and Fault Tolerance

### Phase 7: Instruction Tuning & Alignment (Making it a Chatbot)
66. Base Models vs Instruct Models
67. Synthetic Data Generation (Phi/Orca style distillation)
68. Supervised Fine-Tuning (SFT)
69. Formatting Chat Templates (System/User/Assistant prompts)
70. RLHF (Reinforcement Learning from Human Feedback)
71. Reward Modeling
72. PPO (Proximal Policy Optimization)
73. DPO (Direct Preference Optimization) - The modern RLHF
74. KTO and ORPO (Alternative alignment strategies)

### Phase 8: Efficiency & Democratization (Consumer Hardware)
75. Parameter-Efficient Fine Tuning (PEFT)
76. LoRA (Low-Rank Adaptation)
77. QLoRA (Quantized LoRA)
78. Model Merging (SLERP, Task Arithmetic, Mergekit)
79. Post-Training Quantization (PTQ)
80. GGUF format and llama.cpp
81. AWQ and GPTQ (Advanced Quantization)
82. Model Pruning (Cutting out dead weights)
83. Knowledge Distillation (Teaching a small model with a big model)

### Phase 9: Test-Time Compute, Reasoning & Generation
84. Reasoning & Test-Time Compute (o1/R1 style, GRPO)
85. Best-of-N and Self-Consistency
86. Chain of Thought (CoT) prompting & Tree of Thoughts (ToT)
87. Why do models hallucinate?
88. Grounding models with real data
89. RAG (Retrieval-Augmented Generation) & Vector Databases
90. GraphRAG (Knowledge Graphs + RAG)
91. Agentic Frameworks (LangChain, LlamaIndex)
92. Tool Calling / Function Calling
93. Constitutional AI (Anthropic's approach)

### Phase 10: Production & Evaluation
94. Serving models with vLLM
95. TensorRT-LLM optimization
96. Continuous Batching
97. Benchmarks: MMLU, HumanEval, GSM8K
98. Perplexity (Measuring model confusion)
99. BLEU and ROUGE scores
100. LLM-as-a-Judge evaluation
101. Red Teaming, Security, and Prompt Injection Attacks
102. Continual Learning (Preventing catastrophic forgetting)
103. Multimodal Models (Vision + Text)
104. Deployment to Cloud (AWS/GCP)
