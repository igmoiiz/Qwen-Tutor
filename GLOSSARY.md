# Glossary of Terms — AI Math Tutor Fine-Tuning Project

> **Author:** Moiz Baloch  
> **Contact:** moaiz3110@gmail.com | +92 306 789 2235  
> **Project:** Fine-tuning Qwen2.5-1.5B-Instruct on GSM8K using QLoRA

This glossary provides detailed, beginner-friendly explanations of every key term used in this project. Terms are grouped thematically for easier navigation.

---

## Table of Contents

- [Core Machine Learning Concepts](#core-machine-learning-concepts)
- [Large Language Models (LLMs)](#large-language-models-llms)
- [Fine-Tuning Techniques](#fine-tuning-techniques)
- [Memory Optimization](#memory-optimization)
- [Tokenization and Data](#tokenization-and-data)
- [Training Infrastructure](#training-infrastructure)
- [Numerical Precision and Optimization](#numerical-precision-and-optimization)
- [Dataset Specific Terms](#dataset-specific-terms)
- [Model Architecture](#model-architecture)
- [Evaluation Concepts](#evaluation-concepts)

---

## Core Machine Learning Concepts

### Neural Network
A computational system loosely inspired by the human brain. It consists of layers of mathematical functions ("neurons") that transform inputs into outputs. Modern LLMs are enormous neural networks with billions of parameters.

### Parameter
A single trainable number inside a neural network. During training, these numbers are adjusted to make the model perform better. Qwen2.5-1.5B has approximately 1,500,000,000 (1.5 billion) parameters.

### Training
The process of feeding examples to a model and adjusting its parameters using backpropagation to minimize a loss function. After training, the model has "learned" patterns in the data.

### Inference
Using a trained model to generate outputs for new inputs, without adjusting any parameters. When you ask the fine-tuned math tutor a question, that is inference.

### Loss Function
A mathematical function that measures how wrong the model's predictions are. During training, the goal is to minimize this number. For language models, the typical loss is **cross-entropy loss** over next-token predictions.

### Backpropagation
The algorithm that computes how much each parameter contributed to the loss and how it should be adjusted. It works by applying the chain rule of calculus backwards through the network.

### Gradient
The direction and magnitude of the steepest increase in the loss. The optimizer moves parameters in the *opposite* direction of the gradient (gradient descent) to reduce the loss.

### Overfitting
When a model memorizes the training data so well that it performs poorly on new, unseen data. The model learns the "noise" in training data rather than the underlying patterns. Regularization techniques like dropout help prevent this.

### Epoch
One complete pass through the entire training dataset. In this project, we train for 2 epochs, meaning the model sees every training example twice.

### Batch
A subset of training examples processed together in one forward and backward pass. Using batches (rather than one example at a time) makes training more stable and efficient.

---

## Large Language Models (LLMs)

### LLM (Large Language Model)
A neural network with billions of parameters, trained on enormous amounts of text data to understand and generate human language. Examples include GPT-4, Llama, Gemini, and Qwen.

### Pre-Training
The initial phase of training an LLM on a massive corpus of internet text (books, websites, code, etc.). The model learns general language understanding, world knowledge, and reasoning capabilities. Pre-training is extremely expensive — costing millions of dollars and requiring thousands of GPUs.

### Fine-Tuning
Continuing to train an already pre-trained model on a smaller, task-specific dataset. The model starts with its general knowledge from pre-training and learns to specialize. Fine-tuning is much cheaper than pre-training.

### Instruction Tuning
A form of fine-tuning where the model is trained on (instruction, response) pairs formatted as conversations. This teaches the model to follow instructions, be helpful, and respond appropriately to questions. The Qwen2.5-Instruct models have already undergone instruction tuning.

### Catastrophic Forgetting
A phenomenon where a neural network "forgets" previously learned skills when trained on new data. For example, if you fine-tune on raw question/answer pairs without using the model's expected chat format, it may forget how to behave as a conversational assistant. Using the correct chat template mitigates this.

### Decoder-Only Transformer
The architecture used by almost all modern LLMs (GPT, Llama, Qwen). The model only looks at past tokens when predicting the next token (causal/autoregressive). This is opposed to encoder-decoder models (like T5) that see the full input before generating output.

### Causal Language Modeling (Causal LM)
The training objective where the model learns to predict the next token given all previous tokens. Given "The cat sat on the", it predicts "mat". This is how all GPT-style models are trained.

### Autoregressive Generation
The process of generating text by producing one token at a time, where each new token is conditioned on all previously generated tokens. The model generates "The" then "cat" then "sat" and so on.

### Context Window / Context Length
The maximum number of tokens a model can consider at once. Qwen2.5-1.5B supports 131,072 tokens. This includes both the input (prompt) and the generated output.

---

## Fine-Tuning Techniques

### SFT (Supervised Fine-Tuning)
Fine-tuning using labeled examples where both the input and the desired output are known. The model is trained to produce the desired output given the input. In this project, inputs are math questions and outputs are step-by-step solutions.

### PEFT (Parameter-Efficient Fine-Tuning)
A family of techniques that train only a small fraction of the model's parameters rather than all of them. This dramatically reduces memory and compute requirements while achieving quality close to full fine-tuning.

### LoRA (Low-Rank Adaptation)
A PEFT technique that injects small trainable matrices into the model's existing weight matrices. Instead of modifying the full weight matrix W (huge), LoRA adds two small matrices A and B such that the update is A × B (much smaller). The original weights are frozen; only A and B are trained.

**Mathematical intuition:**  
Original: `W` (d × d matrix, frozen)  
LoRA update: `W + (A × B)` where A is (d × r) and B is (r × d), with r << d  
Only A and B are trained, saving ~99% of the parameters.

### QLoRA (Quantized LoRA)
The combination of LoRA with 4-bit quantization of the base model. The base model is loaded in 4-bit precision (saving memory), while LoRA adapters are in full precision. This enables fine-tuning large models on consumer hardware.

### Rank (r) in LoRA
The inner dimension of the LoRA matrices. Higher rank means the LoRA matrices can represent more complex updates, but use more memory. r=16 is a common starting point.

### Alpha in LoRA
A scaling factor applied to the LoRA output: `output *= alpha / r`. It controls how strongly the LoRA adapters influence the model's behavior. Setting alpha = 2r (e.g., alpha=32, r=16) means the effective scale is 2.0.

### LoRA Dropout
A regularization technique applied to LoRA layers. During training, a fraction of LoRA activations are randomly set to zero, preventing the adapters from overfitting to the training data.

### Target Modules
The specific layers in the model where LoRA adapters are injected. In this project: `q_proj`, `k_proj`, `v_proj`, `o_proj` — the four projection matrices inside each transformer attention block. These are chosen because attention layers are the most critical for task adaptation.

---

## Memory Optimization

### VRAM (Video RAM)
The dedicated memory chip on a GPU. All model weights, activations, gradients, and optimizer states must fit in VRAM during training. A Tesla T4 has 16 GB VRAM.

### Quantization
The process of reducing the numerical precision used to store model weights. Lower precision = smaller numbers = less memory.

| Format | Bits per value | Memory for 1.5B params |
|---|---|---|
| FP32 | 32 bits | ~6 GB |
| BF16 / FP16 | 16 bits | ~3 GB |
| INT8 | 8 bits | ~1.5 GB |
| NF4 (4-bit) | 4 bits | ~0.75 GB |

### NF4 (Normal Float 4)
A specialized 4-bit data type designed specifically for neural network weights. Unlike naive 4-bit integers, NF4 distributes its representable values according to a normal (Gaussian) distribution — which is exactly the distribution that pre-trained neural network weights follow. This makes NF4 more accurate than INT4 for the same number of bits.

### Double Quantization
An additional memory optimization in QLoRA where the quantization constants (scale factors used to perform quantization) are themselves quantized. This saves approximately 0.37 additional bits per parameter on average.

### BitsAndBytes
A Python library by Tim Dettmers that implements efficient quantization and 8-bit/4-bit operations for PyTorch. It provides the `BitsAndBytesConfig` used in this project.

### Paged Optimizer States
A technique where optimizer states (which can be as large as the model weights themselves) are stored in CPU RAM and paged to GPU VRAM only when needed. This prevents out-of-memory errors during training. The `paged_adamw_8bit` optimizer uses this technique.

### KV Cache
During inference (text generation), the model computes Key and Value tensors for each token and caches them to avoid recomputation on subsequent tokens. This cache is **disabled during training** (`use_cache=False`) because it interferes with gradient computation.

---

## Tokenization and Data

### Token
The basic unit of text that a language model processes. Tokens can be whole words, sub-words, or individual characters. The model never sees raw text — it sees sequences of integer token IDs.

### Tokenizer
A component that converts raw text into sequences of token IDs (and back). The tokenizer has a fixed vocabulary mapping words/sub-words to numbers.

### Vocabulary
The complete set of tokens a tokenizer knows. Qwen's tokenizer has a vocabulary of 151,643 tokens.

### BPE (Byte-Pair Encoding)
The tokenization algorithm used by Qwen. Starting with individual characters, it iteratively merges the most frequent pairs of adjacent tokens until reaching the desired vocabulary size. This results in common words being single tokens while rare words are split into sub-word pieces.

### Input IDs
The sequence of integer token IDs corresponding to tokenized text. For example, "John has 25 apples" might become [13079, 702, 220, 17, 20, 40676, 13].

### Attention Mask
A binary sequence (same length as input IDs) where 1 means "attend to this token" and 0 means "ignore this token (it's padding)". During training with variable-length sequences, shorter sequences are padded to a fixed length and the attention mask tells the model to ignore padding tokens.

### Padding
Adding dummy tokens (pad tokens) to the end of a short sequence to make it the same length as the longest sequence in a batch. Transformers require all sequences in a batch to be the same length. Padding tokens are masked out via the attention mask.

### Truncation
When a sequence is longer than `max_length`, it is cut down to exactly `max_length` tokens. In this project, `max_length=512`. The length analysis confirms very few samples exceed 512 tokens.

### Chat Template
A formatting specification that defines how multi-turn conversations should be structured as text before tokenization. Qwen uses the **ChatML** format with `<|im_start|>` and `<|im_end|>` special tokens.

### ChatML (Chat Markup Language)
The conversation format used by Qwen models:

```
<|im_start|>system
[system message]<|im_end|>
<|im_start|>user
[user message]<|im_end|>
<|im_start|>assistant
[assistant response]<|im_end|>
```

### System Prompt / System Message
A message prepended to the conversation that defines the model's persona, behavior, and task. In this project: *"You are a helpful math tutor. Solve the user's math problem step by step."*

### Special Tokens
Tokens with specific syntactic meaning that appear in the vocabulary but are not regular words. Examples: EOS (end of sequence), PAD (padding), `<|im_start|>`, `<|im_end|>`.

### EOS Token (End of Sequence)
A special token that signals to the model it should stop generating. For Qwen, this is `<|im_end|>`.

### PAD Token
The token used for padding shorter sequences. For Qwen, this is `<|endoftext|>` (ID: 151643).

---

## Training Infrastructure

### SFTTrainer
The `SFTTrainer` class from HuggingFace TRL (Transformer Reinforcement Learning). It wraps the base `Trainer` class with SFT-specific features:
- Automatically tokenizes text from a `"text"` column
- Handles response-only loss masking (only compute loss on the assistant's response)
- Supports sequence packing for efficiency

### TrainingArguments
A HuggingFace class that bundles all training hyperparameters and configuration. Passed to the `Trainer` or `SFTTrainer`.

### Gradient Accumulation
Instead of updating model parameters after every batch, gradients are accumulated (summed) over N batches before performing one update. This simulates a larger effective batch size without requiring more VRAM.

- Physical batch size: 2
- Gradient accumulation steps: 4
- Effective batch size: 2 × 4 = 8

### Mixed Precision Training
Using lower-precision floating-point numbers (BF16 or FP16) for most computations, while keeping critical operations in FP32. This speeds up training and reduces memory without significantly hurting accuracy.

### `device_map="auto"`
An `accelerate` feature that automatically distributes model layers across available GPUs (and CPU if needed). On a single T4, all layers go to GPU 0.

### Checkpoint
A saved snapshot of the model's weights, optimizer state, and training progress at a specific point during training. Checkpoints allow training to be resumed if interrupted. In this project, checkpoints are saved at the end of each epoch.

### `enable_input_require_grads()`
A PEFT method that enables gradient flow through the input embedding layer. Required when using gradient checkpointing or quantization with LoRA to ensure gradients can propagate back to the LoRA adapters.

---

## Numerical Precision and Optimization

### FP32 (Float 32-bit)
The standard 32-bit floating-point format. High precision but uses the most memory. Used for optimizer states and critical computations.

### FP16 (Float 16-bit)
16-bit floating-point. Half the memory of FP32. Faster on NVIDIA GPUs. Risk of numerical overflow for very large or very small values.

### BF16 (Brain Float 16)
16-bit floating-point format developed by Google Brain. Same memory as FP16 but with a wider exponent range (8 bits vs FP16's 5 bits), making it more numerically stable for training. Supported natively on A100, and in limited capacity on T4.

### AdamW Optimizer
A variant of the Adam (Adaptive Moment Estimation) optimizer with **weight decay** regularization. Standard for fine-tuning transformers. Maintains a moving average of gradients and squared gradients to adaptively scale learning rates per parameter.

### Paged AdamW 8-bit
An 8-bit quantized version of AdamW from the bitsandbytes library. Uses paging to move optimizer states to CPU RAM when GPU memory is scarce. Reduces VRAM required for the optimizer from ~6 GB to ~1.5 GB.

### Learning Rate
Controls how large each parameter update step is. Too high: training becomes unstable (loss diverges). Too low: training is slow or gets stuck. For LoRA fine-tuning, 2e-4 (0.0002) is a standard starting value.

---

## Dataset Specific Terms

### GSM8K (Grade School Math 8K)
A dataset of 8,500 grade-school math word problems created by OpenAI. The "K" stands for thousand. It is widely used as a benchmark for evaluating mathematical reasoning in LLMs.

**Citation:** Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, et al. "Training Verifiers to Solve Math Word Problems." arXiv:2110.14168 (2021).

### Chain-of-Thought (CoT)
A reasoning technique where the model generates intermediate steps before the final answer, rather than jumping straight to the result. GSM8K answers contain full chains of thought, training the model to "think out loud."

### `#### ` Annotation
GSM8K uses `#### ` (four hash marks followed by a space) to denote the final numerical answer at the end of each solution. This makes it easy to programmatically extract and evaluate the final answer.

### `<<expression=result>>` Annotation
GSM8K uses double angle-bracket annotations to mark intermediate calculations inline within the reasoning chain. For example: `48/2 = <<48/2=24>>24`. This helps verify computational steps.

### "main" Configuration
One of two configurations of the GSM8K dataset on HuggingFace. "main" contains standard human-written solutions. The alternative "socratic" contains solutions written in a Socratic dialogue format.

---

## Model Architecture

### Transformer
The neural network architecture underlying virtually all modern LLMs. Introduced in "Attention Is All You Need" (Vaswani et al., 2017). Key components: multi-head self-attention, feed-forward networks, residual connections, and layer normalization.

### Self-Attention
A mechanism that allows each token in a sequence to "attend to" (gather information from) every other token. This enables the model to understand long-range dependencies and context.

### Multi-Head Attention
Running multiple self-attention operations in parallel ("heads"), each learning to focus on different aspects of the input. The outputs are concatenated and projected.

### Query (Q), Key (K), Value (V) Projections
The three linear transformations applied to the input within each attention head:
- **Query** — "What am I looking for?"
- **Key** — "What do I have to offer?"
- **Value** — "What information do I carry?"
Attention weights are computed as softmax(Q × K^T / sqrt(d)), then applied to V.

### Output Projection (o_proj)
A linear layer that combines the outputs of all attention heads back into the model's hidden dimension. LoRA is applied to this layer.

### Feed-Forward Network (FFN / MLP)
A two-layer fully connected network applied to each token independently after the attention layer. Contains the majority of a transformer's parameters.

### Residual Connection
A shortcut that adds the input of a layer directly to its output. This helps with gradient flow during training and allows very deep networks to be trained effectively.

### Layer Normalization
A normalization technique applied within each transformer block to stabilize training by normalizing the activations.

---

## Evaluation Concepts

### Perplexity
A measure of how well a language model predicts a held-out test set. Defined as exp(average cross-entropy loss). Lower perplexity = better model. A perplexity of 1 would mean perfect prediction.

### Validation / Evaluation Set
A portion of data held out from training and used to monitor the model's performance on unseen examples during training. In this project, the GSM8K test set (1,319 examples) serves as the evaluation set.

### Accuracy on GSM8K
The standard metric: percentage of problems where the model's final numerical answer (after ####) matches the ground-truth answer. State-of-the-art large models achieve 90%+; fine-tuned 1.5B models typically achieve 40-60%.

---

*© 2026 Moiz Baloch — All Rights Reserved*  
*See [LICENSE](./LICENSE) for usage terms.*
