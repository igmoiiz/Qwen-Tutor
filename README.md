<div align="center">

# 🧮 AI Math Tutor — Fine-Tuning Qwen2.5-1.5B on GSM8K

**Fine-tuning a 1.5-billion-parameter language model to solve grade-school math problems step-by-step using QLoRA and supervised fine-tuning.**

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg?style=flat)](./LICENSE.md)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HF_Transformers-4.56.1-yellow.svg)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-0.17.1-green.svg)](https://huggingface.co/docs/peft)
[![TRL](https://img.shields.io/badge/TRL-0.21.0-purple.svg)](https://huggingface.co/docs/trl)

*Authored by **Moiz Baloch** · [moaiz3110@gmail.com](mailto:moaiz3110@gmail.com)*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Model](#model)
- [Training Pipeline](#training-pipeline)
- [Hyperparameter Reference](#hyperparameter-reference)
- [Terminology Glossary](#terminology-glossary)
- [Hardware Requirements](#hardware-requirements)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project implements a complete **Supervised Fine-Tuning (SFT)** pipeline that adapts the **Qwen2.5-1.5B-Instruct** language model into a step-by-step math tutor. The model is trained on the **GSM8K** benchmark dataset — a collection of 8,500+ grade-school math word problems — using the **QLoRA** technique (Quantized Low-Rank Adaptation).

The result is a lightweight, specialized model that:
- Reads a natural-language math word problem
- Produces a fully explained, step-by-step solution
- Concludes with the final numerical answer clearly marked

The entire fine-tuning run is designed to fit within a **single NVIDIA Tesla T4 GPU (16 GB VRAM)** — exactly the hardware available in free Kaggle and Google Colab notebooks — making this an accessible, reproducible experiment.

---

## Key Concepts

| Concept | Short Explanation |
|---|---|
| **SFT** | Supervised Fine-Tuning — teaching a pre-trained model new behavior using labeled examples |
| **QLoRA** | Quantized LoRA — combines 4-bit quantization and LoRA for memory-efficient fine-tuning |
| **LoRA** | Low-Rank Adaptation — adds tiny trainable layers to a frozen model |
| **Quantization** | Compressing model weights from 16-bit to 4-bit floats to reduce memory |
| **Chat Template** | A structured prompt format (System, User, Assistant) the model was originally trained on |
| **Instruction Tuning** | Training a model to follow natural-language instructions, not just predict text |
| **Tokenization** | Converting raw text into numeric token IDs the model can process |
| **Attention Mask** | A binary mask telling the model which tokens are real (1) vs. padding (0) |
| **Gradient Accumulation** | Simulating a larger batch size by accumulating gradients over multiple steps |
| **BF16** | Brain Float 16 — a numeric format that speeds up training on modern GPUs |

---

## Project Architecture

```
AI-Math-Tutor-Finetuning/
|
+-- finetuned-math-tutor-qwen.ipynb   <- Main training notebook
+-- README.md                          <- This file
+-- GLOSSARY.md                        <- Extended terminology reference
+-- TRAINING_CONFIG.md                 <- Detailed hyperparameter documentation
+-- LICENSE                            <- Proprietary license (read before use)
+-- math_tutor_qwen/                   <- Output directory (created during training)
    +-- checkpoint-epoch-1/
    +-- checkpoint-epoch-2/
```

---

## Dataset

### GSM8K — Grade School Math 8K

| Property | Value |
|---|---|
| **Name** | GSM8K (Grade School Math 8,000) |
| **Source** | OpenAI |
| **HuggingFace Hub** | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| **Direct Link** | https://huggingface.co/datasets/openai/gsm8k |
| **Paper** | [Training Verifiers to Solve Math Word Problems (Cobbe et al., 2021)](https://arxiv.org/abs/2110.14168) |
| **Training samples** | 7,473 |
| **Test samples** | 1,319 |
| **Columns** | `question`, `answer` |
| **Configuration used** | `"main"` |

### Data Format

Each sample contains:
- **`question`** — A natural-language grade-school math word problem.
- **`answer`** — A step-by-step solution with intermediate calculations annotated using `<<expression=result>>` markers, followed by the final answer after `#### `.

**Example sample:**

```
QUESTION:
Natalia sold clips to 48 of her friends in April, and then she sold half as
many clips in May. How many clips did Natalia sell altogether in April and May?

ANSWER:
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72
```

### Why GSM8K?

GSM8K is the standard benchmark for evaluating mathematical reasoning in language models. It requires:
- **Multi-step arithmetic reasoning** across 2-8 steps per problem
- **Natural-language understanding** to extract relevant numbers and relationships
- **Chain-of-thought generation** — the model must show its work, not just produce an answer

---

## Model

### Qwen2.5-1.5B-Instruct

| Property | Value |
|---|---|
| **Model Name** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **HuggingFace Hub** | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
| **Parameters** | ~1.5 Billion |
| **Architecture** | Decoder-only Transformer |
| **Context Length** | 131,072 tokens |
| **Vocabulary Size** | 151,643 tokens |
| **Tokenizer** | `Qwen2TokenizerFast` |
| **Developer** | Alibaba Cloud (Qwen Team) |

**Why Qwen2.5-1.5B-Instruct?**
- Small enough to fine-tune on free-tier GPU hardware
- Already instruction-tuned — it understands the System/User/Assistant conversational format
- Strong baseline math reasoning for its size
- Uses BF16 natively for efficient training

---

## Training Pipeline

### Step 1 — Environment Setup

```bash
pip install -U transformers==4.56.1 trl==0.21.0 peft==0.17.1 accelerate==1.10.1
```

Pinned versions ensure full reproducibility. Key libraries:
- **`transformers`** — Model loading, tokenization, training arguments
- **`trl`** — `SFTTrainer`, the high-level supervised fine-tuning wrapper
- **`peft`** — LoRA configuration and model wrapping
- **`accelerate`** — Distributed training and device management
- **`bitsandbytes`** — 4-bit quantization backend

---

### Step 2 — Imports and Version Check

The notebook verifies that:
- CUDA is available (`torch.cuda.is_available()`)
- The GPU supports BF16 (`torch.cuda.is_bf16_supported()`)
- Correct library versions are loaded
- The training GPU (Tesla T4) is detected

---

### Step 3 — Dataset Loading and Exploration

```python
dataset = load_dataset("openai/gsm8k", "main")
```

The `"main"` configuration contains human-written solutions. The dataset is explored to understand its structure, column names, and sample content before any processing.

---

### Step 4 — Tokenizer Initialization

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
```

The **Qwen2TokenizerFast** tokenizer uses Byte-Pair Encoding (BPE) and contains special tokens for structured chat formatting:

| Special Token | Purpose |
|---|---|
| `<|im_start|>` | Marks the start of a conversation turn |
| `<|im_end|>` | Marks the end of a conversation turn (also EOS) |
| `<|endoftext|>` | Padding token |

---

### Step 5 — Chat Template Formatting

Each example is formatted into the **ChatML** conversation structure:

```
<|im_start|>system
You are a helpful math tutor. Solve the user's math problem step by step.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

**Why this matters:** The Qwen2.5-Instruct model was pre-trained on conversational data. Training on raw question/answer pairs would cause it to forget its assistant behavior (**catastrophic forgetting**). Using the chat template preserves **instruction tuning**.

---

### Step 6 — Tokenization and Length Analysis

```python
MAX_LENGTH = 512
```

A length analysis is performed on the full training set to verify 512 tokens is sufficient:

| Statistic | Value |
|---|---|
| Shortest sequence | ~50 tokens |
| 90th percentile | < 300 tokens |
| 95th percentile | < 400 tokens |
| 99th percentile | < 512 tokens |

---

### Step 7 — Quantization (BitsAndBytes)

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

| Parameter | Value | Meaning |
|---|---|---|
| `load_in_4bit` | `True` | Load weights in 4-bit precision (~75% memory reduction) |
| `bnb_4bit_quant_type` | `"nf4"` | Normal Float 4 — optimal for neural network weights |
| `bnb_4bit_compute_dtype` | `bfloat16` | Use BF16 for forward/backward computations |
| `bnb_4bit_use_double_quant` | `True` | Quantize the quantization constants for extra savings |

---

### Step 8 — Model Loading

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
```

`device_map="auto"` automatically places model layers across available GPUs, managed by the `accelerate` library.

---

### Step 9 — LoRA Configuration

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

| Parameter | Value | Meaning |
|---|---|---|
| `r` | `16` | Rank — inner dimension of LoRA matrices |
| `lora_alpha` | `32` | Scaling factor (effective scale = alpha / r = 2.0) |
| `lora_dropout` | `0.05` | Dropout to prevent overfitting |
| `target_modules` | attention projections | Inject LoRA into Q, K, V, O projections only |

**Trainable parameters:** Approximately 5-15 million out of 1.5 billion total — roughly **0.5-1% of the model**.

---

### Step 10 — Training Configuration

```python
training_args = TrainingArguments(
    output_dir="./math_tutor_qwen",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,    # Effective batch = 8
    learning_rate=2e-4,
    bf16=True,
    optim="paged_adamw_8bit",
    save_strategy="epoch",
    eval_strategy="epoch",
)
```

| Parameter | Value | Rationale |
|---|---|---|
| `num_train_epochs` | `2` | Two full passes over the training set |
| `per_device_train_batch_size` | `2` | T4 VRAM constraint |
| `gradient_accumulation_steps` | `4` | Effective batch size = 8 |
| `learning_rate` | `2e-4` | Standard LoRA learning rate |
| `bf16` | `True` | Mixed-precision training |
| `optim` | `"paged_adamw_8bit"` | Memory-efficient optimizer |

---

### Step 11 — SFT Trainer and Training

```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
    processing_class=tokenizer,
)
trainer.train()
```

`SFTTrainer` handles tokenization internally, expecting the `"text"` column from `formatted_dataset`. The pre-tokenized dataset is intentionally NOT used here because `SFTTrainer` manages sequence packing and response-only loss masking.

---

## Hyperparameter Reference

| Hyperparameter | Value | Typical Range | Effect of Increasing |
|---|---|---|---|
| LoRA rank `r` | 16 | 4 – 64 | More capacity, more VRAM |
| LoRA alpha | 32 | 8 – 128 | Stronger LoRA signal |
| Learning rate | 2e-4 | 1e-5 – 5e-4 | Faster learning, risk of instability |
| Batch size | 2 | 1 – 8 | Better gradients, more VRAM |
| Grad. accumulation | 4 | 2 – 16 | Larger effective batch, more stable |
| Epochs | 2 | 1 – 5 | More exposure, risk of overfitting |
| Max sequence length | 512 | 256 – 2048 | Handles longer problems, more VRAM |

---

## Terminology Glossary

> A comprehensive standalone glossary is available in [GLOSSARY.md](./GLOSSARY.md).

| Term | Definition |
|---|---|
| **LLM** | Large Language Model — a neural network trained on vast text corpora |
| **Fine-Tuning** | Continuing to train a pre-trained model on a smaller, task-specific dataset |
| **Pre-Training** | Initial large-scale training on internet-scale text data |
| **Instruction Tuning** | Fine-tuning with System/User/Assistant pairs so the model follows instructions |
| **LoRA** | Low-Rank Adaptation — adds small trainable matrices to frozen model weights |
| **QLoRA** | Quantized LoRA — LoRA on a 4-bit quantized base model |
| **Quantization** | Reducing numerical precision of weights (16-bit to 4-bit) to save memory |
| **NF4** | Normal Float 4 — a 4-bit data type optimized for normally-distributed weights |
| **BF16** | Brain Float 16 — 16-bit float with wide exponent range, better for training |
| **VRAM** | Video RAM — dedicated memory on a GPU |
| **Tokenization** | Splitting text into sub-word units and mapping them to integer IDs |
| **BPE** | Byte-Pair Encoding — tokenization algorithm used by Qwen |
| **Attention Mask** | Binary tensor (0/1) indicating which tokens the model should attend to |
| **Chat Template** | Structured prompt format preserving conversational context |
| **ChatML** | Chat Markup Language — the im_start/im_end format used by Qwen |
| **SFT** | Supervised Fine-Tuning — training with labeled (input, output) pairs |
| **SFTTrainer** | HuggingFace TRL class simplifying SFT with automatic tokenization |
| **PEFT** | Parameter-Efficient Fine-Tuning — trains only a small subset of parameters |
| **Gradient Accumulation** | Accumulating gradients over N steps before updating, simulating larger batch |
| **Mixed Precision** | Using lower-precision floats for speed while keeping FP32 for stability |
| **AdamW** | Adam optimizer with weight decay — standard for fine-tuning transformers |
| **Paged AdamW** | BitsAndBytes variant that pages optimizer states to CPU when GPU is full |
| **Catastrophic Forgetting** | A model losing previously learned capabilities when trained on new data |
| **Chain-of-Thought** | Prompting or training a model to produce step-by-step reasoning |
| **GSM8K** | Grade School Math 8K — OpenAI benchmark of 8,500 word problems |
| **Checkpoint** | A saved snapshot of model weights and optimizer state during training |
| **KV Cache** | Key-Value cache for faster inference; disabled during training |
| **EOS Token** | End-of-Sequence token — signals the model to stop generating |
| **PAD Token** | Padding token used to align sequences in a batch |
| **Causal LM** | Causal Language Modeling — next-token prediction (GPT-style) |
| **Double Quantization** | Quantizing the quantization constants for additional memory savings |

---

## Hardware Requirements

| Component | Minimum | Used in Project |
|---|---|---|
| GPU | NVIDIA 8 GB VRAM | Tesla T4 (16 GB VRAM) |
| CUDA | 11.8+ | CUDA 12.8 |
| RAM | 16 GB | Kaggle/Colab default |
| Storage | 10 GB free | Model weights + checkpoints |
| Python | 3.10+ | Python 3.12.13 |

**Compatible Free Platforms:**
- [Kaggle Notebooks](https://www.kaggle.com/) — T4 x 2 GPUs, 30 hrs/week
- [Google Colab](https://colab.research.google.com/) — T4 GPU (free tier)

---

## Usage

### Running on Kaggle (Recommended)

1. Create a new Kaggle Notebook
2. Enable **GPU T4 x 2** in Settings → Accelerator
3. Upload `finetuned-math-tutor-qwen.ipynb` or import from GitHub
4. Run all cells sequentially

### Running on Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `finetuned-math-tutor-qwen.ipynb`
3. Set Runtime → Change runtime type → **T4 GPU**
4. Run all cells sequentially

### Local Setup

```bash
# Clone this repository
git clone <repository-url>
cd AI-Math-Tutor-Finetuning

# Install dependencies
pip install transformers==4.56.1 trl==0.21.0 peft==0.17.1 accelerate==1.10.1 bitsandbytes

# Launch Jupyter
jupyter notebook finetuned-math-tutor-qwen.ipynb
```

> **Note:** Local execution requires a CUDA-capable GPU with at least 8 GB VRAM and CUDA 11.8+.

---

## License

This project is released under a **Proprietary License**. All rights are reserved by the author.

**You MAY NOT** use, copy, modify, distribute, or build upon this Software without prior explicit written consent of the author.

**If consent is granted**, you MUST provide full attribution including the author's name, email, and a link to this repository in any publication, project, or derivative work.

See the full [LICENSE](./LICENSE) file for complete terms.

---

## Contact

**Moiz Baloch**

| Channel | Details |
|---|---|
| Email | moaiz3110@gmail.com |
| Phone | +92 306 789 2235 |

For licensing inquiries, collaboration requests, or questions about this project, please reach out via email with a clear subject line describing your intent.

---

*© 2026 Moiz Baloch — All Rights Reserved*
