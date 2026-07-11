# Training Configuration Reference

> **Author:** Moiz Baloch  
> **Contact:** moaiz3110@gmail.com | +92 306 789 2235  
> **Project:** Fine-tuning Qwen2.5-1.5B-Instruct on GSM8K  
> **Hardware:** NVIDIA Tesla T4 (16 GB VRAM) — Kaggle / Google Colab

This document provides an exhaustive reference for every configuration decision in the training pipeline, including what each setting does, why it was chosen, and how it can be modified.

---

## Table of Contents

- [Library Versions](#library-versions)
- [Dataset Configuration](#dataset-configuration)
- [Tokenizer Configuration](#tokenizer-configuration)
- [Quantization Configuration (BitsAndBytes)](#quantization-configuration-bitsandbytes)
- [LoRA Configuration (PEFT)](#lora-configuration-peft)
- [Training Arguments (SFT)](#training-arguments-sft)
- [Optimizer Configuration](#optimizer-configuration)
- [Memory Budget Analysis](#memory-budget-analysis)
- [Experimentation Guide](#experimentation-guide)

---

## Library Versions

All library versions are pinned to ensure exact reproducibility.

| Library | Version | Purpose |
|---|---|---|
| `transformers` | 4.56.1 | Model loading, tokenizer, TrainingArguments |
| `trl` | 0.21.0 | SFTTrainer for supervised fine-tuning |
| `peft` | 0.17.1 | LoRA implementation and model wrapping |
| `accelerate` | 1.10.1 | Device management, mixed precision |
| `bitsandbytes` | 0.49.2 | 4-bit quantization and 8-bit optimizers |
| `torch` | 2.10.0+cu128 | PyTorch deep learning framework |
| `datasets` | 5.0.0 | Dataset loading and processing |
| `numpy` | 2.5.1 | Numerical operations |

**Runtime Environment:**
- Python: 3.12.13
- CUDA: 12.8
- GPU: NVIDIA Tesla T4 (16 GB VRAM)
- BF16 Supported: Yes

---

## Dataset Configuration

```python
dataset = load_dataset("openai/gsm8k", "main")
```

| Property | Value |
|---|---|
| Dataset ID | `openai/gsm8k` |
| Configuration | `"main"` |
| Train Split | 7,473 examples |
| Test Split | 1,319 examples |
| Features | `question` (string), `answer` (string) |

### Format Function

```python
def format_example(example):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math tutor. Solve the user's math problem step by step."
        },
        {
            "role": "user",
            "content": example["question"]
        },
        {
            "role": "assistant",
            "content": example["answer"]
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}
```

**Rationale:**
- `add_generation_prompt=False` — We include the full assistant response in training; we do NOT add the bare `<|im_start|>assistant\n` prompt prefix (that is only for inference).
- System prompt instructs the model to behave as a math tutor, preserving its instruction-following capability.

---

## Tokenizer Configuration

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
```

| Property | Value |
|---|---|
| Type | `Qwen2TokenizerFast` |
| Vocabulary size | 151,643 |
| Max model length | 131,072 |
| Padding side | Right |
| Truncation side | Right |
| EOS token | `<|im_end|>` (ID: 151645) |
| PAD token | `<|endoftext|>` (ID: 151643) |

### Tokenization Parameters

```python
MAX_LENGTH = 512

def tokenize_function(example):
    return tokenizer(
        example['text'],
        truncation=True,
        max_length=MAX_LENGTH
    )
```

| Parameter | Value | Reason |
|---|---|---|
| `MAX_LENGTH` | 512 | Covers >99% of GSM8K examples without truncation |
| `truncation` | `True` | Clips sequences exceeding MAX_LENGTH |
| `padding` | Not set (False) | SFTTrainer handles padding internally |

### Token Length Distribution (Training Set)

| Percentile | Token Count |
|---|---|
| Min | ~50 |
| 50th | ~150 |
| 90th | < 300 |
| 95th | < 400 |
| 99th | < 512 |
| Max | ~550 |

**Conclusion:** Setting MAX_LENGTH=512 results in truncation of fewer than 1% of training examples.

---

## Quantization Configuration (BitsAndBytes)

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

### Parameter Details

#### `load_in_4bit=True`
Loads the model weights in 4-bit precision instead of the default 16-bit. Reduces base model memory by approximately 4x.

#### `bnb_4bit_quant_type="nf4"`
Selects the **Normal Float 4 (NF4)** quantization scheme.

NF4 vs alternatives:
| Type | Description | Quality |
|---|---|---|
| `"nf4"` | Quantization levels match normal distribution | Best for neural nets |
| `"fp4"` | Standard 4-bit float | Slightly lower quality |
| INT8 | 8-bit integer | Higher quality, more memory |

NF4 is specifically designed for the approximately normally-distributed weights found in pre-trained neural networks, making it the best choice for QLoRA.

#### `bnb_4bit_compute_dtype=torch.bfloat16`
During forward and backward passes, 4-bit weights are **dequantized** to this dtype for computation. BF16 is preferred over FP16 because:
- Wider exponent range prevents overflow during training
- Native T4 support

#### `bnb_4bit_use_double_quant=True`
Applies a second quantization pass to the quantization scale factors themselves, saving an additional ~0.37 bits per parameter (roughly 500 MB for a 1.5B model).

### Memory Impact of Quantization

| Configuration | Approximate VRAM for 1.5B model |
|---|---|
| FP32 (no quantization) | ~6 GB |
| BF16 (standard) | ~3 GB |
| INT8 | ~1.5 GB |
| NF4 (4-bit) | ~0.75 GB |
| NF4 + Double Quant | ~0.65 GB |

---

## LoRA Configuration (PEFT)

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)
```

### Parameter Details

#### `r=16` (Rank)
The inner dimension of the LoRA adapter matrices. For a weight matrix W of shape (d_out, d_in), LoRA adds:
- Matrix A: shape (r, d_in)
- Matrix B: shape (d_out, r)
- Update: B @ A (of shape d_out × d_in)

Higher rank = more expressive adapter = more memory. Common values: 4, 8, 16, 32, 64.

#### `lora_alpha=32` (Alpha)
The scaling factor for the LoRA output. The effective contribution is scaled by `alpha / r`.

With r=16, alpha=32: scale = 32/16 = **2.0**

This double-scaling means LoRA adapters have a stronger effect per parameter. Common practice: set alpha = 2r.

#### `lora_dropout=0.05`
5% dropout applied to LoRA layer activations during training. Prevents the small set of adapter parameters from overfitting. Typical range: 0.0 – 0.1.

#### `bias="none"`
Whether to train bias terms alongside LoRA. Options:
- `"none"` — Do not train any biases (recommended for most cases)
- `"all"` — Train all biases
- `"lora_only"` — Train only biases in LoRA layers

#### `task_type="CAUSAL_LM"`
Specifies the model type so PEFT can correctly wrap it. CAUSAL_LM means the model does next-token prediction (GPT-style).

#### `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`
The specific linear layers in the model where LoRA adapters are injected:

| Module | Role | Reason to Target |
|---|---|---|
| `q_proj` | Query projection in attention | Controls what the model queries |
| `k_proj` | Key projection in attention | Controls what information is available |
| `v_proj` | Value projection in attention | Controls information content |
| `o_proj` | Output projection of attention | Combines all attention heads |

These are the attention layers. Research shows that adapting attention weights is most impactful for task-specific fine-tuning. MLP layers are left frozen.

### Trainable Parameters Summary

After `get_peft_model()`:
```
trainable params: ~5,570,560  ||  all params: ~1,549,000,000
trainable%: ~0.36%
```

Only ~0.36% of the model is trained, yet the model meaningfully adapts to the math tutoring task.

### Model Preparation

```python
model.config.use_cache = False
model.enable_input_require_grads()
```

**`use_cache = False`:** The KV cache stores intermediate attention states to speed up autoregressive inference. During training, this conflicts with gradient checkpointing and gradient computation. Must be disabled.

**`enable_input_require_grads()`:** Required for gradient computation to flow through the embedding layer when the first module (embeddings) is not directly trainable. This is a PEFT requirement when using quantized models.

---

## Training Arguments (SFT)

```python
training_args = TrainingArguments(
    output_dir="./math_tutor_qwen",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=20,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=False,
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    remove_unused_columns=False,
)
```

### Parameter Details

| Parameter | Value | Notes |
|---|---|---|
| `output_dir` | `"./math_tutor_qwen"` | Directory for checkpoints and final model |
| `num_train_epochs` | `2` | Number of passes through the full training set |
| `per_device_train_batch_size` | `2` | Samples per GPU step (memory constrained) |
| `per_device_eval_batch_size` | `2` | Samples per GPU step during evaluation |
| `gradient_accumulation_steps` | `4` | Effective batch = 2 × 4 = 8 |
| `learning_rate` | `2e-4` | Initial learning rate |
| `logging_steps` | `20` | Log training metrics every 20 optimizer steps |
| `save_strategy` | `"epoch"` | Save checkpoint after each epoch |
| `eval_strategy` | `"epoch"` | Run evaluation after each epoch |
| `fp16` | `False` | Disabled (using BF16 instead) |
| `bf16` | `True` | Brain Float 16 mixed precision |
| `optim` | `"paged_adamw_8bit"` | Memory-efficient 8-bit AdamW optimizer |
| `report_to` | `"none"` | No external logging (no W&B, no TensorBoard) |
| `remove_unused_columns` | `False` | Keep all dataset columns available |

### Effective Batch Size Calculation

```
Effective Batch Size = per_device_train_batch_size × gradient_accumulation_steps × num_gpus
                     = 2 × 4 × 1
                     = 8
```

### Steps Per Epoch

```
Total train examples: 7,473
Per-GPU batch size:   2
Steps per epoch:      7,473 / 2 = 3,736.5 → 3,737 steps
Optimizer updates:    3,737 / 4 = 934 updates per epoch
Total updates (2 epochs): 1,868
```

---

## Optimizer Configuration

### `paged_adamw_8bit`

The standard AdamW optimizer with two modifications:

**8-bit quantization:** Adam maintains two moment estimates (m_t and v_t) for each parameter. These are stored in 8-bit integers rather than 32-bit floats, reducing optimizer state memory by 4x.

| Memory source | FP32 AdamW | Paged AdamW 8-bit |
|---|---|---|
| Model weights (4-bit) | ~0.65 GB | ~0.65 GB |
| LoRA adapters (BF16) | ~0.05 GB | ~0.05 GB |
| Gradient (BF16) | ~0.05 GB | ~0.05 GB |
| Optimizer states (m_t, v_t) | ~0.12 GB | ~0.03 GB |
| Activations (batch) | ~1-2 GB | ~1-2 GB |

**Paging:** When GPU VRAM is insufficient to hold all optimizer states, they are automatically paged to CPU RAM and brought back when needed. This prevents out-of-memory errors.

---

## Memory Budget Analysis

Approximate VRAM usage during training on a single T4:

| Component | Memory |
|---|---|
| Base model weights (NF4 + double quant) | ~0.65 GB |
| LoRA adapters (BF16) | ~0.05 GB |
| Gradients for LoRA (BF16) | ~0.05 GB |
| Optimizer states (paged 8-bit AdamW) | ~0.03 GB |
| Activations (batch=2, seq=512, BF16) | ~2-4 GB |
| Tokenizer and dataset caching | ~0.5 GB |
| PyTorch overhead | ~1-2 GB |
| **Total Estimated** | **~4-7 GB** |
| **T4 VRAM Available** | **16 GB** |
| **Safety Margin** | **~9-12 GB** |

The configuration is designed with substantial headroom to avoid out-of-memory errors.

---

## Experimentation Guide

The following table suggests modifications to experiment with after the baseline run:

### Increasing Quality

| What to change | Suggested value | Trade-off |
|---|---|---|
| `num_train_epochs` | 3 or 4 | Risk of overfitting; monitor eval loss |
| `r` (LoRA rank) | 32 or 64 | More VRAM required |
| `lora_alpha` | 64 or 128 | Stronger LoRA signal |
| `MAX_LENGTH` | 768 or 1024 | Handles longer problems; more VRAM |
| Add MLP to `target_modules` | `"gate_proj"`, `"up_proj"`, `"down_proj"` | More trainable params; slower |

### Reducing Memory

| What to change | Suggested value | Trade-off |
|---|---|---|
| `per_device_train_batch_size` | 1 | Increase grad. accumulation to compensate |
| `gradient_accumulation_steps` | 8 | Larger effective batch; slower steps |
| `r` (LoRA rank) | 4 or 8 | Fewer parameters; potentially lower quality |

### Changing Training Speed

| What to change | Suggested value | Effect |
|---|---|---|
| `logging_steps` | 50 or 100 | Less logging overhead |
| `save_strategy` | `"steps"` + `save_steps=500` | More granular checkpointing |
| `eval_strategy` | `"no"` | Skip evaluation entirely (faster) |

---

*© 2026 Moiz Baloch — All Rights Reserved*  
*See [LICENSE](./LICENSE) for usage terms.*
