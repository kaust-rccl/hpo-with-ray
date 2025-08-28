# ────────────────────────────── Imports ──────────────────────────────
# Core Python libraries
import os, time, re, string, json
from pathlib import Path
import argparse

# ML and numerical processing
import torch, numpy as np

# Hugging Face datasets + transformers
from datasets import load_dataset, disable_caching
from transformers import (
    BloomForCausalLM,
    BloomTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# ────────────────────────────── Config ──────────────────────────────
MODEL_NAME = "bigscience/bloomz-560m"  # base model used for fine-tuning

# ──────────────────────── Argument Parser ────────────────────────────
def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for one hyperparameter combination.
    Used for grid/manual search. Each run tries a different (lr, bs, wd).
    """
    p = argparse.ArgumentParser(
        prog="bloom_hpo_trial.py",
        description="Fine-tune BLOOM-560M on SQuAD for a single trial."
    )
    p.add_argument("--lr",  type=float, required=True, help="Learning rate")
    p.add_argument("--bs",  type=int,   required=True, help="Per-GPU batch size")
    p.add_argument("--wd",  type=float, required=True, help="Weight decay")
    p.add_argument("--deepspeed", type=str, required=True, default="./config/ds_config.json", help="Path to DeepSpeed config JSON")
    return p.parse_args()

# ─────────────────────── Dataset Loading & Preprocessing ───────────────────────
def load_squad():
    """
    Loads and tokenizes the SQuAD dataset for generative QA using BLOOM.
    - Tokenizer adds prompt: "Question: ... Context: ... Answer:"
    - Answers are the first ground-truth string (or "No Answer").
    - Padding is done to fixed length (512 tokens).
    - Labels have -100 at padding positions (ignored in loss).
    """
    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    dataset = load_dataset("squad")

    def preprocess_function(examples):
        tokenizer.padding_side = "right"  # important for generation tasks

        # Create input prompts and answers
        inputs = [
            f"Question: {q} Context: {c} Answer:"
            for q, c in zip(examples["question"], examples["context"])
        ]
        answers = [a["text"][0] if a["text"] else "No Answer" for a in examples["answers"]]

        # Tokenize inputs and labels
        model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
        labels = tokenizer(answers, truncation=True, padding="max_length", max_length=512)

        # Replace padding tokens with -100 in labels so they’re ignored in the loss
        labels["input_ids"] = [
            [(token if token != tokenizer.pad_token_id else -100) for token in label]
            for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply preprocessing to train and validation
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset, tokenizer

# ──────────────────────── Answer Normalization ────────────────────────
def normalize_answer(s):
    """
    Normalizes a string by:
    - Lowercasing
    - Removing punctuation
    - Removing articles (a, an, the)
    - Removing extra whitespace
    This prepares text for EM/F1 comparison.
    """
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# ──────────────────────── EM and F1 Computation ────────────────────────
def compute_em_and_f1(predicted, ground_truth):
    """
    Given a predicted string and reference answer, compute:
    - Exact Match (EM): 1 if normalized strings match, else 0
    - F1: token-level overlap score
    """
    pred = normalize_answer(predicted)
    gt = normalize_answer(ground_truth)

    em = int(pred == gt)
    pred_tokens = pred.split()
    gt_tokens = gt.split()
    common = set(pred_tokens) & set(gt_tokens)

    if not common:
        return em, 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return em, f1

# ─────────────────────────── Evaluation Loop ───────────────────────────
def evaluate_model(trainer, dataset, tokenizer):
    """
    Generates answers from model and compares them to ground-truth answers.
    Computes average EM and F1 across the validation set.
    """
    em_scores = []
    f1_scores = []

    for example in dataset:
        # Convert inputs to tensors
        input_ids = torch.tensor([example["input_ids"]]).to(trainer.args.device)
        attention_mask = torch.tensor([example["attention_mask"]]).to(trainer.args.device)

        # Run model inference (greedy generation)
        outputs = trainer.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=False  # deterministic
        )

        # Decode generated string
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract predicted answer from full prompt+response
        if "Answer:" in generated_text:
            generated_answer = generated_text.split("Answer:")[-1].strip()
        else:
            generated_answer = generated_text.strip()

        # Get reference answer from tokenized label (strip ignored -100s)
        reference_answer = tokenizer.decode(
            [token for token in example["labels"] if token != -100],
            skip_special_tokens=True
        ).strip()

        # Score it
        em, f1 = compute_em_and_f1(generated_answer, reference_answer)
        em_scores.append(em)
        f1_scores.append(f1)

    return {
        "exact_match": np.mean(em_scores),
        "f1": np.mean(f1_scores)
    }

# ────────────────────────────── Main Script ──────────────────────────────
def main():
    args = parse_args()

    # Load + preprocess dataset
    ds, tok = load_squad()
    train_ds = ds["train"].shuffle(seed=42).select(range(1000))   # small train subset
    eval_ds  = ds["validation"].shuffle(seed=42).select(range(100))  # small eval subset

    tag = f"lr{args.lr}_bs{args.bs}_wd{args.wd}"
    print(f"\n=== Training {tag} ===")

    # Load BLOOM model and data collator
    model = BloomForCausalLM.from_pretrained(MODEL_NAME)
    collator = DataCollatorForSeq2Seq(tok, model, label_pad_token_id=-100)

    # HuggingFace Trainer configuration
    targs = TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=args.bs,
        learning_rate=args.lr,
        weight_decay=args.wd,
        num_train_epochs=5,
        fp16=True,
        bf16=False,
        report_to="none",
        deepspeed=args.deepspeed
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=collator
    )

    # Train and time it
    start = time.time()
    trainer.train()
    eval_results = trainer.evaluate()  # HuggingFace eval
    metrics = evaluate_model(trainer, eval_ds, tok)  # custom eval
    metrics["eval_loss"] = eval_results["eval_loss"]
    runtime = time.time() - start

    # Final log
    summary = {
        "tag": tag,
        "lr": args.lr,
        "bs": args.bs,
        "wd": args.wd,
        "eval_loss": float(metrics.get("eval_loss", float("nan"))),
        "runtime_s": round(runtime, 2),
    }

    print("saved:", summary)

# ───────────────────────────── Entry Point ─────────────────────────────
if __name__ == "__main__":
    main()

