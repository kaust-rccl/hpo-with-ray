"""
-------------------------------------------------
Fine‑tune **BLOOM‑560M** on the SQuAD v1.1 dataset with **Ray-Tune** +
Optuna hyper‑parameter search and ASHA early‑stopping.

Key features
============
* Hugging-Face **Trainer** API – supports fp16 + DeepSpeed.
* **TorchTrainer** – handles multi‑GPU workers.
* **OptunaSearch** – search over LR, BS, WD.
* **ASHA** scheduler – early‑terminate poor trials, fits cluster budget.
"""

# ───────────────────────────── Imports ──────────────────────────────
# Std‑lib & scientific
import os, re, string, sys, time, json
from typing import Dict, Any, Tuple

import numpy as np
import torch
from datasets import load_dataset, disable_caching  # HF Datasets
from transformers import (
    BloomForCausalLM,
    BloomTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# Ray-AI Runtime
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, report
import ray.train.huggingface.transformers as rhf

# ─────────────────────────── Configuration ──────────────────────────
MODEL_NAME = "bigscience/bloomz-560m"  # baseline checkpoint to fine‑tune
MAX_LEN   = 512                       # token length for both prompt & answer

# ───────────────────── Dataset Loading & Preprocessing ──────────────

def load_squad() -> Tuple[Dict[str, Any], BloomTokenizerFast]:
    """
    Load **SQuADv1.1** and tokenize for generative QA.

    Returns
    -------
    tokenized_dataset : Dict[str, HF Dataset]
        Contains "train" and "validation" splits with token IDs & labels.
    tokenizer : BloomTokenizerFast
        Tokenizer used for both training and decoding.
    """
    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    dataset   = load_dataset("squad")  # ~85K train / 10K dev rows

    def _preprocess(examples):
        """Convert (question, context, answer) → model input/labels."""
        tokenizer.padding_side = "right"  # allow left‑padding for causal LM

        # Prompt format expected by BLOOM for QA generation
        inputs = [f"Question: {q} Context: {c} Answer:" for q, c in zip(examples["question"], examples["context"])]

        # Use first ground‑truth answer; if none, mark as "No Answer"
        answers = [a["text"][0] if a["text"] else "No Answer" for a in examples["answers"]]

        model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=MAX_LEN)
        labels       = tokenizer(answers, padding="max_length", truncation=True, max_length=MAX_LEN)

        # Replace PAD tokens in labels with ‑100 so HF Trainer ignores them in loss
        labels["input_ids"] = [
            [tok if tok != tokenizer.pad_token_id else -100 for tok in seq]
            for seq in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(_preprocess, batched=True, remove_columns=dataset["train"].column_names)
    return tokenized_dataset, tokenizer

# ───────────────────── String‑normalisation Helpers ──────────────────

def normalize_answer(s: str) -> str:
    """Lowercase + strip punctuation, articles, and extra whitespace."""
    remove_articles  = lambda txt: re.sub(r"\b(a|an|the)\b", " ", txt)
    white_space_fix  = lambda txt: " ".join(txt.split())
    remove_punc      = lambda txt: "".join(ch for ch in txt if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_em_and_f1(predicted: str, ground_truth: str) -> Tuple[int, float]:
    """Exact‑Match (EM) and F1 for a single QA pair."""
    pred, gt = normalize_answer(predicted), normalize_answer(ground_truth)
    em  = int(pred == gt)
    p_toks, g_toks = pred.split(), gt.split()
    common         = set(p_toks) & set(g_toks)
    if not common:
        return em, 0.0
    precision = len(common) / len(p_toks)
    recall    = len(common) / len(g_toks)
    f1        = 2 * precision * recall / (precision + recall)
    return em, f1

# ───────────────────────── Evaluation Utility ────────────────────────

def evaluate_model(trainer: Trainer, dataset, tokenizer):
    """Run generation on *dataset* and compute average EM / F1."""
    em_scores, f1_scores = [], []
    for ex in dataset:
        input_ids      = torch.tensor([ex["input_ids"]]).to(trainer.args.device)
        attn_mask      = torch.tensor([ex["attention_mask"]]).to(trainer.args.device)
        outputs        = trainer.model.generate(input_ids, attention_mask=attn_mask, max_new_tokens=50)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract answer substring
        gen_ans = generated_text.split("Answer:")[-1].strip() if "Answer:" in generated_text else generated_text.strip()
        ref_ans = tokenizer.decode([tok for tok in ex["labels"] if tok != -100], skip_special_tokens=True).strip()
        em, f1  = compute_em_and_f1(gen_ans, ref_ans)
        em_scores.append(em); f1_scores.append(f1)
    return {"exact_match": np.mean(em_scores), "f1": np.mean(f1_scores)}

# ───────────────────────── Train loop (per worker) ────────────────────

def train_loop_per_worker(config: Dict[str, Any]):
    """Executed **inside each Ray worker**. Builds HF Trainer and trains."""
    # Load a *subset* of SQuAD to keep trials quick
    ds, tok = load_squad()
    train_ds = ds["train"].shuffle(seed=42).select(range(1_000))
    eval_ds  = ds["validation"].shuffle(seed=42).select(range(100))

    # Instantiate model and data collator
    model     = BloomForCausalLM.from_pretrained(MODEL_NAME)
    collator  = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100)

    # Build TrainingArguments from hyper‑params provided by Tune
    args = TrainingArguments(
        output_dir="checkpoints",
        learning_rate=float(config["lr"]),
        per_device_train_batch_size=int(config["per_device_bs"]),
        num_train_epochs=5,
        fp16=True,
        weight_decay=float(config["wd"]),
        report_to="none",
        deepspeed="/ibex/user/x_mohameta/distributed/otta/ray/deepspeed/ds_config.json",
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      tokenizer=tok, data_collator=collator)

    # Wrap Trainer for Ray Train (handles DDP, FSDP, etc.)
    trainer = rhf.prepare_trainer(trainer)
    trainer.train()

    # Evaluate and report metrics back to Tune
    eval_results          = trainer.evaluate()
    custom_metrics        = evaluate_model(trainer, eval_ds, tok)
    custom_metrics.update({"eval_loss": eval_results["eval_loss"]})
    report(custom_metrics)  # ray.train.report

# ────────────────────── Ray-Train + Tune Configuration ───────────────

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={"lr": 1e-5, "per_device_bs": 1, "wd": 0.0},
    scaling_config=ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={"CPU": 2, "GPU": 1}),
    run_config=RunConfig(name="bloom_fsdp_tune", checkpoint_config=CheckpointConfig(num_to_keep=1)),
)

search_alg = OptunaSearch(metric="eval_loss", mode="min")  # Bayesian/TPE

# Tune will mutate *train_loop_config* automatically below

param_space = {
    "train_loop_config": {
        "lr": tune.loguniform(5e-6, 2e-4),
        "per_device_bs": tune.choice([1, 2]),
        "wd": tune.choice([0.0, 0.01]),
    }
}

tuner = tune.Tuner(
    trainer.as_trainable(),
    tune_config=tune.TuneConfig(
        search_alg=search_alg,
        scheduler=tune.schedulers.ASHAScheduler(metric="eval_loss", mode="min"),
        num_samples=12,            # total trials
        max_concurrent_trials=4,   # placement‑group limit
    ),
    param_space=param_space,
)

# ─────────────────────────── Driver entry ────────────────────────────
if __name__ == "__main__":
    # Ray cluster addresses provided via SLURM env‑vars in launch experiments
    print("=== Starting bloom_ray_tune.py ===", file=sys.stderr)
    print(f"RAY ADDR: {os.environ.get('ip_head')}", file=sys.stderr)

    ray.init(address=os.environ["ip_head"],
             _node_ip_address=os.environ["head_node_ip"],
             _redis_password=os.environ["redis_password"])

    print("Training started…")
    result       = tuner.fit()
    best_result  = result.get_best_result(metric="eval_loss", mode="min")
    print("\nBest Trial Result:")
    print(json.dumps(best_result.metrics, indent=2))
