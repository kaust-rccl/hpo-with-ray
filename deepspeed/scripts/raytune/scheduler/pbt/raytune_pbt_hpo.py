import os, re, string, torch, numpy as np

import argparse
import sys

# --- Argument Parser --------------------------------------------------------
parser = argparse.ArgumentParser(description="Ray Tune ASHA for BLOOM fine-tuning")
parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--deepspeed", type=str, default="./config/ds_config.json", help="Path to DeepSpeed config JSON")
parser.add_argument("--lr_lower", type=float, default=5e-6, help="Lower bound for learning rate")
parser.add_argument("--lr_upper", type=float, default=2e-4, help="Upper bound for learning rate")
parser.add_argument("--per_device_bs_choices", nargs="+", type=int, default=[1, 2], help="Batch size choices per device")
parser.add_argument("--wd_choices", nargs="+", type=float, default=[0.0, 0.01], help="Weight decay choices")

args_cli = parser.parse_args()

# --- Dataset & Transformers Imports -----------------------------------------
from datasets import load_dataset, disable_caching
from transformers import (BloomForCausalLM, BloomTokenizerFast,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainerCallback)

# --- Ray & Tune Imports -----------------------------------------------------
import ray
from ray import tune
from ray.train import get_context, report, Checkpoint, session
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
import ray.train.huggingface.transformers as rhf
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import PopulationBasedTraining

# --- Experiment Constants ---------------------------------------------------
user = os.getenv("USER")
MODEL_NAME = "bigscience/bloomz-560m"
EXPERIMENT_NAME = "bloom_fsdp_tune"
STORAGE_PATH = f"/ibex/user/{user}"
OUTPUT_DIR_BASE = os.path.join(STORAGE_PATH, EXPERIMENT_NAME)
HF_CHKPT = os.path.join(OUTPUT_DIR_BASE, "hf")
RAY_CHKPT = os.path.join(OUTPUT_DIR_BASE, "ray")

# --- Data Preprocessing -----------------------------------------------------
def load_squad():
    """Loads and preprocesses the SQuAD dataset for generative QA."""
    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    dataset = load_dataset("squad")

    def preprocess_function(examples):
        tokenizer.padding_side = "right"
        inputs = [f"Question: {q} Context: {c} Answer:" for q, c in zip(examples["question"], examples["context"])]
        answers = [a["text"][0] if a["text"] else "No Answer" for a in examples["answers"]]

        model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
        labels = tokenizer(answers, truncation=True, padding="max_length", max_length=512)
        labels["input_ids"] = [[token if token != tokenizer.pad_token_id else -100 for token in label]
                                for label in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess_function, batched=True), tokenizer

# --- Evaluation Helpers -----------------------------------------------------
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_em_and_f1(predicted, ground_truth):
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
    f1 = 2 * precision * recall / (precision + recall)
    return em, f1

def evaluate_model(trainer, dataset, tokenizer):
    em_scores, f1_scores = [], []
    model = trainer.model
    model.eval()

    for ex in dataset:
        input_ids = torch.tensor([ex["input_ids"]]).to(trainer.args.device)
        attn_mask = torch.tensor([ex["attention_mask"]]).to(trainer.args.device)

        outputs = model.generate(input_ids, attention_mask=attn_mask, max_new_tokens=50, do_sample=False)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = generated.split("Answer:")[-1].strip() if "Answer:" in generated else generated.strip()

        reference = tokenizer.decode([t for t in ex["labels"] if t != -100], skip_special_tokens=True).strip()
        em, f1 = compute_em_and_f1(generated_answer, reference)
        em_scores.append(em)
        f1_scores.append(f1)

    return {"exact_match": np.mean(em_scores), "f1": np.mean(f1_scores)}

# --- Ray Tune Callback ------------------------------------------------------
class TuneReportCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and state.global_step:
            ckpt = Checkpoint.from_directory(args.output_dir)
            report(metrics, checkpoint=ckpt)

# --- Training Function ------------------------------------------------------
def train_loop_per_worker(config):
    ds, tok = load_squad()
    train_ds = ds["train"].shuffle(seed=42).select(range(1000))
    eval_ds = ds["validation"].shuffle(seed=42).select(range(100))

    model = BloomForCausalLM.from_pretrained(MODEL_NAME)
    trial_dir = get_context().get_trial_name()
    hf_output_dir = os.path.join(HF_CHKPT, trial_dir)

    args = TrainingArguments(
        output_dir=hf_output_dir,
        learning_rate=config["lr"],
        per_device_train_batch_size=config["per_device_bs"],
        num_train_epochs=args_cli.num_train_epochs,
        bf16=False, fp16=True,
        weight_decay=config["wd"],
        report_to="none",
        save_strategy="steps",
        save_total_limit=1,
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        save_only_model=True,
        deepspeed=args_cli.deepspeed
    )
    collator = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        callbacks=[TuneReportCallback()],
        data_collator=collator
    )

    trainer = rhf.prepare_trainer(trainer)
    trainer.train()

    eval_results = trainer.evaluate()
    metrics = evaluate_model(trainer, eval_ds, tok)
    metrics["eval_loss"] = eval_results["eval_loss"]
    ckpt = Checkpoint.from_directory(args.output_dir)
    report(metrics, checkpoint=ckpt)

# --- TorchTrainer Setup -----------------------------------------------------
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={"lr": 1e-5, "per_device_bs": 1},
    scaling_config=ScalingConfig(
        num_workers=2,
        use_gpu=True,
        resources_per_worker={"CPU": 2, "GPU": 1}
    )
)

# --- Population Based Tuning Strategy ---------------------------------------
pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="eval_loss",
    mode="min",
    perturbation_interval=2,
    hyperparam_mutations={
        "train_loop_config": {
            "lr": tune.loguniform(args_cli.lr_lower, args_cli.lr_upper),
            "per_device_bs": tune.choice(args_cli.per_device_bs_choices),
            "wd": tune.choice(args_cli.wd_choices)
        }
    },
)

# --- Ray Tune Tuner ---------------------------------------------------------
tuner = tune.Tuner(
    trainer.as_trainable(),
    run_config=RunConfig(
        name="bloom_fsdp_tune",
        storage_path=RAY_CHKPT,
        checkpoint_config=CheckpointConfig(num_to_keep=5)
    ),
    tune_config=tune.TuneConfig(
        scheduler=pbt,
        num_samples=12,
        max_concurrent_trials=4
    ),
    param_space={
        "train_loop_config": {
            "lr": tune.loguniform(5e-6, 2e-4),
            "per_device_bs": tune.choice([1, 2]),
            "wd": tune.choice([0.0, 0.01])
        }
    },
)

# --- Entry Point ------------------------------------------------------------
if __name__ == "__main__":
    print("=== Starting bloom_ray_tune.py ===", file=sys.stderr)
    print(f"RAY ADDR: {os.environ.get('ip_head')}", file=sys.stderr)

    ray.init(address=os.environ["ip_head"],
             _node_ip_address=os.environ["head_node_ip"],
             _redis_password=os.environ["redis_password"])

    print("Training started...")
    result = tuner.fit()
    best_result = result.get_best_result(metric="eval_loss", mode="min")
    print("\nBest Trial Result:")
    print(best_result.metrics)
