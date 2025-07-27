import os, re, string, torch, numpy as np
from datasets import load_dataset

from transformers import (BloomForCausalLM, BloomTokenizerFast,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq)
import ray
from ray import tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
import ray.train.huggingface.transformers as rhf
import sys
from datasets import disable_caching
from ray.tune.search.optuna import OptunaSearch
from ray.train.huggingface.transformers import RayTrainReportCallback
from transformers import TrainerCallback
from ray.train import report

MODEL_NAME = "bigscience/bloomz-560m"  # Base model for fine-tuning


def load_squad():
    """Loads and preprocesses the SQuAD dataset for generative QA."""
    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    dataset = load_dataset("squad")

    def preprocess_function(examples):
        tokenizer.padding_side = "right"
        # Combine question and context into a single prompt for generation
        inputs = ["Question: " + q + " Context: " + c + " Answer:"
                  for q, c in zip(examples["question"], examples["context"])]
        # Use the first available answer or "No Answer" if empty
        answers = [a["text"][0] if len(a["text"]) > 0 else "No Answer" for a in examples["answers"]]

        # Tokenize input and labels (answers)
        model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
        labels = tokenizer(answers, truncation=True, padding="max_length", max_length=512)
        # Replace padding tokens in labels with -100 so they are ignored in loss computation
        labels["input_ids"] = [
            [(token if token != tokenizer.pad_token_id else -100) for token in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset, tokenizer


def normalize_answer(s):
    """Normalize strings for EM/F1 evaluation (lowercase, remove punctuation, articles, extra spaces)."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em_and_f1(predicted, ground_truth):
    """Compute Exact Match (EM) and F1 scores for one prediction vs ground truth."""
    pred = normalize_answer(predicted)
    gt = normalize_answer(ground_truth)

    em = int(pred == gt)
    pred_tokens = pred.split()
    gt_tokens = gt.split()
    common = set(pred_tokens) & set(gt_tokens)
    if len(common) == 0:
        return em, 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return em, f1


def evaluate_model(trainer, dataset, tokenizer):
    """Custom evaluation loop for the validation split, computing EM and F1 scores."""
    em_scores = []
    f1_scores = []

    model = trainer.model
    model.eval()  # Switch to eval mode

    for example in dataset:
        # Prepare input tensors
        input_ids = torch.tensor([example["input_ids"]]).to(trainer.args.device)
        attention_mask = torch.tensor([example["attention_mask"]]).to(trainer.args.device)

        # Generate answer (greedy decoding, max 50 tokens)
        outputs = trainer.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=False
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer after "Answer:" if present
        if "Answer:" in generated_text:
            generated_answer = generated_text.split("Answer:")[-1].strip()
        else:
            generated_answer = generated_text.strip()

        # Convert labels back to string (ignoring -100)
        reference_answer = tokenizer.decode(
            [token for token in example["labels"] if token != -100],
            skip_special_tokens=True
        ).strip()

        # Compute metrics
        em, f1 = compute_em_and_f1(generated_answer, reference_answer)
        em_scores.append(em)
        f1_scores.append(f1)

    avg_em = np.mean(em_scores)
    avg_f1 = np.mean(f1_scores)
    return {"exact_match": avg_em, "f1": avg_f1}


class TuneReportCallback(TrainerCallback):
    """Custom callback: reports Hugging Face Trainer metrics back to Ray Tune."""

    def on_evaluate(
            self, args, state, control, metrics=None, **kwargs):
        if metrics:  # Called after each eval step
            report(metrics)  # Increments training_iteration for ASHA/PBT


def train_loop_per_worker(config):
    """
    Main training function executed by each Ray worker.
    Each worker runs one trial with its own hyperparameter config.
    """
    # 1. Load + subset dataset for faster trials
    ds, tok = load_squad()
    train_ds = ds["train"].shuffle(seed=42).select(range(1000))
    eval_ds = ds["validation"].shuffle(seed=42).select(range(100))

    # 2. Instantiate the BLOOM model
    model = BloomForCausalLM.from_pretrained("bigscience/bloomz-560m")

    # 3. TrainingArguments (hyperparameters passed from Ray Tune via `config`)
    args = TrainingArguments(
        output_dir="checkpoints",
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="no",  # Avoid saving checkpoints per epoch
        learning_rate=config["lr"],  # Sampled by Ray Tune
        per_device_train_batch_size=config["per_device_bs"],
        num_train_epochs=5,
        bf16=False, fp16=True,  # Mixed precision to save GPU memory
        weight_decay=config["wd"],
        report_to="none",
        deepspeed="/ibex/user/x_mohameta/distributed/otta/ray/deepspeed/ds_config.json"
    )

    collator = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100)

    # 4. Create the Trainer, attaching Ray Tune reporting callback
    trainer = Trainer(model=model,
                      args=args,
                      train_dataset=train_ds,
                      eval_dataset=eval_ds,
                      tokenizer=tok,
                      callbacks=[TuneReportCallback()],
                      data_collator=collator, )

    # 5. Prepare Trainer for Ray (wraps DDP/FSDP for multi-worker)
    trainer = rhf.prepare_trainer(trainer)
    trainer.train()

    # 6. Final evaluation + report back best metrics to Ray Tune
    eval_results = trainer.evaluate()
    metrics = evaluate_model(trainer, eval_ds, tok)
    metrics["eval_loss"] = eval_results["eval_loss"]
    report(metrics)  # Final report (important for ASHA and get_best_result)


# --- Ray Tune wrapper (head node logic) ---------------------------------------

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,  # Training logic per worker
    train_loop_config={  # Default config (overridden by Tune)
        "lr": 1e-5,
        "per_device_bs": 1,
    },
    scaling_config=ScalingConfig(
        num_workers=2,  # 2 workers per trial
        use_gpu=True,  # Use GPUs (assigned automatically by Ray)
        resources_per_worker={"CPU": 2, "GPU": 1}  # 2 CPUs + 1 GPU per worker
    ),
    run_config=RunConfig(
        name="bloom_fsdp_tune",
        checkpoint_config=CheckpointConfig(num_to_keep=1)  # Keep only latest checkpoint per trial
    )
)

# --- Ray Tune search & scheduler ---------------------------------------------

search_alg = OptunaSearch(metric="eval_loss", mode="min")  # Optuna for sampling hyperparameters

tuner = tune.Tuner(
    trainer.as_trainable(),
    tune_config=tune.TuneConfig(
        search_alg=search_alg,  # Hyperparameter search strategy
        scheduler=tune.schedulers.ASHAScheduler(  # ASHA for early stopping
            metric="eval_loss",
            mode="min",
            grace_period=1,
            max_t=5,
            reduction_factor=2
        ),
        num_samples=12,  # Total trials to run
        max_concurrent_trials=4,  # Max active trials at once
    ),
    param_space={  # Hyperparameter search space
        "train_loop_config": {
            "lr": tune.loguniform(5e-6, 2e-4),
            "per_device_bs": tune.choice([1, 2]),
            "wd": tune.choice([0.0, 0.01])
        }
    },
)

# --- Main: executed on head node ---------------------------------------------

if __name__ == "__main__":
    print("=== Starting bloom_ray_tune.py ===", file=sys.stderr)
    print(f"RAY ADDR: {os.environ.get('ip_head')}", file=sys.stderr)

    # Connects to the Ray head node (IP set by SLURM head script)
    ray.init(address=os.environ["ip_head"],
             _node_ip_address=os.environ["head_node_ip"],
             _redis_password=os.environ["redis_password"])

    print("Training started...")

    # Launch all trials
    result = tuner.fit()

    # Print best trial’s metrics
    best_result = result.get_best_result(metric="eval_loss",
