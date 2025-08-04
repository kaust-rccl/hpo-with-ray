import os, re, string, torch, numpy as np, argparse, sys
from datasets import load_dataset
from transformers import (BloomForCausalLM, BloomTokenizerFast,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq)
import ray
from ray import tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, report
import ray.train.huggingface.transformers as rhf
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from transformers import TrainerCallback

# --- Argument Parser --------------------------------------------------------
parser = argparse.ArgumentParser(description="Ray Tune BOHB for BLOOM fine-tuning")
parser.add_argument("--num_train_epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--deepspeed", type=str, default="./config/ds_config.json", help="Path to DeepSpeed config JSON")
parser.add_argument("--lr_lower", type=float, default=5e-6, help="Lower bound for learning rate")
parser.add_argument("--lr_upper", type=float, default=2e-4, help="Upper bound for learning rate")
parser.add_argument("--per_device_bs_choices", nargs="+", type=int, default=[1, 2], help="Batch size choices per device")
parser.add_argument("--wd_choices", nargs="+", type=float, default=[0.0, 0.01], help="Weight decay choices")

args_cli = parser.parse_args()

MODEL_NAME = "bigscience/bloomz-560m"  # Base model


def load_squad():
    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    dataset = load_dataset("squad")
    def preprocess_function(examples):
        tokenizer.padding_side = "right"
        inputs = ["Question: " + q + " Context: " + c + " Answer:"
                  for q, c in zip(examples["question"], examples["context"])]
        answers = [a["text"][0] if len(a["text"]) > 0 else "No Answer" for a in examples["answers"]]
        model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
        labels = tokenizer(answers, truncation=True, padding="max_length", max_length=512)
        labels["input_ids"] = [
            [(token if token != tokenizer.pad_token_id else -100) for token in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return dataset.map(preprocess_function, batched=True), tokenizer


class TuneReportCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            report(metrics)


def train_loop_per_worker(config):
    ds, tok = load_squad()
    train_ds = ds["train"].shuffle(seed=42).select(range(1000))
    eval_ds = ds["validation"].shuffle(seed=42).select(range(100))
    model = BloomForCausalLM.from_pretrained(MODEL_NAME)
    training_args = TrainingArguments(
        output_dir="checkpoints",
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=config["lr"],
        per_device_train_batch_size=config["per_device_bs"],
        num_train_epochs=args_cli.num_train_epochs,
        fp16=True,
        weight_decay=config["wd"],
        report_to="none",
        deepspeed=args_cli.deepspeed
    )
    collator = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_ds, eval_dataset=eval_ds,
                      tokenizer=tok, callbacks=[TuneReportCallback()],
                      data_collator=collator)
    trainer = rhf.prepare_trainer(trainer)
    trainer.train()
    eval_results = trainer.evaluate()
    metrics = {"eval_loss": eval_results["eval_loss"]}
    report(metrics)


trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={"lr": 1e-5, "per_device_bs": 1, "wd": 0.0},
    scaling_config=ScalingConfig(num_workers=2, use_gpu=True,
                                 resources_per_worker={"CPU": 2, "GPU": 1}),
    run_config=RunConfig(name="bloom_fsdp_tune",
                         checkpoint_config=CheckpointConfig(num_to_keep=1))
)

search_alg = TuneBOHB(metric="eval_loss", mode="min")
search_alg = tune.search.ConcurrencyLimiter(search_alg, max_concurrent=8)
bohb_scheduler = HyperBandForBOHB(time_attr="training_iteration",
                                  max_t=args_cli.num_train_epochs,
                                  reduction_factor=4,
                                  metric="eval_loss", mode="min")

tuner = tune.Tuner(
    trainer.as_trainable(),
    tune_config=tune.TuneConfig(search_alg=search_alg, scheduler=bohb_scheduler, num_samples=12),
    param_space={
        "train_loop_config": {
            "lr": tune.loguniform(args_cli.lr_lower, args_cli.lr_upper),
            "per_device_bs": tune.choice(args_cli.per_device_bs_choices),
            "wd": tune.choice(args_cli.wd_choices),
        }
    },
)

if __name__ == "__main__":
    print("=== Starting raytune_bayesian.py ===", file=sys.stderr)
    ray.init(address=os.environ["ip_head"],
             _node_ip_address=os.environ["head_node_ip"],
             _redis_password=os.environ["redis_password"])
    result = tuner.fit()
    best_result = result.get_best_result(metric="eval_loss", mode="min")
    print("\nBest Trial Result:")
    print(best_result.metrics)
