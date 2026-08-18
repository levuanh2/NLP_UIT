"""QLoRA fine-tune of the 3B answerer on the BTC train answers.

The competition caps the model at 4B parameters, so the base stays
Vi-Qwen2-3B-RAG and only a LoRA adapter is trained, over a 4-bit NF4 copy of
the weights to leave room on a 16GB card.

Loss is computed on the answer alone; the prompt tokens are masked out, because
the model is being taught to write like the expert references, not to reproduce
its own instructions.

Usage:
  python scripts/train_qlora.py [--dataset ...] [--epochs 2] [--max-length 6144]
"""

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
IGNORE_INDEX = -100


class AnswerOnlyDataset(torch.utils.data.Dataset):
    """Grounded prompt in, expert answer out, with the prompt masked from the loss."""

    def __init__(self, records: list[dict], tokenizer, max_length: int) -> None:
        self.examples: list[dict] = []
        self.skipped = 0
        for record in records:
            prompt = tokenizer.apply_chat_template(
                record["messages"], tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(
                record["answer"] + tokenizer.eos_token, add_special_tokens=False
            )["input_ids"]
            if len(prompt_ids) + len(answer_ids) > max_length:
                self.skipped += 1
                continue
            self.examples.append(
                {
                    "input_ids": prompt_ids + answer_ids,
                    "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def collate(batch: list[dict], pad_token_id: int) -> dict:
    width = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention = [], [], []
    for item in batch:
        padding = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        labels.append(item["labels"] + [IGNORE_INDEX] * padding)
        attention.append([1] * len(item["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="AITeamVN/Vi-Qwen2-3B-RAG")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/train/sft.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "models/qlora-answerer")
    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Upper bound; early stopping usually cuts first. At 57s a step an "
        "epoch is 2.2 hours, so the bound is what keeps a stalled run from "
        "burning days.",
    )
    parser.add_argument("--max-length", type=int, default=5120)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    # Sequences run 2200-5120 tokens, so a batch of 2 pads the shorter one up
    # to the longer and spends about a third of the step on padding: 72s
    # against 56s for one sequence at a time.
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Share of the dataset held back for validation loss. A fifth cost "
        "8.2 minutes an evaluation and 2.2 hours across a run; a twentieth of "
        "6800 examples is still 340, and loss varies far less across examples "
        "than METEOR does. The metric that decides whether the adapter is worth "
        "keeping is METEOR over the generated dev answers, which costs a "
        "generation each and stays on the 200-question dev set.",
    )
    parser.add_argument(
        "--precision",
        choices=("bf16", "nf4"),
        default="nf4",
        help="Base weight format. Measured on this box they run at the same "
        "speed — 56s a step against 57s — because the card saturates either "
        "way, so nf4 wins on the 6GB it does not spend.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Hard stop, roughly one pass over 6800 examples at an effective "
        "batch of 16. The epoch bound alone would allow 1181 steps and 18 "
        "hours, and early stopping cannot be relied on to cut that when the "
        "validation loss is still falling.",
    )
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Evaluations without an improvement in validation loss before "
        "stopping.",
    )
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"dataset: {len(records)} examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = AnswerOnlyDataset(records, tokenizer, args.max_length)
    print(f"usable: {len(dataset)}  skipped as too long: {dataset.skipped}")
    if not len(dataset):
        print("nothing to train on")
        return 1

    validation = None
    held_back = int(len(dataset) * args.val_fraction)
    if held_back:
        validation = torch.utils.data.Subset(
            dataset, range(len(dataset) - held_back, len(dataset))
        )
        dataset = torch.utils.data.Subset(dataset, range(len(dataset) - held_back))
    print(f"train: {len(dataset)}  validation: {len(validation or [])}")

    # Cross-entropy over a 151936-token vocabulary materialises a logit tensor
    # of seq x vocab, then an fp32 copy for the loss and another for its
    # gradient — several GB that pushed the allocator into OOM-retry loops and
    # made a step take 52s. Liger fuses the projection into the loss and
    # computes it in slices, so the full matrix never exists.
    loader = AutoModelForCausalLM
    try:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        loader = AutoLigerKernelForCausalLM
        print("using liger fused cross-entropy")
    except Exception as exc:  # noqa: BLE001 - training still works without it
        print(f"liger unavailable ({exc}); falling back to standard loss")

    quantization = None
    if args.precision == "nf4":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    print(f"base precision: {args.precision}")
    model = loader.from_pretrained(
        args.model,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    if args.precision == "nf4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.accumulation,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            # Cosine over a 100-epoch bound would barely decay before early
            # stopping fires, so the schedule is flat with a short warmup.
            lr_scheduler_type="constant_with_warmup",
            # transformers 5 dropped warmup_ratio; with a 100-epoch bound a
            # ratio was meaningless anyway, so warmup is a flat step count.
            warmup_steps=20,
            logging_steps=10,
            eval_strategy="steps" if validation else "no",
            eval_steps=args.eval_steps,
            per_device_eval_batch_size=args.batch_size,
            # load_best_model_at_end needs saves on the same cadence as evals.
            save_strategy="steps" if validation else "epoch",
            save_steps=args.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=bool(validation),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            report_to=[],
        ),
        train_dataset=dataset,
        eval_dataset=validation,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
        callbacks=(
            [EarlyStoppingCallback(early_stopping_patience=args.patience)]
            if validation
            else []
        ),
    )
    trainer.train()
    model.save_pretrained(str(args.output / "final"))
    tokenizer.save_pretrained(str(args.output / "final"))
    print(f"adapter saved to {args.output / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
