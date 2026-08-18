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
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=16)
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
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
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            report_to=[],
        ),
        train_dataset=dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(str(args.output / "final"))
    tokenizer.save_pretrained(str(args.output / "final"))
    print(f"adapter saved to {args.output / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
