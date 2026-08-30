from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optional LoRA supervised fine-tuning demo")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="outputs/lora")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            'Install optional dependencies with: python -m pip install -e ".[finetune]"'
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    dataset = load_dataset("json", data_files=args.dataset, split="train")

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        return encoded

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
