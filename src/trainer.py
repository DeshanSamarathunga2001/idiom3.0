"""
Model training module with LoRA fine-tuning.
Handles model setup, training loop, and checkpointing.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from transformers import (
    NllbTokenizer,
    AutoModelForSeq2SeqLM,  # ✅ CHANGED FROM M2M100
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import yaml


def setup_model_and_tokenizer(
    model_name: str = "facebook/nllb-200-distilled-600M",
    special_tokens: List[str] = None
):
    """
    Setup the NLLB model and tokenizer.
    
    Args:
        model_name: Pretrained model name
        special_tokens: NOT USED - kept for compatibility
        
    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = NllbTokenizer.from_pretrained(model_name)
    print(f"✓ Loaded {tokenizer.__class__.__name__}")
    print(f"  Tokenizer type: {tokenizer.__class__.__name__}")
    
    # ✅ FIXED: Use AutoModelForSeq2SeqLM instead of M2M100
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    print(f"✓ Vocabulary size: {len(tokenizer)} (unchanged)")
    
    # Manually add lang_code_to_id if missing
    if not hasattr(tokenizer, 'lang_code_to_id'):
        import re
        vocab = tokenizer.get_vocab()
        lang_code_pattern = re.compile(r'^[a-z]{3}_[A-Z][a-z]{3}$')
        
        tokenizer.lang_code_to_id = {}
        for token, token_id in vocab.items():
            if lang_code_pattern.match(token):
                tokenizer.lang_code_to_id[token] = token_id
        
        tokenizer.id_to_lang_code = {v: k for k, v in tokenizer.lang_code_to_id.items()}
        print(f"✓ Manually added {len(tokenizer.lang_code_to_id)} language codes")
    else:
        print(f"✓ lang_code_to_id already present ({len(tokenizer.lang_code_to_id)} languages)")
    
    # Set source language
    tokenizer.src_lang = "eng_Latn"
    print(f"✓ Source language: eng_Latn")
    print(f"✓ Model loaded successfully")
    
    return model, tokenizer


def apply_lora(model, config: Dict):
    """
    Apply LoRA adapters to the model.

    Args:
        model: Base model
        config: LoRA configuration dictionary

    Returns:
        PEFT model with LoRA adapters
    """
    lora_config = LoraConfig(
        r=config.get('r', 16),
        lora_alpha=config.get('lora_alpha', 32),
        lora_dropout=config.get('lora_dropout', 0.05),
        target_modules=config.get('target_modules', ['q_proj', 'v_proj']),
        task_type=TaskType.SEQ_2_SEQ_LM,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("✓ LoRA adapters applied")
    return model


def prepare_dataset(data_path: str, tokenizer, src_lang: str, tgt_lang: str, max_length: int = 128):
    """
    Prepare dataset for training.
    """
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data)} examples from {data_path}")

    def preprocess_function(examples):
        # Set source language
        tokenizer.src_lang = src_lang
        
        # Tokenize inputs
        model_inputs = tokenizer(
            examples['source_en'],
            max_length=max_length,
            truncation=True,
            padding=False
        )

        # Tokenize targets - for NLLB, we need to set forced_bos_token_id manually
        # Get target language token ID
        tgt_lang_id = tokenizer.lang_code_to_id.get(tgt_lang)
        
        # Tokenize target text
        labels = tokenizer(
            examples['target_si'],
            max_length=max_length,
            truncation=True,
            padding=False
        )

        # Add the target language ID at the beginning if needed
        if tgt_lang_id is not None:
            # Prepend target language token to labels
            processed_labels = []
            for label_ids in labels['input_ids']:
                # NLLB expects: [tgt_lang_id, ...tokens..., eos_token_id]
                if label_ids[0] != tgt_lang_id:
                    processed_labels.append([tgt_lang_id] + label_ids)
                else:
                    processed_labels.append(label_ids)
            model_inputs['labels'] = processed_labels
        else:
            model_inputs['labels'] = labels['input_ids']

        return model_inputs

    # Convert to dataset
    dataset = Dataset.from_dict({
        'source_en': [ex['source_en'] for ex in data],
        'target_si': [ex['target_si'] for ex in data]
    })

    # Tokenize
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset

def train_model(model, tokenizer, train_dataset, config: Dict, output_dir: str):
    """Train the model."""
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=float(config.get('learning_rate', 3e-4)),
        num_train_epochs=int(config.get('num_epochs', 10)),
        per_device_train_batch_size=int(config.get('batch_size', 4)),
        gradient_accumulation_steps=int(config.get('gradient_accumulation_steps', 4)),
        warmup_steps=int(config.get('warmup_steps', 100)),
        weight_decay=float(config.get('weight_decay', 0.01)),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        fp16=config.get('use_fp16', False) and torch.cuda.is_available(),
        report_to="none",
        seed=int(config.get('seed', 42)),
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    print("Starting training...")
    trainer.train()
    print("✓ Training completed")

    return model, trainer


def save_checkpoint(model, tokenizer, path: str) -> None:
    """Save model checkpoint."""
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"✓ Checkpoint saved to {path}")