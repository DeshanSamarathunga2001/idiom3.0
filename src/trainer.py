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
    M2M100ForConditionalGeneration,
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
    
    # Load model (DON'T add special tokens or resize)
    model = M2M100ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )
    
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

    Args:
        data_path: Path to JSON data file
        tokenizer: Tokenizer instance
        src_lang: Source language code
        tgt_lang: Target language code
        max_length: Maximum sequence length

    Returns:
        HuggingFace Dataset
    """
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data)} examples from {data_path}")

    # Prepare dataset
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

        # Tokenize targets with target language
        original_lang = tokenizer.src_lang
        tokenizer.src_lang = tgt_lang
        
        labels = tokenizer(
            examples['target_si'],
            max_length=max_length,
            truncation=True,
            padding=False
        )
        
        # Restore original language
        tokenizer.src_lang = original_lang

        model_inputs['labels'] = labels['input_ids']
        
        return model_inputs

    # Convert to HuggingFace dataset
    dataset = Dataset.from_dict({
        'source_en': [ex['source_en'] for ex in data],
        'target_si': [ex['target_si'] for ex in data]
    })

    # Apply preprocessing
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset


def train_model(
    model,
    tokenizer,
    train_dataset,
    config: Dict,
    output_dir: str
):
    """
    Train the model with the given configuration.

    Args:
        model: Model to train
        tokenizer: Tokenizer instance
        train_dataset: Training dataset
        config: Training configuration
        output_dir: Directory to save checkpoints

    Returns:
        Trained model and trainer
    """
    # Get target language token ID
    tgt_lang = config.get('target_lang', 'sin_Sinh')
    forced_bos_token_id = None
    
    if hasattr(tokenizer, 'lang_code_to_id') and tgt_lang in tokenizer.lang_code_to_id:
        forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
        print(f"✓ Using forced_bos_token_id: {forced_bos_token_id} for {tgt_lang}")
    else:
        print(f"⚠️  Warning: Could not get forced_bos_token_id for {tgt_lang}")
    
    # Training arguments
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
        predict_with_generate=False,
        generation_config=None
    )

    # Set the forced_bos_token_id on the model
    if forced_bos_token_id is not None:
        if hasattr(model, 'generation_config'):
            model.generation_config.forced_bos_token_id = forced_bos_token_id
            print(f"✓ Set model.generation_config.forced_bos_token_id = {forced_bos_token_id}")
        if hasattr(model, 'config'):
            model.config.forced_bos_token_id = forced_bos_token_id
            print(f"✓ Set model.config.forced_bos_token_id = {forced_bos_token_id}")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id
    )

    # Trainer
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
    """
    Save model checkpoint.

    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        path: Output path
    """
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"✓ Checkpoint saved to {path}")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config