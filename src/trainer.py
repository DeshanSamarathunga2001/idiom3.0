"""
Model training module with LoRA fine-tuning.
Handles model setup, training loop, and checkpointing.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import yaml


def setup_model_and_tokenizer(model_name: str, special_tokens: Optional[List[str]] = None):
    """
    Load NLLB model and tokenizer, add special tokens.
    
    Args:
        model_name: HuggingFace model identifier
        special_tokens: List of special tokens to add (e.g., ['<IDIOM>', '</IDIOM>'])
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens if provided
    if special_tokens:
        tokenizer.add_tokens(special_tokens)
        print(f"✓ Added special tokens: {special_tokens}")
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Resize token embeddings to match tokenizer
    if special_tokens:
        model.resize_token_embeddings(len(tokenizer))
        print(f"✓ Resized token embeddings to {len(tokenizer)}")
    
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
        
        # Set target language for labels
        tokenizer.tgt_lang = tgt_lang
        
        # Tokenize targets
        labels = tokenizer(
            examples['target_si'],
            max_length=max_length,
            truncation=True,
            padding=False
        )
        
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
        Trained model
    """
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=config.get('learning_rate', 3e-4),
        num_train_epochs=config.get('num_epochs', 10),
        per_device_train_batch_size=config.get('batch_size', 4),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
        warmup_steps=config.get('warmup_steps', 100),
        weight_decay=config.get('weight_decay', 0.01),
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        fp16=config.get('use_fp16', False) and torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        seed=config.get('seed', 42)
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config.get('early_stopping_patience', 3)
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping]
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
