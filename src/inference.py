"""
Inference module for translation.
Handles model loading and translation generation.
"""

import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


def load_trained_model(checkpoint_path: str, base_model: str = None):
    """
    Load fine-tuned model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        base_model: Base model name (required if loading LoRA adapters)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Try to load as full model first
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
        print("✓ Loaded full model")
    except (OSError, ValueError) as e:
        # If that fails, try loading as PEFT model
        if base_model is None:
            raise ValueError("base_model must be provided when loading PEFT adapters")
        
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(base, checkpoint_path)
        print("✓ Loaded PEFT model with adapters")
    
    # Set to evaluation mode
    model.eval()
    
    return model, tokenizer


def translate(
    text: str,
    model,
    tokenizer,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "sin_Sinh",
    max_length: int = 128,
    num_beams: int = 5
) -> str:
    """
    Translate a single text.
    
    Args:
        text: Source text to translate
        model: Translation model
        tokenizer: Tokenizer
        src_lang: Source language code
        tgt_lang: Target language code
        max_length: Maximum length of generated translation
        num_beams: Number of beams for beam search
        
    Returns:
        Translated text
    """
    # Set source language
    tokenizer.src_lang = src_lang
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get target language token
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    # Generate translation
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    # Decode
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    return translation


def batch_translate(
    texts: List[str],
    model,
    tokenizer,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "sin_Sinh",
    max_length: int = 128,
    batch_size: int = 8,
    num_beams: int = 5
) -> List[str]:
    """
    Translate multiple texts in batches.
    
    Args:
        texts: List of source texts
        model: Translation model
        tokenizer: Tokenizer
        src_lang: Source language code
        tgt_lang: Target language code
        max_length: Maximum length of generated translations
        batch_size: Batch size for processing
        num_beams: Number of beams for beam search
        
    Returns:
        List of translations
    """
    translations = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Set source language
        tokenizer.src_lang = src_lang
        
        # Tokenize batch
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get target language token
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        
        # Generate translations
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode
        batch_translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translations.extend(batch_translations)
    
    return translations


def translate_with_idiom(text: str, model, tokenizer, src_lang: str = "eng_Latn", tgt_lang: str = "sin_Sinh") -> str:
    """
    Translate text that contains idiom tags.
    This is a wrapper around the translate function with appropriate settings.
    
    Args:
        text: Source text with <IDIOM>...</IDIOM> tags
        model: Translation model
        tokenizer: Tokenizer
        src_lang: Source language code
        tgt_lang: Target language code
        
    Returns:
        Translated text
    """
    return translate(text, model, tokenizer, src_lang, tgt_lang)
