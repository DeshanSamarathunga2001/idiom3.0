"""
test_model.py - Test if model loads and works locally
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import json

print("="*80)
print("LOADING MODEL LOCALLY")
print("="*80)

# Paths
base_model_name = "facebook/nllb-200-distilled-600M"
lora_path = "./models/lora_adapters"
idiom_db_path = "./data/idiom_database.json"

# Load base model
print("\n1. Loading base model (this will download ~2.5GB first time)...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

print("✅ Base model loaded")

# Load LoRA adapters
print("\n2. Loading LoRA adapters (your trained weights)...")
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

print("✅ LoRA adapters loaded")

# Rebuild lang_code_to_id (if needed)
print("\n3. Setting up language codes...")
tokenizer.src_lang = "eng_Latn"

# Get language codes
import re
vocab = tokenizer.get_vocab()
lang_pattern = re.compile(r'^[a-z]{3}_[A-Z][a-z]{3}$')
lang_code_to_id = {token: token_id for token, token_id in vocab.items() if lang_pattern.match(token)}
tokenizer.lang_code_to_id = lang_code_to_id

print(f"✅ Found {len(lang_code_to_id)} language codes")

# Load idiom database
print("\n4. Loading idiom database...")
with open(idiom_db_path, 'r', encoding='utf-8') as f:
    idioms = json.load(f)

print(f"✅ Loaded {len(idioms)} idioms")

# Test translation
print("\n" + "="*80)
print("TESTING TRANSLATIONS")
print("="*80)

test_sentences = [
    "He kicked the bucket yesterday.",
    "She has a green thumb for gardening.",
    "It's raining cats and dogs outside.",
    "I love my family.",
    "The cat is sleeping."
]

def translate(text):
    """Translate English to Sinhala"""
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            forced_bos_token_id=tokenizer.lang_code_to_id["sin_Sinh"]
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

for i, text in enumerate(test_sentences, 1):
    print(f"\n{i}. EN: {text}")
    translation = translate(text)
    print(f"   SI: {translation}")

print("\n" + "="*80)
print("✅ MODEL WORKS LOCALLY!")
print("="*80)