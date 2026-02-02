"""
api.py - Simple Flask API for translations
Run: python api.py
Test: http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import json
import re

app = Flask(__name__)
CORS(app)  # Allow frontend to call API

# Load model once at startup
print("Loading model...")
base_model_name = "facebook/nllb-200-distilled-600M"
lora_path = "./models/lora_adapters"
idiom_db_path = "./data/idiom_database.json"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

# Setup language codes
tokenizer.src_lang = "eng_Latn"
vocab = tokenizer.get_vocab()
lang_pattern = re.compile(r'^[a-z]{3}_[A-Z][a-z]{3}$')
lang_code_to_id = {token: token_id for token, token_id in vocab.items() if lang_pattern.match(token)}
tokenizer.lang_code_to_id = lang_code_to_id

# Load idiom database
with open(idiom_db_path, 'r', encoding='utf-8') as f:
    idioms = json.load(f)

print(f"✅ Model loaded with {len(idioms)} idioms")

# Translation function
def translate_text(text):
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

# Detect idiom in text
def detect_idiom(text):
    text_lower = text.lower()
    for idiom in idioms:
        if idiom['idiom_en'].lower() in text_lower:
            return idiom
    return None

# Routes
@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "model": "NLLB-200 + LoRA (Idiom Translation)",
        "idioms": len(idioms),
        "endpoints": {
            "translate": "/api/translate",
            "idioms": "/api/idioms",
            "search": "/api/search?q=kick"
        }
    })

@app.route('/api/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Detect idiom
    detected_idiom = detect_idiom(text)
    
    # Translate
    translation = translate_text(text)
    
    return jsonify({
        "input": text,
        "output": translation,
        "idiom_detected": detected_idiom is not None,
        "idiom_info": detected_idiom if detected_idiom else None
    })

@app.route('/api/idioms', methods=['GET'])
def get_idioms():
    return jsonify({
        "total": len(idioms),
        "idioms": idioms[:20]  # Return first 20
    })

@app.route('/api/search', methods=['GET'])
def search_idioms():
    query = request.args.get('q', '').lower()
    
    results = [
        idiom for idiom in idioms
        if query in idiom['idiom_en'].lower() or query in idiom.get('meaning', '').lower()
    ]
    
    return jsonify({
        "query": query,
        "count": len(results),
        "results": results[:10]
    })

if __name__ == '__main__':
    print("\n🚀 API Server Starting...")
    print("📍 http://localhost:5000")
    print("📖 Test: http://localhost:5000/api/idioms")
    app.run(debug=True, port=5000)