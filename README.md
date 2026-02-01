# English-Sinhala Idiom-Aware Translation

A complete fine-tuning pipeline for NLLB-200-distilled-600M that translates English sentences with idioms to Sinhala, preserving idiomatic meaning (not literal translation).

## 🎯 Overview

This project demonstrates idiom-aware neural machine translation using explicit idiom tagging. By marking idioms with `<IDIOM>` tags in the source text, we teach the model to translate idiomatically rather than literally.

**Example:**
```
English: "That matter has now been <IDIOM>in abeyance</IDIOM> for a number of years."
Sinhala:  "ඒ කරුණ දැන් අවුරුදු ගණනකට අත් හිටලාය."
          (Uses Sinhala idiom "අත් හිටලා" instead of literal translation)
```

## 📋 Prerequisites

- Python 3.8+
- 8GB+ RAM (16GB recommended)
- GPU recommended (CUDA/MPS) or CPU (slower training)
- 10GB free disk space

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/DeshanSamarathunga2001/idiom3.0.git
cd idiom3.0
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Notebooks in Order

Execute the following Jupyter notebooks sequentially:

1. **`notebooks/01_data_preparation.ipynb`** - Process Excel data
   - Loads `data/raw/idiom_dataset.xlsx`
   - Validates data quality
   - Splits into train/test (first 50 rows → test)
   - Tags idioms with `<IDIOM>` markers
   - Exports to JSON format

2. **`notebooks/02_data_augmentation.ipynb`** - Generate training variants
   - Creates augmented examples from base dataset
   - Adds untagged variants for robustness
   - Validates augmentation quality
   - Saves to `data/processed/augmented_train.json`

3. **`notebooks/03_model_training.ipynb`** - Fine-tune NLLB model
   - Loads NLLB-200-distilled-600M base model
   - Adds special tokens (`<IDIOM>`, `</IDIOM>`)
   - Applies LoRA adapters for efficient fine-tuning
   - Trains on augmented dataset
   - Saves checkpoints and final model

4. **`notebooks/04_inference_test.ipynb`** - Test on 50 examples
   - Loads fine-tuned model
   - Generates translations for test set
   - Displays side-by-side comparisons
   - Saves predictions for evaluation

5. **`notebooks/05_evaluation.ipynb`** - Calculate metrics
   - Computes BLEU scores
   - Measures idiom accuracy
   - Analyzes per-idiom performance
   - Generates visualizations and reports

### 4. Alternative: Run via Command Line

You can also execute notebooks programmatically:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_preparation.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_data_augmentation.ipynb
# ... and so on
```

## 📊 Dataset

The dataset is located at `data/raw/idiom_dataset.xlsx` with 510 rows containing:

- **Sinhala Idiom**: Target idiom in Sinhala (e.g., "අත් හිටලා")
- **English Idiom**: Source idiom in English (e.g., "In abeyance")
- **What It Means**: Definition/explanation
- **Figurative Example**: English sentence using the idiom
- **Sinhala Translation Example**: Correct Sinhala translation with idiom
- **Evaluation**: Validation status (Yes/No)

**Split:**
- Training: 460 examples (rows 51-510)
- Test: 50 examples (rows 1-50)

## 🏗️ Project Structure

```
idiom3.0/
├── config/
│   └── training_config.yaml      # Training hyperparameters
├── data/
│   ├── raw/
│   │   └── idiom_dataset.xlsx    # Original dataset
│   └── processed/
│       ├── train.json            # Training data
│       ├── test.json             # Test data
│       └── augmented_train.json  # Augmented training data
├── models/
│   ├── checkpoints/              # Training checkpoints
│   └── final/                    # Final fine-tuned model
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_data_augmentation.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_inference_test.ipynb
│   └── 05_evaluation.ipynb
├── outputs/
│   ├── predictions/              # Model predictions
│   ├── metrics/                  # Evaluation metrics
│   └── logs/                     # Training logs
├── src/
│   ├── data_processor.py         # Data loading and processing
│   ├── augmentation.py           # Data augmentation
│   ├── trainer.py                # Model training with LoRA
│   ├── inference.py              # Translation inference
│   └── evaluation.py             # Metrics calculation
└── README.md
```

## 🔧 Configuration

Edit `config/training_config.yaml` to customize training:

```yaml
model:
  base_model: "facebook/nllb-200-distilled-600M"
  source_lang: "eng_Latn"
  target_lang: "sin_Sinh"

lora:
  r: 16                    # LoRA rank
  lora_alpha: 32
  lora_dropout: 0.05

training:
  learning_rate: 3e-4
  num_epochs: 10
  batch_size: 4
  gradient_accumulation_steps: 4
  max_length: 128
```

## 📈 Expected Results

After training, you should see:

- **BLEU Score**: 30-50 (varies by dataset quality)
- **Idiom Accuracy**: 60-80% (model uses correct Sinhala idiom)
- **Training Time**: 
  - CPU: ~2-4 hours
  - GPU (CUDA): ~30-60 minutes

Results will vary based on:
- Hardware capabilities
- Number of training epochs
- Dataset size and quality

## 🔬 Research Contribution

This project demonstrates:

1. **Idiom-Aware Translation**: Explicit control over idiomatic vs literal translation
2. **Low-Resource Fine-Tuning**: LoRA enables efficient training with limited data
3. **Explicit Tagging**: `<IDIOM>` markers provide translation hints

### What This Project Claims:
✅ Idiom tagging reduces literal translation in controlled settings  
✅ Small-scale fine-tuning can adapt large models to specific tasks  
✅ Proof-of-concept for idiom-aware translation systems  

### What This Project Does NOT Claim:
❌ General idiom understanding without tagging  
❌ Translation of unseen/novel idioms  
❌ Production-ready translation system  

## ⚠️ Limitations

1. **Limited Coverage**: Only works with idioms present in training data
2. **Manual Tagging Required**: Input must have `<IDIOM>` tags for best results
3. **Proof-of-Concept**: Designed for controlled evaluation, not production use
4. **Small Dataset**: 510 examples may not generalize to all contexts
5. **Single Language Pair**: English-Sinhala only

## 🛠️ Development

### Code Quality

All Python modules follow PEP 8 style guidelines and include:
- Type hints for function parameters
- Comprehensive docstrings
- Error handling and validation
- UTF-8 encoding for Sinhala text

### Testing

To test individual components:

```python
# Test data processor
from src.data_processor import process_dataset
stats = process_dataset('data/raw/idiom_dataset.xlsx', 'data/processed', test_size=50)

# Test translation
from src.inference import load_trained_model, translate
model, tokenizer = load_trained_model('models/final')
result = translate("Test <IDIOM>in abeyance</IDIOM>", model, tokenizer)
```

## 📚 References

- **NLLB Model**: [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **PEFT Library**: [HuggingFace PEFT](https://github.com/huggingface/peft)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{idiom-aware-translation-2024,
  author = {Deshan Samarathunga},
  title = {Idiom-Aware English-Sinhala Translation with NLLB},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/DeshanSamarathunga2001/idiom3.0}
}
```

## 📧 Contact

For questions or issues, please open a GitHub issue or contact:
- **Author**: Deshan Samarathunga
- **Repository**: https://github.com/DeshanSamarathunga2001/idiom3.0

## 📄 License

This project is available for educational and research purposes.

---

**Note**: This is a final-year research project demonstrating idiom-aware translation techniques. It is not intended for production use without further development and testing.
