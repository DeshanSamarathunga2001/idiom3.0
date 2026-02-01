# 🚂 Training Guide - Idiom-Aware Translation Model

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run training notebook:**
```bash
jupyter notebook notebooks/03_model_training.ipynb
```

3. **Run all cells in order** (0-13)

---

## Expected Results

### Training Loss
```
Step 10:  Loss: 2.456  ✅ Good!
Step 50:  Loss: 1.823
Step 100: Loss: 1.512
Step 200: Loss: 1.234
```

**⚠️ If loss > 10:** There's a tokenizer configuration issue!

### Evaluation Metrics
```
BLEU Score:             ~25-30
chrF Score:             ~50-60
Idiom Accuracy:         ~60-70%
Idiom Partial Accuracy: ~80-90%
```

---

## Notebook Cells Overview

- **Cell 0-2:** Setup and configuration
- **Cell 3-4:** Model loading and verification
- **Cell 5:** Apply LoRA adapters
- **Cell 6-7:** Prepare dataset
- **Cell 8:** Train model (2-4 hours)
- **Cell 9:** Save trained model
- **Cell 10:** Test translations
- **Cell 11-13:** Full evaluation

---

## Troubleshooting

### High Training Loss (>10)
- Check vocabulary size: `len(tokenizer)` should be **256204**
- Verify no special tokens were added
- Restart kernel and re-run from Cell 0

### CUDA Out of Memory
- Reduce `batch_size` in `config/training_config.yaml`
- Increase `gradient_accumulation_steps`

### Import Errors
- Run Cell 0 first to fix directory
- Ensure `src/inference.py` and `src/evaluation.py` exist

---

## Files Structure

```
idiom3.0/
├── notebooks/
│   ├── 03_model_training.ipynb  ← Main training notebook
│   └── 04_evaluation_only.ipynb ← Evaluation only
├── src/
│   ├── trainer.py               ← Training functions
│   ├── inference.py             ← Translation functions
│   └── evaluation.py            ← Metrics calculation
├── outputs/
│   ├── predictions.json         ← Generated predictions
│   └── full_evaluation.json     ← Full results
└── models/
    ├── checkpoints/             ← Training checkpoints
    └── final/                   ← Final trained model
```

---

## Detailed Cell Descriptions

### Cell 0: Directory Setup
- Changes to project root if running from notebooks folder
- Verifies all required files exist
- Prevents common path-related errors

### Cell 1: Imports
- Imports all necessary modules from src/
- Verifies PyTorch installation and CUDA availability
- Checks all dependencies are correctly installed

### Cell 2: Load Configuration
- Loads training settings from `config/training_config.yaml`
- Displays key parameters (model, epochs, batch size, learning rate)
- Validates configuration is correct

### Cell 3: Setup Model (CRITICAL)
- Loads NLLB-200 base model
- **DOES NOT add special tokens** (vocabulary stays at 256204)
- Verifies language support for English and Sinhala
- Asserts vocabulary size is correct

### Cell 4: Test Tokenization
- Tests how idiom tags are tokenized
- Shows that `<IDIOM>` and `</IDIOM>` are treated as regular text
- Demonstrates the model will learn to handle tags naturally

### Cell 5: Apply LoRA
- Applies LoRA (Low-Rank Adaptation) adapters
- Makes model training efficient by only training ~1% of parameters
- Prints trainable parameters count

### Cell 6: Clean Checkpoints
- Removes old checkpoint files
- Creates fresh checkpoint directory
- Prevents disk space issues

### Cell 7: Prepare Dataset
- Loads training data from JSON
- Tokenizes source and target texts
- Creates HuggingFace Dataset object
- Shows sample to verify preparation

### Cell 8: Train Model
- Main training cell (takes 2-4 hours)
- Displays expected loss values
- Saves checkpoints every epoch
- **Monitor loss values** - should decrease steadily

### Cell 9: Save Model
- Saves final trained model to `models/final/`
- Saves both model weights and tokenizer
- Verifies files were saved correctly

### Cell 10: Test Translations
- Tests model on 5 example sentences with idioms
- Shows both source and translated text
- Extracts and displays idiom tags from both
- Quick sanity check that model works

### Cell 11: Full Evaluation
- Evaluates on 100 test examples
- Calculates BLEU, chrF, and idiom-specific metrics
- Saves detailed predictions to JSON
- Prints comprehensive evaluation report

### Cell 12: Show Example Predictions
- Loads saved predictions from JSON
- Displays first 5 examples with details
- Shows source, prediction, reference, and extracted idioms
- Helps understand model behavior

### Cell 13: Summary Statistics
- Calculates aggregate statistics
- Shows idiom translation accuracy
- Provides quick overview of model performance

---

## Evaluation Metrics Explained

### BLEU Score
- Measures n-gram overlap between prediction and reference
- Range: 0-100 (higher is better)
- Expected: 25-30 for this task

### chrF Score  
- Character-level F-score
- More robust for morphologically rich languages
- Expected: 50-60 for this task

### Idiom Accuracy
- Percentage of idioms correctly translated with tags
- Strict metric requiring exact match
- Expected: 60-70%

### Idiom Partial Accuracy
- Percentage of idioms with any correct translation
- More lenient, includes idioms without tags
- Expected: 80-90%

---

## Advanced Usage

### Custom Evaluation Dataset
To evaluate on a different dataset:

```python
# In Cell 11, replace:
test_data = all_data[:100]

# With:
test_data = all_data[100:200]  # Different subset
# Or load different file:
with open('data/custom_test.json', 'r') as f:
    test_data = json.load(f)
```

### Adjust Training Parameters
Edit `config/training_config.yaml`:

```yaml
training:
  num_epochs: 15        # More epochs for better results
  batch_size: 8         # Larger batch if you have GPU memory
  learning_rate: 0.0001 # Lower for more stable training
```

### Resume Training
To continue from a checkpoint:

```python
# In Cell 3, after loading model:
from peft import PeftModel
checkpoint_path = "models/checkpoints/checkpoint-100"
model = PeftModel.from_pretrained(model, checkpoint_path)
```

---

## Using the Evaluation-Only Notebook

For evaluating an already-trained model without retraining:

1. Open `notebooks/04_evaluation_only.ipynb`
2. Ensure trained model exists at `models/final/`
3. Run all cells in order

This notebook:
- Loads saved model
- Evaluates on full dataset
- Saves results to `outputs/full_evaluation.json`

---

## Performance Tips

### Speed Up Training
- Use GPU: Set `use_fp16: true` in config
- Increase `gradient_accumulation_steps`
- Reduce `max_length` if sequences are short

### Improve Accuracy
- Increase `num_epochs`
- Lower `learning_rate`
- Increase LoRA `r` parameter
- Use more training data

### Save Disk Space
- Set `save_total_limit: 2` in training args
- Delete old checkpoints after training
- Use `fp16` to reduce model size

---

## Common Issues

### Issue: "FileNotFoundError: augmented_train.json not found"
**Solution:** Run data preparation notebooks first:
```bash
jupyter notebook notebooks/01_data_preparation.ipynb
jupyter notebook notebooks/02_data_augmentation.ipynb
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in config:
```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8
```

### Issue: "Vocabulary size is 256205, should be 256204"
**Solution:** Special tokens were added by mistake. Restart kernel and ensure Cell 3 uses `special_tokens=None`.

### Issue: Training loss is NaN or very high (>50)
**Solution:** 
- Check data quality - ensure JSON is properly formatted
- Verify tokenizer vocabulary size
- Lower learning rate to 1e-4

---

## Next Steps

After training completes successfully:

1. **Test on new sentences:** Use the translate_with_idioms function
2. **Deploy model:** Export to ONNX or use with Transformers pipeline
3. **Fine-tune further:** Use the checkpoint for continued training
4. **Evaluate on held-out test set:** Use separate test data

---

## Support

For issues or questions:
- Check the Troubleshooting section
- Review cell outputs for error messages
- Ensure all dependencies are installed
- Verify data files exist and are properly formatted

---

**Happy Training! 🚂🎯**
