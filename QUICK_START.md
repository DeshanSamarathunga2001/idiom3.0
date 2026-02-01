# Quick Start Guide

Get started with the English-Sinhala Idiom-Aware Translation System in 5 minutes!

## Prerequisites

- Python 3.8+
- 8GB+ RAM
- 10GB disk space

## Installation (2 minutes)

```bash
# Clone repository
git clone https://github.com/DeshanSamarathunga2001/idiom3.0.git
cd idiom3.0

# Install dependencies
pip install -r requirements.txt
```

## Running the Pipeline (3 steps)

### Step 1: Data Preparation

```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```

**What it does:**
- Loads 510 idiom examples from Excel
- Splits into train (460) and test (50)
- Tags idioms with `<IDIOM>` markers
- Exports to JSON

**Output:**
- `data/processed/train.json`
- `data/processed/test.json`

### Step 2: Data Augmentation

```bash
jupyter notebook notebooks/02_data_augmentation.ipynb
```

**What it does:**
- Creates augmented training examples (~2x)
- Adds untagged variants
- Validates quality

**Output:**
- `data/processed/augmented_train.json`

### Step 3: Model Training

```bash
jupyter notebook notebooks/03_model_training.ipynb
```

**What it does:**
- Downloads NLLB-200-distilled-600M (~1.2GB)
- Applies LoRA adapters
- Trains for 10 epochs
- Saves checkpoints

**Time:**
- CPU: 2-4 hours
- GPU: 30-60 minutes

**Output:**
- `models/final/` (fine-tuned model)

### Step 4: Testing

```bash
jupyter notebook notebooks/04_inference_test.ipynb
```

**What it does:**
- Translates 50 test examples
- Side-by-side comparisons
- Quick quality check

**Output:**
- `outputs/predictions/test_results.json`

### Step 5: Evaluation

```bash
jupyter notebook notebooks/05_evaluation.ipynb
```

**What it does:**
- Calculates BLEU scores
- Measures idiom accuracy
- Generates visualizations

**Output:**
- `outputs/metrics/evaluation_results.json`
- Visualization plots

## Alternative: Run All at Once

```bash
# Execute all notebooks sequentially
for nb in notebooks/*.ipynb; do
    jupyter nbconvert --to notebook --execute "$nb"
done
```

## Using Individual Components

### Just Process Data

```python
from src.data_processor import process_dataset

stats = process_dataset(
    excel_path='data/raw/idiom_dataset.xlsx',
    output_dir='data/processed',
    test_size=50
)
```

### Just Translate (after training)

```python
from src.inference import load_trained_model, translate

model, tokenizer = load_trained_model('models/final')
result = translate(
    "Test <IDIOM>in abeyance</IDIOM>",
    model,
    tokenizer
)
print(result)
```

## Expected Results

After running all notebooks:

✅ **BLEU Score**: 30-50 (translation quality)  
✅ **Idiom Accuracy**: 60-80% (correct Sinhala idiom usage)  
✅ **Training Loss**: Decreasing over epochs  
✅ **Test Predictions**: 50 translations saved  

## Common Issues

### Out of Memory
- Reduce `batch_size` in `config/training_config.yaml`
- Use CPU instead of GPU (slower but works)

### Slow Training
- Expected on CPU (2-4 hours)
- Use GPU if available
- Reduce `num_epochs` for faster testing

### Model Download Fails
- Check internet connection
- Model downloads automatically from HuggingFace
- ~1.2GB download size

## File Structure After Running

```
idiom3.0/
├── data/processed/
│   ├── train.json          ✅ 460 examples
│   ├── test.json           ✅ 50 examples
│   └── augmented_train.json ✅ ~920 examples
├── models/final/           ✅ Fine-tuned model
├── outputs/
│   ├── predictions/        ✅ Test results
│   └── metrics/            ✅ Evaluation metrics
```

## What's Next?

1. **Experiment**: Try different hyperparameters
2. **Extend**: Add more idioms to the dataset
3. **Deploy**: Create a simple translation API
4. **Research**: Write up your findings

## Help & Support

- **Documentation**: See README.md
- **Issues**: Open GitHub issue
- **Questions**: Check IMPLEMENTATION_SUMMARY.md

---

**Total Setup Time**: 5 minutes  
**Total Pipeline Time**: 2-4 hours (mostly training)  
**Difficulty**: Beginner-friendly 🎓
