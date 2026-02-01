# Data Directory

This directory contains all datasets for the English-Sinhala idiom-aware translation project.

## 📁 Directory Structure

```
data/
├── raw/
│   ├── idiom_dataset.xlsx    # Original Excel dataset (510 rows)
│   └── .gitkeep
└── processed/
    ├── train.json            # Training data (460 examples, rows 51-510)
    ├── test.json             # Test data (50 examples, rows 1-50)
    ├── augmented_train.json  # Augmented training data (~920 examples)
    └── .gitkeep
```

## 📊 Raw Dataset (`raw/idiom_dataset.xlsx`)

### Columns

| Column | Name | Description | Example |
|--------|------|-------------|---------|
| A | Sinhala Idiom | Target idiom in Sinhala | අත් හිටලා |
| B | English Idiom | Source idiom in English | In abeyance |
| C | What It Means | Definition/explanation | A state of not happening or being used at present |
| D | Figurative Example | English sentence using the idiom | That matter has now been in abeyance for a number of years. |
| E | Sinhala Translation Example | Sinhala translation with idiom | ඒ කරුණ දැන් අවුරුදු ගණනකට අත් හිටලාය. |
| F | Evaluation | Validation status | Yes/No |

### Dataset Statistics

- **Total rows**: 510
- **Validated entries**: ~500+ (marked as "Yes")
- **Unique idioms**: ~500+ different idiom pairs
- **Language pair**: English (eng_Latn) ↔ Sinhala (sin_Sinh)

### Data Split

- **Test set**: First 50 rows (rows 1-50)
- **Training set**: Remaining rows (rows 51-510)

This split ensures:
- Fixed test set for consistent evaluation
- Larger training set for model learning
- No overlap between train and test

## 📄 Processed Data Format

### JSON Structure

Each example in the processed JSON files follows this structure:

```json
{
  "idiom_en": "in abeyance",
  "idiom_si": "අත් හිටලා",
  "meaning": "a state of not happening or being used at present",
  "source_en": "That matter has now been <IDIOM>in abeyance</IDIOM> for a number of years.",
  "target_si": "ඒ කරුණ දැන් අවුරුදු ගණනකට අත් හිටලාය.",
  "evaluation": "Yes"
}
```

### Field Descriptions

- **`idiom_en`**: The English idiom phrase
- **`idiom_si`**: The corresponding Sinhala idiom
- **`meaning`**: Definition/explanation of the idiom
- **`source_en`**: English sentence with idiom tagged using `<IDIOM>...</IDIOM>`
- **`target_si`**: Correct Sinhala translation (uses Sinhala idiom, not literal)
- **`evaluation`**: Quality validation flag from original dataset

### Idiom Tagging

The `source_en` field contains `<IDIOM>` tags around the idiom phrase:

**Before tagging:**
```
"That matter has now been in abeyance for a number of years."
```

**After tagging:**
```
"That matter has now been <IDIOM>in abeyance</IDIOM> for a number of years."
```

This tagging:
- Signals the model to translate idiomatically
- Helps identify the idiom location in the sentence
- Improves translation accuracy for idiomatic expressions

## 📈 Augmented Data (`augmented_train.json`)

The augmented dataset contains multiple variants of each training example:

1. **Original tagged** - Example with `<IDIOM>` tags (as in `train.json`)
2. **Untagged variant** - Same example without tags (for robustness)

**Purpose of augmentation:**
- Helps model generalize to both tagged and untagged input
- Increases training data size (~2x)
- Maintains quality (no synthetic/fake data)

### Augmentation Example

From one original example:
```json
{
  "source_en": "He is <IDIOM>above board</IDIOM> in all his dealings.",
  "target_si": "ඔහු ඔහුගේ සියළු ගනුදෙනුවල සෘජු ය."
}
```

We create:
```json
[
  {
    "source_en": "He is <IDIOM>above board</IDIOM> in all his dealings.",
    "target_si": "ඔහු ඔහුගේ සියළු ගනුදෙනුවල සෘජු ය."
  },
  {
    "source_en": "He is above board in all his dealings.",
    "target_si": "ඔහු ඔහුගේ සියළු ගනුදෙනුවල සෘජු ය.",
    "augmentation_type": "untagged"
  }
]
```

## 🔍 Data Quality

### Validation Checks

All processed data goes through:
- ✅ UTF-8 encoding verification for Sinhala text
- ✅ Missing value detection
- ✅ Idiom tagging accuracy check
- ✅ Train/test overlap prevention

### Known Issues

- Some idioms may not be found automatically (fuzzy matching used)
- Multi-word idioms with punctuation may need manual review
- Very long sentences may be truncated during training (max_length=128)

## 📝 Usage

### Loading Data

```python
import json

# Load training data
with open('data/processed/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Load test data
with open('data/processed/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"Training examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")
```

### Processing Pipeline

The data processing happens in `notebooks/01_data_preparation.ipynb`:

1. Load Excel file
2. Validate data quality
3. Split into train/test
4. Tag idioms with `<IDIOM>` markers
5. Export to JSON format

To reprocess data:
```python
from src.data_processor import process_dataset

stats = process_dataset(
    excel_path='data/raw/idiom_dataset.xlsx',
    output_dir='data/processed',
    test_size=50
)
```

## 🚫 What NOT to Do

- ❌ Don't modify the raw Excel file directly
- ❌ Don't add fake/synthetic idioms
- ❌ Don't change the train/test split (breaks reproducibility)
- ❌ Don't mix training and test data

## 📊 Statistics Summary

| Metric | Value |
|--------|-------|
| Total examples | 510 |
| Training examples | 460 |
| Test examples | 50 |
| Augmented training | ~920 |
| Unique idioms | ~500+ |
| Average sentence length | ~10-15 words |
| Encoding | UTF-8 |

## 🔄 Regenerating Data

If you need to regenerate the processed data:

```bash
# Run the data preparation notebook
jupyter nbconvert --to notebook --execute notebooks/01_data_preparation.ipynb

# Run the augmentation notebook
jupyter nbconvert --to notebook --execute notebooks/02_data_augmentation.ipynb
```

This will recreate:
- `data/processed/train.json`
- `data/processed/test.json`
- `data/processed/augmented_train.json`

---

**Note**: The raw dataset (`idiom_dataset.xlsx`) is the source of truth. All processed files can be regenerated from it.
