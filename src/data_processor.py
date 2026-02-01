"""
Data processing module for idiom-aware translation.
Handles loading Excel data, splitting into train/test, and tagging idioms.
"""

import pandas as pd
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_excel(filepath: str) -> pd.DataFrame:
    """
    Load Excel file with proper encoding for Sinhala text.
    
    Args:
        filepath: Path to Excel file
        
    Returns:
        DataFrame with idiom data
    """
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
        print(f"✓ Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        raise Exception(f"Failed to load Excel file: {e}")


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return statistics.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    stats = {
        'total_rows': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'evaluation_counts': df['Evaluation'].value_counts().to_dict() if 'Evaluation' in df.columns else {},
        'has_issues': False
    }
    
    # Check for critical missing values
    critical_cols = ['English Idiom', 'Figurative Example', 'Sinhala Translation Example', 'Sinhala Idiom']
    for col in critical_cols:
        if col in stats['missing_values'] and stats['missing_values'][col] > 0:
            print(f"⚠ Warning: {stats['missing_values'][col]} missing values in '{col}'")
            stats['has_issues'] = True
    
    return stats


def tag_idioms(sentence: str, idiom: str) -> str:
    """
    Add <IDIOM> tags around the idiom phrase in the sentence.
    Uses case-insensitive matching and handles whitespace variations.
    
    Args:
        sentence: The sentence containing the idiom
        idiom: The idiom phrase to tag
        
    Returns:
        Sentence with <IDIOM>...</IDIOM> tags around the idiom
    """
    if pd.isna(sentence) or pd.isna(idiom):
        return sentence
    
    # Clean the idiom and sentence
    idiom_clean = str(idiom).strip()
    sentence_str = str(sentence).strip()
    
    # Case-insensitive search for the idiom
    # Use word boundaries to avoid partial matches
    pattern = re.compile(re.escape(idiom_clean), re.IGNORECASE)
    
    # Find the idiom in the sentence
    match = pattern.search(sentence_str)
    
    if match:
        # Get the actual matched text (preserves original case)
        matched_text = sentence_str[match.start():match.end()]
        # Replace with tagged version
        tagged = sentence_str[:match.start()] + f"<IDIOM>{matched_text}</IDIOM>" + sentence_str[match.end():]
        return tagged
    else:
        # If exact match not found, try fuzzy matching (handle punctuation)
        # Remove common punctuation from idiom for matching
        idiom_words = re.findall(r'\w+', idiom_clean.lower())
        sentence_lower = sentence_str.lower()
        
        # Try to find the idiom words in sequence
        for i in range(len(sentence_str)):
            # Check if idiom starts at this position
            remaining = sentence_lower[i:]
            if all(word in remaining for word in idiom_words):
                # Try to extract the idiom portion
                words_found = []
                pos = i
                for word in idiom_words:
                    word_match = re.search(r'\b' + re.escape(word) + r'\b', sentence_lower[pos:])
                    if word_match:
                        pos += word_match.end()
                        
                # If we found all words in sequence, tag that portion
                if len(words_found) == len(idiom_words):
                    break
        
        # Return original if no match found
        print(f"⚠ Warning: Could not find idiom '{idiom_clean}' in sentence: {sentence_str[:50]}...")
        return sentence_str


def split_data(df: pd.DataFrame, test_size: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    First test_size rows go to test, rest to train.
    
    Args:
        df: DataFrame to split
        test_size: Number of rows for test set (default: 50)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    test_df = df.iloc[:test_size].copy()
    train_df = df.iloc[test_size:].copy()
    
    print(f"✓ Split data: {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df


def convert_to_json_format(df: pd.DataFrame, add_tags: bool = True) -> List[Dict]:
    """
    Convert DataFrame to JSON format with idiom tagging.
    
    Args:
        df: DataFrame to convert
        add_tags: Whether to add <IDIOM> tags to English sentences
        
    Returns:
        List of dictionaries in the required JSON format
    """
    data = []
    
    for idx, row in df.iterrows():
        # Get the English sentence and optionally tag the idiom
        source_en = str(row['Figurative Example']).strip()
        if add_tags and not pd.isna(row['English Idiom']):
            source_en = tag_idioms(source_en, row['English Idiom'])
        
        entry = {
            'idiom_en': str(row['English Idiom']).strip() if not pd.isna(row['English Idiom']) else "",
            'idiom_si': str(row['Sinhala Idiom']).strip() if not pd.isna(row['Sinhala Idiom']) else "",
            'meaning': str(row['What It Means']).strip() if not pd.isna(row['What It Means']) else "",
            'source_en': source_en,
            'target_si': str(row['Sinhala Translation Example']).strip() if not pd.isna(row['Sinhala Translation Example']) else "",
            'evaluation': str(row['Evaluation']).strip() if not pd.isna(row['Evaluation']) else ""
        }
        data.append(entry)
    
    return data


def export_to_json(data: List[Dict], output_path: str) -> None:
    """
    Export data to JSON file with proper UTF-8 encoding.
    
    Args:
        data: List of dictionaries to export
        output_path: Path to output JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Exported {len(data)} examples to {output_path}")


def process_dataset(excel_path: str, output_dir: str, test_size: int = 50) -> Dict[str, Any]:
    """
    Main pipeline to process the dataset.
    
    Args:
        excel_path: Path to input Excel file
        output_dir: Directory for output JSON files
        test_size: Number of examples for test set
        
    Returns:
        Dictionary with processing statistics
    """
    # Load data
    df = load_excel(excel_path)
    
    # Validate data
    stats = validate_data(df)
    
    # Split data
    train_df, test_df = split_data(df, test_size)
    
    # Convert to JSON format
    train_data = convert_to_json_format(train_df, add_tags=True)
    test_data = convert_to_json_format(test_df, add_tags=True)
    
    # Export to JSON
    train_path = f"{output_dir}/train.json"
    test_path = f"{output_dir}/test.json"
    
    export_to_json(train_data, train_path)
    export_to_json(test_data, test_path)
    
    stats['train_examples'] = len(train_data)
    stats['test_examples'] = len(test_data)
    stats['output_files'] = {
        'train': train_path,
        'test': test_path
    }
    
    return stats
