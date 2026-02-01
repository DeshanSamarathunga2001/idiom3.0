"""
Data augmentation module for creating training variants.
Creates conservative augmentations without fake data.
"""

import json
from typing import Dict, List
from pathlib import Path


def create_augmented_examples(data: List[Dict]) -> List[Dict]:
    """
    Create augmented training examples from base dataset.
    
    Augmentation strategies:
    1. Tagged idiomatic (main training example) - already done in data_processor
    2. Untagged version (for robustness)
    3. Conservative variations (only when safe)
    
    Args:
        data: List of training examples
        
    Returns:
        Augmented dataset
    """
    augmented = []
    
    for idx, example in enumerate(data):
        # Original tagged example
        augmented.append(example)
        
        # Create untagged version (remove <IDIOM> tags)
        # This helps model generalize to non-tagged input
        untagged_source = example['source_en'].replace('<IDIOM>', '').replace('</IDIOM>', '')
        
        untagged_example = {
            **example,
            'source_en': untagged_source,
            'augmentation_type': 'untagged'
        }
        augmented.append(untagged_example)
    
    print(f"✓ Created {len(augmented)} augmented examples from {len(data)} originals")
    print(f"  - Original tagged: {len(data)}")
    print(f"  - Untagged variants: {len(data)}")
    
    return augmented


def validate_augmentation(original: List[Dict], augmented: List[Dict]) -> Dict[str, any]:
    """
    Validate augmented data quality.
    
    Args:
        original: Original dataset
        augmented: Augmented dataset
        
    Returns:
        Validation statistics
    """
    stats = {
        'original_count': len(original),
        'augmented_count': len(augmented),
        'augmentation_ratio': len(augmented) / len(original) if len(original) > 0 else 0,
        'issues': []
    }
    
    # Check that all original examples are preserved
    original_sources = set(ex['source_en'] for ex in original)
    augmented_sources = set(ex['source_en'] for ex in augmented)
    
    # Validate no data loss
    for orig_ex in original:
        matching = [aug_ex for aug_ex in augmented if aug_ex['idiom_en'] == orig_ex['idiom_en']]
        if len(matching) == 0:
            stats['issues'].append(f"Original example with idiom '{orig_ex['idiom_en']}' not found in augmented data")
    
    if len(stats['issues']) == 0:
        print("✓ Augmentation validation passed")
    else:
        print(f"⚠ Found {len(stats['issues'])} validation issues")
    
    return stats


def save_augmented_data(data: List[Dict], output_path: str) -> None:
    """
    Save augmented data to JSON file.
    
    Args:
        data: Augmented dataset
        output_path: Path to output file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved {len(data)} augmented examples to {output_path}")


def augment_dataset(input_path: str, output_path: str) -> Dict[str, any]:
    """
    Main pipeline for data augmentation.
    
    Args:
        input_path: Path to input JSON file (train.json)
        output_path: Path to output JSON file (augmented_train.json)
        
    Returns:
        Augmentation statistics
    """
    # Load training data
    with open(input_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"✓ Loaded {len(train_data)} training examples from {input_path}")
    
    # Create augmented examples
    augmented_data = create_augmented_examples(train_data)
    
    # Validate augmentation
    validation_stats = validate_augmentation(train_data, augmented_data)
    
    # Save augmented data
    save_augmented_data(augmented_data, output_path)
    
    return {
        'input_examples': len(train_data),
        'output_examples': len(augmented_data),
        'validation': validation_stats
    }
