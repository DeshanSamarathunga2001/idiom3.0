"""
Evaluation module for translation quality assessment.
Calculates BLEU scores and idiom-specific metrics.
"""

import json
from typing import Dict, List, Any
from pathlib import Path
import sacrebleu
from collections import defaultdict


def calculate_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Calculate BLEU score for translations.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        
    Returns:
        BLEU score (0-100)
    """
    # Ensure references is a list of lists for sacrebleu
    refs = [[ref] for ref in references]
    
    # Calculate BLEU
    bleu = sacrebleu.corpus_bleu(predictions, list(zip(*refs)))
    
    return bleu.score


def check_idiom_presence(translation: str, idiom: str) -> bool:
    """
    Check if the Sinhala idiom appears in the translation.
    
    Args:
        translation: Translated text
        idiom: Expected Sinhala idiom
        
    Returns:
        True if idiom is present
    """
    if not idiom or not translation:
        return False
    
    # Simple substring check
    # Note: Could be enhanced with fuzzy matching
    return idiom.strip() in translation.strip()


def check_literal_translation(prediction: str, source: str, idiom_en: str) -> bool:
    """
    Heuristic check if translation might be literal.
    This is a simplified check - looks for English idiom words in output.
    
    Args:
        prediction: Predicted translation
        source: Source English text
        idiom_en: English idiom phrase
        
    Returns:
        True if translation appears to be literal
    """
    # This is a simple heuristic
    # A more sophisticated approach would use backtranslation
    # For now, we just check if English words appear in Sinhala output
    
    # Check if any ASCII characters (indicating English) appear in prediction
    has_ascii = any(ord(c) < 128 and c.isalpha() for c in prediction)
    
    return has_ascii


def evaluate_single(
    prediction: str,
    reference: str,
    idiom_en: str,
    idiom_si: str,
    source: str
) -> Dict[str, Any]:
    """
    Evaluate a single translation.
    
    Args:
        prediction: Predicted translation
        reference: Reference translation
        idiom_en: English idiom
        idiom_si: Expected Sinhala idiom
        source: Source English text
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate sentence-level BLEU
    bleu = calculate_bleu([prediction], [reference])
    
    # Check idiom presence
    idiom_correct = check_idiom_presence(prediction, idiom_si)
    
    # Check for literal translation
    is_literal = check_literal_translation(prediction, source, idiom_en)
    
    return {
        'bleu': bleu,
        'idiom_correct': idiom_correct,
        'avoided_literal': not is_literal,
        'prediction': prediction,
        'reference': reference
    }


def evaluate_batch(predictions: List[str], test_data: List[Dict]) -> Dict[str, Any]:
    """
    Evaluate a batch of translations.
    
    Args:
        predictions: List of predicted translations
        test_data: List of test examples with reference data
        
    Returns:
        Dictionary with aggregate metrics
    """
    if len(predictions) != len(test_data):
        raise ValueError(f"Number of predictions ({len(predictions)}) must match test data ({len(test_data)})")
    
    # Evaluate each example
    results = []
    for pred, example in zip(predictions, test_data):
        result = evaluate_single(
            prediction=pred,
            reference=example['target_si'],
            idiom_en=example['idiom_en'],
            idiom_si=example['idiom_si'],
            source=example['source_en']
        )
        result['idiom_en'] = example['idiom_en']
        result['idiom_si'] = example['idiom_si']
        result['source'] = example['source_en']
        results.append(result)
    
    # Calculate aggregate metrics
    references = [ex['target_si'] for ex in test_data]
    
    metrics = {
        'overall_bleu': calculate_bleu(predictions, references),
        'idiom_accuracy': sum(r['idiom_correct'] for r in results) / len(results) * 100,
        'literal_translation_rate': sum(not r['avoided_literal'] for r in results) / len(results) * 100,
        'total_examples': len(results),
        'detailed_results': results
    }
    
    # Per-idiom breakdown
    idiom_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'bleu_sum': 0})
    for result in results:
        idiom = result['idiom_en']
        idiom_stats[idiom]['total'] += 1
        idiom_stats[idiom]['bleu_sum'] += result['bleu']
        if result['idiom_correct']:
            idiom_stats[idiom]['correct'] += 1
    
    # Calculate per-idiom metrics
    per_idiom = {}
    for idiom, stats in idiom_stats.items():
        per_idiom[idiom] = {
            'accuracy': stats['correct'] / stats['total'] * 100,
            'avg_bleu': stats['bleu_sum'] / stats['total'],
            'count': stats['total']
        }
    
    metrics['per_idiom_performance'] = per_idiom
    
    return metrics


def generate_report(metrics: Dict, output_path: str = None) -> str:
    """
    Generate a human-readable evaluation report.
    
    Args:
        metrics: Evaluation metrics dictionary
        output_path: Optional path to save report
        
    Returns:
        Report text
    """
    report = []
    report.append("=" * 60)
    report.append("EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    report.append("Overall Metrics:")
    report.append(f"  BLEU Score: {metrics['overall_bleu']:.2f}")
    report.append(f"  Idiom Accuracy: {metrics['idiom_accuracy']:.1f}%")
    report.append(f"  Literal Translation Rate: {metrics['literal_translation_rate']:.1f}%")
    report.append(f"  Total Examples: {metrics['total_examples']}")
    report.append("")
    
    if 'per_idiom_performance' in metrics:
        report.append("Per-Idiom Performance:")
        report.append("-" * 60)
        
        # Sort by count (most common first)
        sorted_idioms = sorted(
            metrics['per_idiom_performance'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        for idiom, stats in sorted_idioms[:10]:  # Top 10
            report.append(f"  {idiom}:")
            report.append(f"    Accuracy: {stats['accuracy']:.1f}% ({stats['count']} examples)")
            report.append(f"    Avg BLEU: {stats['avg_bleu']:.2f}")
        
        if len(sorted_idioms) > 10:
            report.append(f"  ... and {len(sorted_idioms) - 10} more idioms")
    
    report.append("")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"✓ Report saved to {output_path}")
    
    return report_text


def save_metrics(metrics: Dict, output_path: str) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        output_path: Path to output file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Metrics saved to {output_path}")


def evaluate_model(
    model,
    tokenizer,
    test_data: List[Dict],
    src_lang: str = "eng_Latn",
    tgt_lang: str = "sin_Sinh",
    max_length: int = 128,
    num_beams: int = 5
) -> tuple:
    """
    Evaluate model on test data.
    
    Args:
        model: Translation model
        tokenizer: Tokenizer
        test_data: List of test examples with 'source_en', 'target_si', 'idiom_en', 'idiom_si'
        src_lang: Source language code
        tgt_lang: Target language code
        max_length: Maximum generation length
        num_beams: Number of beams for generation
        
    Returns:
        Tuple of (metrics_dict, predictions_list)
    """
    from .inference import batch_translate, extract_idioms
    
    # Extract source texts
    source_texts = [ex['source_en'] for ex in test_data]
    
    # Generate translations
    print(f"Translating {len(source_texts)} examples...")
    predictions = batch_translate(
        texts=source_texts,
        model=model,
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_length=max_length,
        batch_size=8,
        num_beams=num_beams
    )
    
    # Calculate metrics
    references = [ex['target_si'] for ex in test_data]
    bleu_score = calculate_bleu(predictions, references)
    
    # Calculate chrF score
    try:
        chrf = sacrebleu.corpus_chrf(predictions, [[ref] for ref in references])
        chrf_score = chrf.score
    except Exception as e:
        print(f"⚠️  Could not calculate chrF: {e}")
        chrf_score = 0.0
    
    # Idiom-specific metrics
    idiom_correct = 0
    idiom_partial = 0
    total_with_idioms = 0
    
    detailed_predictions = []
    
    for i, (pred, example) in enumerate(zip(predictions, test_data)):
        source = example['source_en']
        reference = example['target_si']
        idiom_si = example.get('idiom_si', '')
        
        # Extract idioms from source, prediction, and reference
        source_idioms = extract_idioms(source)
        predicted_idioms = extract_idioms(pred)
        reference_idioms = extract_idioms(reference)
        
        # Check if idiom is correctly translated
        has_idiom = len(source_idioms) > 0
        if has_idiom:
            total_with_idioms += 1
            
            # Exact match: predicted idioms match reference idioms
            if predicted_idioms and reference_idioms:
                if any(p_idiom in r_idiom or r_idiom in p_idiom 
                       for p_idiom in predicted_idioms 
                       for r_idiom in reference_idioms):
                    idiom_correct += 1
                    idiom_partial += 1
                elif idiom_si and any(idiom_si in pred_idiom for pred_idiom in predicted_idioms):
                    idiom_partial += 1
            # Partial match: idiom_si appears in prediction (even without tags)
            elif idiom_si and idiom_si in pred:
                idiom_partial += 1
        
        detailed_predictions.append({
            'source': source,
            'prediction': pred,
            'reference': reference,
            'source_idioms': source_idioms,
            'predicted_idioms': predicted_idioms,
            'reference_idioms': reference_idioms,
            'bleu': calculate_bleu([pred], [reference])
        })
    
    # Calculate idiom accuracy
    idiom_accuracy = (idiom_correct / total_with_idioms * 100) if total_with_idioms > 0 else 0.0
    idiom_partial_accuracy = (idiom_partial / total_with_idioms * 100) if total_with_idioms > 0 else 0.0
    
    metrics = {
        'bleu': bleu_score,
        'chrf': chrf_score,
        'idiom_accuracy': idiom_accuracy,
        'idiom_partial_accuracy': idiom_partial_accuracy,
        'total_examples': len(test_data),
        'examples_with_idioms': total_with_idioms,
        'idioms_correctly_translated': idiom_correct,
        'idioms_partially_translated': idiom_partial
    }
    
    return metrics, detailed_predictions


def print_evaluation_report(metrics: Dict) -> None:
    """
    Print a formatted evaluation report.
    
    Args:
        metrics: Metrics dictionary from evaluate_model
    """
    print("=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    print()
    print(f"Overall Metrics:")
    print(f"  BLEU Score:             {metrics.get('bleu', 0):.2f}")
    print(f"  chrF Score:             {metrics.get('chrf', 0):.2f}")
    print()
    print(f"Idiom Translation:")
    print(f"  Total examples:         {metrics.get('total_examples', 0)}")
    print(f"  Examples with idioms:   {metrics.get('examples_with_idioms', 0)}")
    print(f"  Idiom Accuracy:         {metrics.get('idiom_accuracy', 0):.2f}%")
    print(f"  Idiom Partial Accuracy: {metrics.get('idiom_partial_accuracy', 0):.2f}%")
    print()
    print("=" * 80)


def save_predictions(
    predictions: List[Dict],
    references: List[str],
    source_texts: List[str],
    output_path: str
) -> None:
    """
    Save predictions to JSON file.
    
    Args:
        predictions: List of prediction dictionaries from evaluate_model
        references: List of reference translations (not used, kept for compatibility)
        source_texts: List of source texts (not used, kept for compatibility)
        output_path: Path to save predictions
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Predictions saved to {output_path}")
