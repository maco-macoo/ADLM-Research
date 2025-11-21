import collections
import json
import math
import os
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import datasets
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import transformers
from tqdm import tqdm


CACHE_DIR = '/home_lab/b220083/PROJECT/datasets'
DATASET_NAME = 'openwebtext'
DATASET_SPLIT = 'train[:-100000]'  # Exclude last 100k as in dataloader.py
OUTPUT_DIR = '/home_lab/b220083/PROJECT/ADLM-Research/outputs/tf-idf'
OUTPUT_PATH = OUTPUT_DIR + '/tf-idf_stats.json'
VALIDATION_OUTPUT_PATH = OUTPUT_DIR + '/tf-idf_validation.json'
VALIDATION_MATRIX_PATH = OUTPUT_DIR + '/tf-idf_validation_matrix.txt'

# TF-IDF Vocabulary Selection Parameters
# MAX_FEATURES: Maximum vocabulary size. Larger = more tokens but higher computation cost.
#   Recommendation: 50k-200k for large datasets. 100k is a good balance.
MAX_FEATURES = 100_000

# NGRAM_RANGE: Token n-gram range. (1,1) = unigrams only, (1,2) = unigrams + bigrams.
#   Recommendation: (1,1) for most cases. Bigrams increase vocabulary size significantly.
NGRAM_RANGE = (1, 1)

# MIN_DF: Minimum document frequency. Tokens appearing in fewer documents are excluded.
#   If < 1.0: treated as proportion of total documents
#   If >= 1.0: treated as absolute count
#   Recommendation: 5-10 for large datasets (removes typos/rare terms)
MIN_DF = 5

# MAX_DF: Maximum document frequency. Tokens appearing in too many documents are excluded.
#   If <= 1.0: treated as proportion of total documents
#   If > 1.0: treated as absolute count
#   Recommendation: 0.2-0.5 (removes common stopwords like "the", "a", "is")
MAX_DF = 0.2

# Display Parameters
TOP_K_PRINT = 15  # Number of top tokens to print

# Percentile Parameters
# PERCENTILE_POINTS: Percentiles to compute (0.01% steps for high resolution)
#   This allows flexible threshold selection without recomputation.
PERCENTILE_STEP = 0.01
PERCENTILE_POINTS = np.round(
    np.arange(PERCENTILE_STEP, 100.0, PERCENTILE_STEP),
    4).tolist()

# DEFAULT_THRESHOLD_PERCENTILE: Default threshold for sample scoring (optional, can be overridden)
#   Note: All percentiles are computed and saved, so this is just for display/sample scoring.
DEFAULT_THRESHOLD_PERCENTILE = 90

TOKENIZER_NAME = 'gpt2'

SAMPLE = (
    'OpenAI developed large language models, and researchers use them '
    'to explore natural language understanding and generation.'
)


def load_cached_dataset():
    """
    Load dataset from cache, matching dataloader.py behavior.

    Note: With streaming=False, Hugging Face Datasets loads the full dataset
    into memory. For OpenWebText (~7.9M examples), this requires significant RAM
    but ensures all data is processed accurately. The split='train[:-100000]'
    excludes the last 100k examples as used in training.
    """
    dataset = datasets.load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        cache_dir=CACHE_DIR,
        streaming=False,  # Load full dataset from cache (not streaming)
        trust_remote_code=True,
    )
    total_examples = len(dataset)
    print(f'Loaded dataset from cache: {total_examples:,} examples')
    print(f'  (Expected: ~7,813,769 examples = 8,013,769 - 100,000)')
    return dataset


def iter_texts(dataset) -> Iterable[str]:
    """Yield texts from cached dataset."""
    for example in dataset:
        yield example['text']


def first_pass(analyzer, dataset) -> Tuple[collections.Counter, collections.Counter, int]:
    """Collect term counts and doc frequencies over first pass."""
    term_counts = collections.Counter()
    doc_freq = collections.Counter()
    total_docs = 0

    # Use tqdm for progress tracking
    dataset_size = len(dataset)
    for text in tqdm(iter_texts(dataset), total=dataset_size, desc='First pass'):
        total_docs += 1
        tokens = list(analyzer(text))
        if not tokens:
            continue
        term_counts.update(tokens)
        doc_freq.update(set(tokens))

    print(f'First pass complete. Total documents: {total_docs:,}')
    return term_counts, doc_freq, total_docs


def compute_df_bounds(total_docs: int) -> Tuple[int, int]:
    if MIN_DF < 1:
        min_df = math.ceil(MIN_DF * total_docs)
    else:
        min_df = int(MIN_DF)
    if MAX_DF <= 1:
        max_df = math.floor(MAX_DF * total_docs)
    else:
        max_df = int(MAX_DF)
    max_df = max(max_df, min_df)
    return min_df, max_df


def print_vocabulary_statistics(term_counts, doc_freq, total_docs):
    """Print statistics to help with parameter selection."""
    min_df, max_df = compute_df_bounds(total_docs)
    all_tokens = len(doc_freq)
    filtered_tokens = sum(1 for df in doc_freq.values() if min_df <= df <= max_df)

    print(f'\nVocabulary Selection Statistics:')
    print(f'  Total unique tokens: {all_tokens:,}')
    print(f'  Tokens after MIN_DF={MIN_DF} and MAX_DF={MAX_DF} filtering: {filtered_tokens:,}')
    print(f'    (MIN_DF threshold: {min_df} docs, MAX_DF threshold: {max_df} docs)')
    print(f'  MAX_FEATURES limit: {MAX_FEATURES:,}')
    print(f'  Final vocabulary size: {min(filtered_tokens, MAX_FEATURES):,}')

    # Show distribution of document frequencies
    df_values = np.array(list(doc_freq.values()))
    print(f'\nDocument Frequency Distribution:')
    print(f'  Min DF: {df_values.min():,}')
    print(f'  25th percentile: {np.percentile(df_values, 25):.0f}')
    print(f'  Median DF: {np.percentile(df_values, 50):.0f}')
    print(f'  75th percentile: {np.percentile(df_values, 75):.0f}')
    print(f'  Max DF: {df_values.max():,}')
    print(f'  Mean DF: {df_values.mean():.1f}')


def select_vocabulary(term_counts, doc_freq, total_docs):
    min_df, max_df = compute_df_bounds(total_docs)
    filtered = [
        (token, term_counts[token])
        for token, df in doc_freq.items()
        if min_df <= df <= max_df
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)
    vocab_tokens = [token for token, _ in filtered[:MAX_FEATURES]]
    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
    print(f'\nSelected vocabulary size: {len(vocab):,} tokens')
    return vocab


def compute_idf(doc_freq: Dict[str, int], vocab: Dict[str, int], total_docs: int):
    idf = np.zeros(len(vocab))
    for token, idx in vocab.items():
        df = doc_freq[token]
        idf[idx] = math.log((1 + total_docs) / (1 + df)) + 1
    return idf


def second_pass(analyzer, vocab, idf, dataset):
    """Compute TF-IDF scores and accumulate mean scores."""
    tfidf_sums = np.zeros(len(vocab))
    total_docs = 0

    # Use tqdm for progress tracking
    dataset_size = len(dataset)
    for text in tqdm(iter_texts(dataset), total=dataset_size, desc='Second pass'):
        total_docs += 1
        token_counts = collections.Counter(
            token for token in analyzer(text) if token in vocab)
        if not token_counts:
            continue
        tfidf_vals = {}
        for token, count in token_counts.items():
            idx = vocab[token]
            tfidf_vals[idx] = count * idf[idx]
        norm = math.sqrt(sum(value * value for value in tfidf_vals.values()))
        if norm == 0:
            continue
        for idx, value in tfidf_vals.items():
            tfidf_sums[idx] += value / norm

    print(f'Second pass complete. Processed {total_docs:,} documents.')
    return tfidf_sums / max(total_docs, 1)


def map_tokens_to_ids(tokens: List[str]) -> dict:
    """Map TF-IDF tokens to GPT-2 tokenizer IDs."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    token_to_id = {}
    for token in tqdm(tokens, desc='Mapping tokens to IDs'):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            token_to_id[token] = int(token_id)
    return token_to_id


def score_sample_global(tfidf_stats_path, sample_text, percentile=90, verbose=False):
    """
    Score sample using pre-computed global TF-IDF scores from JSON.
    This matches the approach used in training (adlm_diffusion.py).
    
    Returns:
        dict with keys: percentile, threshold_value, tokens (list), important_count, total_count
    """
    # Load pre-computed TF-IDF statistics
    with open(tfidf_stats_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    token_scores = stats['token_scores']  # token -> global TF-IDF score
    token_to_id = stats['token_to_id']    # token -> GPT-2 token ID
    percentile_table = stats['percentile_table']
    
    # Get threshold value
    threshold_key = str(percentile)
    if threshold_key not in percentile_table:
        # Find closest percentile
        available_percentiles = sorted([float(k) for k in percentile_table.keys()])
        closest = min(available_percentiles, key=lambda x: abs(x - percentile))
        threshold_value = percentile_table[str(closest)]
        actual_percentile = closest
    else:
        threshold_value = percentile_table[threshold_key]
        actual_percentile = percentile
    
    if verbose:
        print(f'\n=== Global TF-IDF Scoring (using {actual_percentile}th percentile) ===')
        print(f'  Threshold value: {threshold_value:.6f}')
    
    # Tokenize sample text using GPT-2 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    token_ids = tokenizer.encode(sample_text, add_special_tokens=False)
    
    # Create reverse mapping: token_id -> token_name (for TF-IDF lookup)
    # Note: GPT-2 uses subword tokenization, so one word may map to multiple token IDs
    id_to_token = {v: k for k, v in token_to_id.items()}
    
    if verbose:
        print(f'\nSample text token importance (global TF-IDF scores):')
        print(f'  Text: {sample_text[:80]}...')
        print(f'\n  Token ID | Decoded Token      | TF-IDF Token    | Global TF-IDF | Important?')
        print(f'  {"-" * 75}')
    
    tokens_info = []
    important_count = 0
    
    for token_id in token_ids:
        # Get token string from tokenizer (may be subword like "open" or "ai")
        decoded_token = tokenizer.decode([token_id]).strip()
        
        # Try direct lookup: token_id -> TF-IDF token name
        tfidf_token = id_to_token.get(token_id, None)
        
        # If not found, try to match decoded token with TF-IDF vocabulary
        # (handles case where GPT-2 splits words into subwords)
        if not tfidf_token:
            # Normalize decoded token (lowercase, remove spaces)
            normalized = decoded_token.lower().strip()
            if normalized in token_scores:
                tfidf_token = normalized
        
        token_info = {
            'token_id': int(token_id),
            'decoded_token': decoded_token,
            'tfidf_token': tfidf_token if tfidf_token else None,
            'global_tfidf_score': None,
            'is_important': False,
            'in_vocab': False
        }
        
        if tfidf_token and tfidf_token in token_scores:
            global_score = token_scores[tfidf_token]
            is_important = global_score >= threshold_value
            token_info['global_tfidf_score'] = float(global_score)
            token_info['is_important'] = is_important
            token_info['in_vocab'] = True
            
            if is_important:
                important_count += 1
            
            if verbose:
                status = "YES" if is_important else "no"
                print(f'  {token_id:8d} | {decoded_token:<18s} | {tfidf_token:<15s} | {global_score:13.6f} | {status}')
        else:
            if verbose:
                print(f'  {token_id:8d} | {decoded_token:<18s} | {"N/A":<15s} | {"N/A":<13s} | no (not in vocab)')
        
        tokens_info.append(token_info)
    
    if verbose:
        print(f'\n  Summary: {important_count}/{len(token_ids)} tokens marked as important')
    
    return {
        'percentile': float(actual_percentile),
        'threshold_value': float(threshold_value),
        'tokens': tokens_info,
        'important_count': important_count,
        'total_count': len(token_ids),
        'important_ratio': important_count / len(token_ids) if len(token_ids) > 0 else 0.0
    }


def run_validation(tfidf_stats_path, sample_text, percentiles=[85, 90, 95, 99]):
    """
    Run validation with multiple percentile thresholds and save results to JSON.
    Creates a token-by-percentile matrix showing which tokens are important at each threshold.
    """
    print('\n' + '=' * 70)
    print('VALIDATION: Global TF-IDF Scoring (matches training approach)')
    print('=' * 70)
    
    results = {
        'sample_text': sample_text,
        'validation_timestamp': datetime.now().isoformat(),
        'results_by_percentile': {},
        'token_percentile_matrix': {}
    }
    
    # Collect results for each percentile
    for percentile in percentiles:
        print(f'\nTesting {percentile}th percentile...', end='', flush=True)
        result = score_sample_global(tfidf_stats_path, sample_text, percentile=percentile, verbose=False)
        results['results_by_percentile'][str(percentile)] = result
        print(f' {result["important_count"]}/{result["total_count"]} tokens important ({result["important_ratio"]*100:.1f}%)')
    
    # Build token-by-percentile matrix
    # Get all unique tokens from all results
    all_tokens = {}
    for percentile_str, result in results['results_by_percentile'].items():
        for token_info in result['tokens']:
            token_key = f"{token_info['token_id']}_{token_info['decoded_token']}"
            if token_key not in all_tokens:
                all_tokens[token_key] = {
                    'token_id': token_info['token_id'],
                    'decoded_token': token_info['decoded_token'],
                    'tfidf_token': token_info['tfidf_token'],
                    'global_tfidf_score': token_info['global_tfidf_score'],
                    'in_vocab': token_info['in_vocab'],
                    'important_at_percentile': {}
                }
            all_tokens[token_key]['important_at_percentile'][percentile_str] = token_info['is_important']
    
    # Convert to list sorted by token position in the text
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    token_ids = tokenizer.encode(sample_text, add_special_tokens=False)
    token_matrix = []
    for token_id in token_ids:
        decoded_token = tokenizer.decode([token_id]).strip()
        token_key = f"{token_id}_{decoded_token}"
        if token_key in all_tokens:
            token_matrix.append(all_tokens[token_key])
        else:
            # Token not found in any result (shouldn't happen, but handle gracefully)
            token_matrix.append({
                'token_id': token_id,
                'decoded_token': decoded_token,
                'tfidf_token': None,
                'global_tfidf_score': None,
                'in_vocab': False,
                'important_at_percentile': {str(p): False for p in percentiles}
            })
    
    results['token_percentile_matrix'] = {
        'percentiles': [str(p) for p in sorted(percentiles)],
        'tokens': token_matrix
    }
    
    # Save validation results (JSON)
    os.makedirs(os.path.dirname(VALIDATION_OUTPUT_PATH), exist_ok=True)
    with open(VALIDATION_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f'\nSaved validation results to {VALIDATION_OUTPUT_PATH}')
    
    # Save token-by-percentile matrix as text file
    os.makedirs(os.path.dirname(VALIDATION_MATRIX_PATH), exist_ok=True)
    with open(VALIDATION_MATRIX_PATH, 'w', encoding='utf-8') as f:
        f.write('Token-by-Percentile Matrix\n')
        f.write('=' * 60 + '\n')
        f.write(f'Sample text: {sample_text}\n')
        f.write(f'Generated: {datetime.now().isoformat()}\n\n')
        
        percentile_labels = [f'{p}%' for p in sorted(percentiles)]
        header = ' ' * 20 + ' '.join([f'{p:>6}' for p in percentile_labels])
        f.write(header + '\n')
        f.write('-' * len(header) + '\n')
        
        for token_data in token_matrix:
            token_name = token_data['tfidf_token'] if token_data['tfidf_token'] else token_data['decoded_token']
            token_name = token_name[:20].ljust(20)
            
            marks = []
            for p in sorted(percentiles):
                is_important = token_data['important_at_percentile'].get(str(p), False)
                marks.append('✓' if is_important else ' ')
            
            row = token_name + ' ' + ' '.join([f'{m:>6}' for m in marks])
            f.write(row + '\n')
    
    print(f'Saved token-percentile matrix to {VALIDATION_MATRIX_PATH}')
    
    # Print summary comparison table
    print('\nSummary comparison:')
    print('  Percentile | Threshold Value | Important Tokens | Ratio')
    print('  ' + '-' * 60)
    for percentile in sorted([float(p) for p in results['results_by_percentile'].keys()]):
        p_str = str(int(percentile))
        r = results['results_by_percentile'][p_str]
        print(f'  {percentile:>9.0f}% | {r["threshold_value"]:>15.6f} | {r["important_count"]:>15d}/{r["total_count"]:<3d} | {r["important_ratio"]*100:>5.1f}%')
    
    # Print token-by-percentile matrix (user-friendly format)
    print('\nToken-by-Percentile Matrix:')
    percentile_labels = [f'{p}%' for p in sorted(percentiles)]
    
    # Header row
    header = ' ' * 20 + ' '.join([f'{p:>6}' for p in percentile_labels])
    print(header)
    
    # Token rows
    for token_data in token_matrix:
        # Use TF-IDF token name if available, otherwise decoded token
        token_name = token_data['tfidf_token'] if token_data['tfidf_token'] else token_data['decoded_token']
        token_name = token_name[:20].ljust(20)
        
        marks = []
        for p in sorted(percentiles):
            is_important = token_data['important_at_percentile'].get(str(p), False)
            marks.append('✓' if is_important else ' ')
        
        row = token_name + ' ' + ' '.join([f'{m:>6}' for m in marks])
        print(row)
    
    return results


def compute_tfidf(output_path=None, dataset_limit=None):
    """
    Compute TF-IDF scores from dataset and save to JSON.
    
    Args:
        output_path: Path to save TF-IDF stats JSON (default: OUTPUT_PATH)
        dataset_limit: Limit number of documents to process (default: None = all)
    """
    if output_path is None:
        output_path = OUTPUT_PATH
    
    print('Loading dataset from cache...')
    dataset = load_cached_dataset()
    if dataset_limit:
        dataset = dataset.select(range(dataset_limit))
        print(f'Limited to first {dataset_limit:,} documents')

    print('Preparing analyzer...')
    # Note: TfidfVectorizer is used ONLY to get the analyzer function.
    # We do NOT call fit_transform() which would load all data into memory.
    # Instead, we manually implement 2-pass TF-IDF computation to handle
    # large datasets efficiently.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=NGRAM_RANGE,
        token_pattern=r'(?u)\b\w+\b',
        strip_accents='unicode',
    )
    analyzer = vectorizer.build_analyzer()  # Get tokenization function only

    print('\n=== First Pass: Document Statistics ===')
    term_counts, doc_freq, total_docs = first_pass(analyzer, dataset)

    # Print statistics to help with parameter tuning
    print_vocabulary_statistics(term_counts, doc_freq, total_docs)

    vocab = select_vocabulary(term_counts, doc_freq, total_docs)
    idf = compute_idf(doc_freq, vocab, total_docs)

    print('\n=== Second Pass: TF-IDF Accumulation ===')
    mean_scores = second_pass(analyzer, vocab, idf, dataset)
    feature_names = np.array(list(vocab.keys()))
    top_indices = mean_scores.argsort()[::-1]

    print(f'\nTop {TOP_K_PRINT} tokens by mean TF-IDF:')
    for rank, idx in enumerate(top_indices[:TOP_K_PRINT], start=1):
        token = feature_names[idx]
        score = mean_scores[idx]
        print(f'{rank:02d}. {token:<20s} {score:.6f}')

    percentile_values = {
        str(p): float(np.percentile(mean_scores, p))
        for p in PERCENTILE_POINTS
    }
    print(f'\nComputed {len(percentile_values)} percentile thresholds (0.01% steps)')
    print('  Sample thresholds:')
    sample_percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in sample_percentiles:
        if str(p) in percentile_values:
            print(f'    {p:>3}th percentile: {percentile_values[str(p)]:.6f}')

    sorted_indices = np.argsort(mean_scores)[::-1]
    token_scores = {
        feature_names[idx]: float(mean_scores[idx])
        for idx in sorted_indices
    }
    token_to_id = map_tokens_to_ids(list(token_scores.keys()))

    payload = {
        'dataset': DATASET_NAME,
        'split': DATASET_SPLIT,
        'cache_dir': CACHE_DIR,
        'parameters': {
            'max_features': MAX_FEATURES,
            'ngram_range': NGRAM_RANGE,
            'min_df': MIN_DF,
            'max_df': MAX_DF,
        },
        'statistics': {
            'total_documents': total_docs,
            'vocabulary_size': len(vocab),
        },
        'percentile_table': percentile_values,  # All percentiles 0.01-99.99%
        'token_scores': token_scores,
        'token_to_id': token_to_id,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)
    print(f'\nSaved TF-IDF stats to {output_path}')
    
    return output_path


def validate_tfidf(tfidf_stats_path=None, sample_text=None, percentiles=None):
    """
    Validate TF-IDF scores by testing sample text with multiple percentile thresholds.
    This function can be run independently after TF-IDF computation is complete.
    
    Args:
        tfidf_stats_path: Path to TF-IDF stats JSON (default: OUTPUT_PATH)
        sample_text: Sample text to validate (default: SAMPLE)
        percentiles: List of percentiles to test (default: [85, 90, 95, 99])
    """
    if tfidf_stats_path is None:
        tfidf_stats_path = OUTPUT_PATH
    if sample_text is None:
        sample_text = SAMPLE
    if percentiles is None:
        percentiles = [99, 99.2, 99.4, 99.6, 99.8]
    
    # Check if stats file exists
    if not os.path.exists(tfidf_stats_path):
        raise FileNotFoundError(
            f'TF-IDF stats file not found: {tfidf_stats_path}\n'
            f'Please run TF-IDF computation first using compute_tfidf()')
    
    print(f'Loading TF-IDF stats from: {tfidf_stats_path}')
    run_validation(tfidf_stats_path, sample_text, percentiles)


def main():
    """Main entry point. Use --validate-only to skip computation and only validate."""
    import sys
    
    if '--validate-only' in sys.argv or '-v' in sys.argv:
        # Validation only mode
        print('=' * 70)
        print('VALIDATION MODE: Using existing TF-IDF stats')
        print('=' * 70)
        validate_tfidf()
    else:
        # Compute TF-IDF and optionally validate
        compute_tfidf()
        print('\n' + '=' * 70)
        print('TF-IDF computation complete. To validate only, run:')
        print(f'  python {__file__} --validate-only')
        print('=' * 70)


if __name__ == '__main__':
    main()
