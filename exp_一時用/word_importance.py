"""
単語の重要度判定スクリプト

TF-IDFモデルを使って、英文の各単語が重要語かどうかを判定します。
"""

import pickle
import os
import re
import argparse
from pathlib import Path
import numpy as np


def tokenize_text(text):
    """
    テキストを単語に分割（TF-IDFベクトライザーと同じ方法）
    
    Args:
        text: 入力テキスト
    
    Returns:
        単語のリスト
    """
    # TF-IDFベクトライザーのtoken_patternに合わせる: r'\b\w+\b'
    words = re.findall(r'\b\w+\b', text.lower())
    return words


def check_word_importance(
    text: str,
    vectorizer_path: str,
    stats_path: str,
    thresholds_path: str,
    threshold_type: str = 'p75'
):
    """
    テキスト内の各単語が重要語かどうかを判定
    
    Args:
        text: 判定するテキスト
        vectorizer_path: TF-IDFベクトライザーのパス
        stats_path: 統計情報のパス
        thresholds_path: 閾値情報のパス
        threshold_type: 使用する閾値のタイプ ('mean', 'median', 'p25', 'p75', 'p90', 'p95')
    
    Returns:
        各単語の重要度情報を含む辞書のリスト
    """
    # モデルと統計情報を読み込む
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    with open(thresholds_path, 'rb') as f:
        thresholds = pickle.load(f)
    
    # ベクトライザーの設定を取得（除外理由を説明するため）
    max_df = vectorizer.max_df
    min_df = vectorizer.min_df
    
    # 単語の重要度判定用の閾値を取得
    if isinstance(thresholds, dict) and 'word' in thresholds:
        word_thresholds = thresholds['word']
    else:
        # 後方互換性: 古い形式の場合は文書の閾値を使用
        word_thresholds = thresholds
    
    threshold_value = word_thresholds.get(threshold_type)
    if threshold_value is None:
        raise ValueError(f"閾値タイプ '{threshold_type}' が見つかりません。利用可能: {list(word_thresholds.keys())}")
    
    # テキストを単語に分割
    words = tokenize_text(text)
    
    # 各単語の重要度を判定
    results = []
    vocabulary = stats['vocabulary']
    mean_tfidf_per_word = stats['mean_tfidf_per_word']
    feature_names = stats['feature_names']
    
    for word in words:
        # 語彙に含まれているかチェック
        if word in vocabulary:
            word_id = vocabulary[word]
            word_avg_tfidf = mean_tfidf_per_word[word_id]
            is_important = word_avg_tfidf >= threshold_value
            
            results.append({
                'word': word,
                'avg_tfidf': float(word_avg_tfidf),
                'is_important': is_important,
                'threshold': threshold_value,
                'in_vocabulary': True,
                'exclusion_reason': None
            })
        else:
            # 語彙に含まれていない単語
            # max_dfまたはmin_dfにより除外された可能性が高い
            exclusion_reason = f"語彙から除外（max_df={max_df}またはmin_df={min_df}により頻出/稀出語として除外された可能性）"
            results.append({
                'word': word,
                'avg_tfidf': None,
                'is_important': False,
                'threshold': threshold_value,
                'in_vocabulary': False,
                'exclusion_reason': exclusion_reason
            })
    
    return results


def print_results(text: str, results: list, threshold_type: str = 'p75'):
    """
    結果を分かりやすく表示
    
    Args:
        text: 元のテキスト
        results: check_word_importanceの結果
        threshold_type: 使用した閾値のタイプ
    """
    print("=" * 80)
    print("単語の重要度判定結果")
    print("=" * 80)
    print(f"\n入力テキスト: {text}\n")
    if results:
        print(f"使用閾値: {results[0]['threshold']:.6f} (閾値タイプ: {threshold_type})")
    print("\n各単語の判定結果:")
    print("-" * 80)
    
    important_words = []
    unimportant_words = []
    unknown_words = []
    
    for result in results:
        word = result['word']
        if not result['in_vocabulary']:
            reason = result.get('exclusion_reason', '語彙に含まれていません')
            status = f"❓ 除外語 ({reason})"
            unknown_words.append(word)
        elif result['is_important']:
            status = f"✅ 重要語 (平均TF-IDF: {result['avg_tfidf']:.6f})"
            important_words.append(word)
        else:
            status = f"⚪ 非重要語 (平均TF-IDF: {result['avg_tfidf']:.6f})"
            unimportant_words.append(word)
        
        print(f"  {word:20s} -> {status}")
    
    print("\n" + "=" * 80)
    print("サマリー:")
    print(f"  重要語: {len(important_words)}個 - {', '.join(important_words) if important_words else 'なし'}")
    print(f"  非重要語: {len(unimportant_words)}個 - {', '.join(unimportant_words) if unimportant_words else 'なし'}")
    print(f"  未知語: {len(unknown_words)}個 - {', '.join(unknown_words) if unknown_words else 'なし'}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='単語の重要度判定スクリプト')
    parser.add_argument(
        '--text',
        type=str,
        default='Machine learning is a powerful tool for artificial intelligence research.',
        help='判定するテキスト'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='openwebtext-train',
        help='使用するデータセット名（ファイル名の一部）'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='./tfidf_cache',
        help='TF-IDFキャッシュディレクトリ'
    )
    parser.add_argument(
        '--threshold',
        type=str,
        default='p75',
        choices=['mean', 'median', 'p25', 'p75', 'p90', 'p95'],
        help='使用する閾値のタイプ'
    )
    
    args = parser.parse_args()
    
    # ファイルパスを構築
    base_dir = Path(args.cache_dir)
    vectorizer_path = base_dir / f'tfidf_vectorizer_{args.dataset_name}.pkl'
    stats_path = base_dir / f'tfidf_stats_{args.dataset_name}.pkl'
    thresholds_path = base_dir / f'tfidf_thresholds_{args.dataset_name}.pkl'
    
    # ファイルの存在確認
    for path in [vectorizer_path, stats_path, thresholds_path]:
        if not path.exists():
            print(f"エラー: ファイルが見つかりません: {path}")
            print(f"先に 'uv run exp/tf-idf.py --dataset_name {args.dataset_name}' を実行してください。")
            exit(1)
    
    threshold_type = args.threshold
    
    # 重要度判定を実行
    results = check_word_importance(
        text=args.text,
        vectorizer_path=str(vectorizer_path),
        stats_path=str(stats_path),
        thresholds_path=str(thresholds_path),
        threshold_type=threshold_type
    )
    
    # 結果を表示
    print_results(args.text, results, threshold_type)

