"""
TF-IDF計算スクリプト

訓練データ（openwebtext-train）全体に対してTF-IDFを計算し、
その統計情報を保存します。このTF-IDFモデルは訓練データと検証データの
両方に適用できます。
"""

import os
import pickle
import argparse
from pathlib import Path
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
import datasets
from tqdm import tqdm
import sys

# dataloader.pyをインポート
sys.path.append(str(Path(__file__).parent.parent))
import dataloader


def compute_tfidf_from_dataset(
    dataset_name: str = 'openwebtext-train',
    cache_dir: str = '/home/smakoto/.cache/huggingface/hub',
    output_dir: str = './tfidf_cache',
    max_samples: int = None,
    max_features: int = 50000,
    min_df: int = 2,
    max_df: float = 0.95,
    text_column: str = 'text'
):
    """
    データセットからTF-IDFを計算する（分割データを優先的に使用）
    
    Args:
        dataset_name: データセット名（'openwebtext-train' または 'openwebtext-valid'）
        cache_dir: データセットのキャッシュディレクトリ（分割データの保存先）
        output_dir: TF-IDFモデルと統計情報の保存先
        max_samples: 計算に使用する最大サンプル数（Noneの場合は全データ）
        max_features: TF-IDFベクトルの最大特徴数
        min_df: 最小文書頻度（この値未満の単語は除外）
        max_df: 最大文書頻度（この割合以上の文書に出現する単語は除外）
        text_column: テキストが含まれるカラム名
    """
    print(f"データセット '{dataset_name}' からTF-IDFを計算します...")
    
    # 分割データの保存先パス
    dataset_save_path = os.path.join(cache_dir, f'{dataset_name}_split')
    
    # 保存済みの分割データがあるかチェック

    print(f"dataset_save_path: {dataset_save_path}")
    print(f"os.path.exists(dataset_save_path): {os.path.exists(dataset_save_path)}")
    exit()

    if os.path.exists(dataset_save_path):
        print(f"保存済みの分割データを読み込みます: {dataset_save_path}")
        try:
            data = datasets.load_from_disk(dataset_save_path)
            print(f"✓ 読み込み成功（データ数: {len(data):,}件）")
        except Exception as e:
            print(f"✗ 読み込み失敗: {e}")
            print("HuggingFaceからデータをダウンロードします...")
            data = None
    else:
        data = None
    
    # 保存済みデータがない場合、HuggingFaceから読み込む
    if data is None:
        print("HuggingFaceからデータをダウンロードします...")
        
        if dataset_name == 'openwebtext-train':
            dataset = datasets.load_dataset(
                'openwebtext',
                split='train[:-100000]',
                cache_dir=cache_dir,
                streaming=False)
        elif dataset_name == 'openwebtext-valid':
            dataset = datasets.load_dataset(
                'openwebtext',
                split='train[-100000:]',
                cache_dir=cache_dir,
                streaming=False)
        else:
            raise ValueError(f"サポートされていないデータセット: {dataset_name}")
        
        data = dataset
        print(f"✓ 読み込み成功（データ数: {len(data):,}件）")
        
        # 分割データを自動保存
        print(f"\n分割データを保存します: {dataset_save_path}")
        os.makedirs(cache_dir, exist_ok=True)
        data.save_to_disk(dataset_save_path)
        print(f"✓ 保存完了")
    
    print(f"データセットサイズ: {len(data):,}件")
    
    # テキストを収集
    print("テキストデータを収集中...")
    texts = []
    
    # サンプル数を制限（デバッグ用）
    if max_samples is not None:
        data = data.select(range(min(max_samples, len(data))))
        print(f"サンプル数を{len(data)}に制限しました（デバッグ用）")
    
    # dataloader.pyの412-418行と同じロジックでテキストカラムを決定
    # openwebtext-trainの場合は'text'カラムを使用
    for example in tqdm(data, desc="テキスト収集"):
        text = example.get('text', '')
        
        if text and isinstance(text, str) and len(text.strip()) > 0:
            texts.append(text)
    
    print(f"収集したテキスト数: {len(texts)}")
    
    if len(texts) == 0:
        raise ValueError("テキストデータが見つかりませんでした")
    
    # TF-IDFを計算
    print("TF-IDFを計算中...")
    print(f"  処理するテキスト数: {len(texts):,}件")
    
    # メモリ効率とリソースリークを防ぐため、バッチ処理でfit_transformを実行
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        lowercase=True,
        token_pattern=r'\b\w+\b',  # 単語境界でトークン化
        ngram_range=(1, 1),  # 単語単位（必要に応じて(1,2)などに変更可能）
    )
    
    # 大量データの場合はバッチ処理でfit_transformを実行
    # ただし、fitは全データを見る必要があるため、まずfitを実行
    print("  語彙を構築中（fit）...")
    try:
        vectorizer.fit(texts)
        print(f"  ✓ 語彙構築完了（語彙数: {len(vectorizer.vocabulary_):,}）")
    except MemoryError as e:
        print(f"  ✗ メモリ不足エラー: {e}")
        print("  より小さいmax_featuresまたはmax_samplesを試してください")
        raise
    
    # transformをバッチ処理で実行（メモリ効率を改善）
    print("  TF-IDF行列を計算中（transform）...")
    batch_size = 100000  # バッチサイズ
    tfidf_matrices = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="  バッチ処理"):
        batch_texts = texts[i:i+batch_size]
        try:
            batch_matrix = vectorizer.transform(batch_texts)
            tfidf_matrices.append(batch_matrix)
        except MemoryError as e:
            print(f"  ✗ メモリ不足エラー（バッチ {i//batch_size + 1}）: {e}")
            print("  より小さいbatch_sizeまたはmax_featuresを試してください")
            raise
    
    # バッチ結果を結合
    print("  バッチ結果を結合中...")
    tfidf_matrix = vstack(tfidf_matrices)
    
    print(f"TF-IDF行列の形状: {tfidf_matrix.shape}")
    print(f"語彙数: {len(vectorizer.vocabulary_)}")
    
    # 統計情報を計算
    print("統計情報を計算中...")
    # 各文書の平均TF-IDFスコア
    mean_tfidf_per_doc = np.array(tfidf_matrix.mean(axis=1)).flatten()
    # 各単語の平均TF-IDFスコア
    mean_tfidf_per_word = np.array(tfidf_matrix.mean(axis=0)).flatten()
    
    # 結果を保存
    os.makedirs(output_dir, exist_ok=True)
    
    # TF-IDFモデル（vectorizer）を保存
    vectorizer_path = os.path.join(output_dir, f'tfidf_vectorizer_{dataset_name}.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"TF-IDFモデルを保存: {vectorizer_path}")
    
    # 統計情報を保存
    stats = {
        'mean_tfidf_per_doc': mean_tfidf_per_doc,
        'mean_tfidf_per_word': mean_tfidf_per_word,
        'vocabulary': vectorizer.vocabulary_,
        'feature_names': vectorizer.get_feature_names_out().tolist(),
        'num_documents': len(texts),
        'num_features': len(vectorizer.vocabulary_),
        'dataset_name': dataset_name,
    }
    
    stats_path = os.path.join(output_dir, f'tfidf_stats_{dataset_name}.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"統計情報を保存: {stats_path}")
    
    # 閾値の候補を計算（例：平均値、中央値、パーセンタイル）
    # 文書ごとの平均TF-IDFスコアの閾値
    doc_thresholds = {
        'mean': float(np.mean(mean_tfidf_per_doc)),
        'median': float(np.median(mean_tfidf_per_doc)),
        'p25': float(np.percentile(mean_tfidf_per_doc, 25)),
        'p75': float(np.percentile(mean_tfidf_per_doc, 75)),
        'p90': float(np.percentile(mean_tfidf_per_doc, 90)),
        'p95': float(np.percentile(mean_tfidf_per_doc, 95)),
    }
    
    # 単語ごとの平均TF-IDFスコアの閾値（重要語判定用）
    word_thresholds = {
        'mean': float(np.mean(mean_tfidf_per_word)),
        'median': float(np.median(mean_tfidf_per_word)),
        'p25': float(np.percentile(mean_tfidf_per_word, 25)),
        'p75': float(np.percentile(mean_tfidf_per_word, 75)),
        'p90': float(np.percentile(mean_tfidf_per_word, 90)),
        'p95': float(np.percentile(mean_tfidf_per_word, 95)),
    }
    
    # 後方互換性のため、thresholdsにもdoc_thresholdsを含める
    thresholds = {
        'doc': doc_thresholds,
        'word': word_thresholds,
    }
    
    print("\n=== TF-IDF統計サマリー ===")
    print(f"文書数: {len(texts)}")
    print(f"語彙数: {len(vectorizer.vocabulary_)}")
    print(f"\n文書ごとの平均TF-IDFスコア:")
    print(f"  平均: {doc_thresholds['mean']:.4f}")
    print(f"  中央値: {doc_thresholds['median']:.4f}")
    print(f"  25パーセンタイル: {doc_thresholds['p25']:.4f}")
    print(f"  75パーセンタイル: {doc_thresholds['p75']:.4f}")
    print(f"  90パーセンタイル: {doc_thresholds['p90']:.4f}")
    print(f"  95パーセンタイル: {doc_thresholds['p95']:.4f}")
    print(f"\n単語ごとの平均TF-IDFスコア（重要語判定用）:")
    print(f"  平均: {word_thresholds['mean']:.4f}")
    print(f"  中央値: {word_thresholds['median']:.4f}")
    print(f"  25パーセンタイル: {word_thresholds['p25']:.4f}")
    print(f"  75パーセンタイル: {word_thresholds['p75']:.4f}")
    print(f"  90パーセンタイル: {word_thresholds['p90']:.4f}")
    print(f"  95パーセンタイル: {word_thresholds['p95']:.4f}")
    
    # 閾値情報も保存
    thresholds_path = os.path.join(output_dir, f'tfidf_thresholds_{dataset_name}.pkl')
    with open(thresholds_path, 'wb') as f:
        pickle.dump(thresholds, f)
    print(f"\n閾値候補を保存: {thresholds_path}")
    
    return vectorizer, stats, thresholds


def apply_tfidf_to_texts(
    texts: List[str],
    vectorizer_path: str,
    output_path: str = None
):
    """
    保存されたTF-IDFモデルを使って、新しいテキストにTF-IDFを適用する
    
    Args:
        texts: テキストのリスト
        vectorizer_path: 保存されたTF-IDFモデルのパス
        output_path: 結果を保存するパス（オプション）
    """
    # モデルをロード
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # TF-IDFを適用
    tfidf_matrix = vectorizer.transform(texts)
    mean_tfidf_scores = np.array(tfidf_matrix.mean(axis=1)).flatten()
    
    if output_path:
        results = {
            'tfidf_scores': mean_tfidf_scores,
            'tfidf_matrix': tfidf_matrix,
        }
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"結果を保存: {output_path}")
    
    return mean_tfidf_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TF-IDF計算スクリプト')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='openwebtext-train',
        help='データセット名（openwebtext-trainを推奨）'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='/home/smakoto/.cache/huggingface/hub',
        help='データセットのキャッシュディレクトリ（実験ディレクトリに保存されます。'
             'HuggingFaceのデフォルトキャッシュ~/.cache/huggingface/は使用しません）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='tfidf_cache',
        help='TF-IDFモデルと統計情報の保存先'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='計算に使用する最大サンプル数（Noneの場合は全データ）'
    )
    parser.add_argument(
        '--max_features',
        type=int,
        default=50000,
        help='TF-IDFベクトルの最大特徴数'
    )
    parser.add_argument(
        '--min_df',
        type=int,
        default=2,
        help='最小文書頻度'
    )
    parser.add_argument(
        '--max_df',
        type=float,
        default=0.95,
        help='最大文書頻度（割合）'
    )
    
    args = parser.parse_args()
    
    compute_tfidf_from_dataset(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
    )

