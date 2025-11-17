"""語彙を確認するスクリプト"""

import pickle
import os
from pathlib import Path

BASE_DIR = './tfidf_cache'
dataset_name = 'openwebtext-valid'

# ベクトライザーを読み込む
vectorizer_path = Path(BASE_DIR) / f'tfidf_vectorizer_{dataset_name}.pkl'
stats_path = Path(BASE_DIR) / f'tfidf_stats_{dataset_name}.pkl'

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

with open(stats_path, 'rb') as f:
    stats = pickle.load(f)

vocabulary = stats['vocabulary']
feature_names = stats['feature_names']

print(f"語彙数: {len(vocabulary)}")
print(f"特徴数: {len(feature_names)}")

# 「a」が含まれているか確認
test_words = ['a', 'the', 'is', 'machine', 'learning', 'for']

print("\n語彙チェック:")
print("-" * 60)
for word in test_words:
    if word in vocabulary:
        word_id = vocabulary[word]
        word_avg_tfidf = stats['mean_tfidf_per_word'][word_id]
        print(f"  '{word}': ✅ 語彙に含まれています (ID: {word_id}, 平均TF-IDF: {word_avg_tfidf:.6f})")
    else:
        print(f"  '{word}': ❌ 語彙に含まれていません")

# 最初の100単語を表示
print("\n最初の100単語:")
print("-" * 60)
for i, word in enumerate(feature_names[:100]):
    print(f"{i:3d}: {word}")

# ベクトライザーの設定を確認
print("\nベクトライザーの設定:")
print(f"  min_df: {vectorizer.min_df}")
print(f"  max_df: {vectorizer.max_df}")
print(f"  max_features: {vectorizer.max_features}")

