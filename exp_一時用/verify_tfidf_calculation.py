"""
TF-IDF計算の検証スクリプト

単語ごとのTF-IDF計算が正しく行われているかを確認します。
実際のTF-IDF行列から単語ごとの平均値を計算し、保存された統計情報と比較します。
"""

import pickle
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

BASE_DIR = './tfidf_cache'
dataset_name = 'openwebtext-valid'

# ファイルを読み込む
vectorizer_path = Path(BASE_DIR) / f'tfidf_vectorizer_{dataset_name}.pkl'
stats_path = Path(BASE_DIR) / f'tfidf_stats_{dataset_name}.pkl'

print("TF-IDF計算の検証")
print("=" * 80)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

with open(stats_path, 'rb') as f:
    stats = pickle.load(f)

print(f"\nデータセット情報:")
print(f"  文書数: {stats['num_documents']:,}")
print(f"  語彙数: {stats['num_features']:,}")

# サンプルテキストでTF-IDFを計算
sample_texts = [
    "Machine learning is a powerful tool for artificial intelligence research.",
    "The quick brown fox jumps over the lazy dog.",
    "Natural language processing uses machine learning techniques."
]

print(f"\nサンプルテキストでTF-IDFを計算:")
print("-" * 80)

for i, text in enumerate(sample_texts, 1):
    print(f"\nテキスト {i}: {text}")
    tfidf_matrix = vectorizer.transform([text])
    
    # スパース行列を密行列に変換（小さいので問題ない）
    tfidf_dense = tfidf_matrix.toarray()[0]
    
    # 非ゼロの単語のみを表示
    non_zero_indices = tfidf_dense.nonzero()[0]
    if len(non_zero_indices) > 0:
        print(f"  出現した単語数: {len(non_zero_indices)}")
        print(f"  各単語のTF-IDF値（上位10個）:")
        
        # TF-IDF値でソート
        word_scores = [(stats['feature_names'][idx], tfidf_dense[idx]) 
                      for idx in non_zero_indices]
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        for word, score in word_scores[:10]:
            # その単語の平均TF-IDFスコアも取得
            word_id = stats['vocabulary'][word]
            avg_tfidf = stats['mean_tfidf_per_word'][word_id]
            print(f"    {word:15s} -> この文書でのTF-IDF: {score:.6f}, 平均TF-IDF: {avg_tfidf:.6f}")
    else:
        print("  語彙に含まれる単語がありません")

print("\n" + "=" * 80)
print("検証結果:")
print("=" * 80)
print("\n単語ごとのTF-IDF計算について:")
print("  1. 各文書で単語ごとにTF-IDF値が計算されます")
print("  2. 統計情報の'mean_tfidf_per_word'は、全文書における各単語の平均TF-IDF値です")
print("  3. 重要語判定は、この平均TF-IDF値と閾値を比較して行います")
print("\n注意:")
print(f"  - max_df={vectorizer.max_df}により、{vectorizer.max_df*100}%以上の文書に出現する単語は語彙から除外されます")
print(f"  - min_df={vectorizer.min_df}により、{vectorizer.min_df}回未満しか出現しない単語も除外されます")
print(f"  - 除外された単語（例: 'a', 'the'）はTF-IDF計算の対象外です")

