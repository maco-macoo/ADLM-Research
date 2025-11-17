import pickle
import os

BASE_DIR = './tfidf_cache'

with open(os.path.join(BASE_DIR, 'tfidf_vectorizer_openwebtext-valid.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

# 新しいテキストに適用
new_texts = ["This is a sample text", "Another example"]
tfidf_scores = vectorizer.transform(new_texts)
print(tfidf_scores)

# 読み込み例
with open(os.path.join(BASE_DIR, 'tfidf_stats_openwebtext-valid.pkl'), 'rb') as f:
    stats = pickle.load(f)

# 使用例
print(f"文書数: {stats['num_documents']}")  # 例: 7913769
print(f"語彙数: {stats['num_features']}")   # 例: 50000
print(f"最初の10単語: {stats['feature_names'][2000:2500]}")
# 例: ['the', 'to', 'of', 'and', 'a', 'in', 'is', 'it', 'you', 'that']

# 各文書の平均TF-IDFスコア（7913769個の値）
doc_scores = stats['mean_tfidf_per_doc']
print(f"最初の文書の平均TF-IDF: {doc_scores[0]}")