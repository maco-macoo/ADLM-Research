"""
データセットの読み込みと操作・確認用スクリプト

データセットの情報を確認したり、先頭のデータを表示したりできます。
"""

import os
import argparse
import sys
from pathlib import Path
import datasets

# dataloader.pyをインポート
sys.path.append(str(Path(__file__).parent.parent))
import dataloader


def load_dataset(dataset_name='openwebtext-train', cache_dir='./datasets'):
    """
    データセットを読み込む（分割データを優先的に使用）
    
    Args:
        dataset_name: データセット名（'openwebtext-train' または 'openwebtext-valid'）
        cache_dir: キャッシュディレクトリ
    """
    print(f"データセット '{dataset_name}' を読み込みます...")
    
    # 保存済みの分割データがあるかチェック
    dataset_save_path = os.path.join(cache_dir, f'{dataset_name}_split')
    
    if os.path.exists(dataset_save_path):
        print(f"保存済みの分割データを読み込みます: {dataset_save_path}")
        try:
            dataset = datasets.load_from_disk(dataset_save_path)
            print(f"✓ 読み込み成功（データ数: {len(dataset):,}件）")
            return dataset
        except Exception as e:
            print(f"✗ 読み込み失敗: {e}")
            print("HuggingFaceからデータをダウンロードします...")
    
    # 保存済みデータがない場合、HuggingFaceから読み込む
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
    
    print(f"✓ 読み込み成功（データ数: {len(dataset):,}件）")
    
    # 分割データを自動保存
    print(f"\n分割データを保存します: {dataset_save_path}")
    os.makedirs(cache_dir, exist_ok=True)
    dataset.save_to_disk(dataset_save_path)
    print(f"✓ 保存完了")
    
    return dataset


def show_dataset_info(dataset, dataset_name):
    """データセットの基本情報を表示"""
    print("\n" + "="*60)
    print(f"データセット情報: {dataset_name}")
    print("="*60)
    
    # データ数
    print(f"データ数: {len(dataset):,}件")
    
    # スライスの説明
    if dataset_name == 'openwebtext-train':
        print("\nスライス: train[:-100000]")
        print("  → 最後の10万件を除外した残りのデータ")
        print("  → 全データ数（約8,013,769件）から最後の10万件を除いた約791万件")
    elif dataset_name == 'openwebtext-valid':
        print("\nスライス: train[-100000:]")
        print("  → 最後の10万件のデータ")
        print("  → 検証用データ")
    
    # カラム情報
    print(f"\nカラム: {dataset.column_names}")
    
    # 特徴量の情報
    print(f"\n特徴量:")
    for feature_name, feature in dataset.features.items():
        print(f"  - {feature_name}: {feature}")
    
    # データサイズの推定（テキストデータの場合）
    if 'text' in dataset.column_names:
        total_chars = sum(len(str(item)) for item in dataset.select(range(min(1000, len(dataset))))['text'])
        avg_chars = total_chars / min(1000, len(dataset))
        print(f"\n推定平均文字数（サンプル1000件から）: {avg_chars:.0f}文字")
    
    print("="*60)


def show_samples(dataset, num_samples=2, start_idx=0):
    """データセットのサンプルを表示"""
    print(f"\n先頭{num_samples}件のサンプル（インデックス {start_idx} から）:")
    print("="*60)
    
    end_idx = min(start_idx + num_samples, len(dataset))
    
    for i in range(start_idx, end_idx):
        example = dataset[i]
        print(f"\n[サンプル {i}]")
        print("-" * 60)
        
        if 'text' in example:
            text = example['text']
            # 長いテキストは切り詰める
            if len(text) > 500:
                print(f"text: {text[:500]}... (全{len(text)}文字)")
            else:
                print(f"text: {text}")
        else:
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 200:
                    print(f"{key}: {value[:200]}... (全{len(value)}文字)")
                else:
                    print(f"{key}: {value}")
    
    print("="*60)


def show_statistics(dataset, num_samples=10000):
    """データセットの統計情報を表示"""
    print(f"\n統計情報（サンプル数: {min(num_samples, len(dataset)):,}件）:")
    print("="*60)
    
    sample_size = min(num_samples, len(dataset))
    sample_dataset = dataset.select(range(sample_size))
    
    if 'text' in dataset.column_names:
        texts = sample_dataset['text']
        
        # 文字数統計
        lengths = [len(str(text)) for text in texts]
        print(f"\n文字数統計:")
        print(f"  平均: {sum(lengths) / len(lengths):.0f}文字")
        print(f"  最小: {min(lengths):,}文字")
        print(f"  最大: {max(lengths):,}文字")
        print(f"  中央値: {sorted(lengths)[len(lengths)//2]:,}文字")
        
        # 単語数統計（簡易版）
        word_counts = [len(str(text).split()) for text in texts]
        print(f"\n単語数統計（簡易）:")
        print(f"  平均: {sum(word_counts) / len(word_counts):.0f}単語")
        print(f"  最小: {min(word_counts):,}単語")
        print(f"  最大: {max(word_counts):,}単語")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='データセットの読み込みと操作・確認')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='openwebtext-train',
        choices=['openwebtext-train', 'openwebtext-valid'],
        help='データセット名'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='./datasets',
        help='キャッシュディレクトリ'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=2,
        help='表示するサンプル数'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='表示開始インデックス'
    )
    parser.add_argument(
        '--statistics',
        action='store_true',
        help='統計情報を表示'
    )
    parser.add_argument(
        '--statistics_samples',
        type=int,
        default=10000,
        help='統計情報計算に使用するサンプル数'
    )
    
    args = parser.parse_args()
    
    # データセットを読み込む（分割データを優先的に使用）
    dataset = load_dataset(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir
    )
    
    # データセット情報を表示
    show_dataset_info(dataset, args.dataset_name)
    
    # サンプルを表示
    show_samples(dataset, num_samples=args.num_samples, start_idx=args.start_idx)
    
    # 統計情報を表示（オプション）
    if args.statistics:
        show_statistics(dataset, num_samples=args.statistics_samples)


if __name__ == '__main__':
    main()

