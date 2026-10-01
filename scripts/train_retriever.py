#!/usr/bin/env python3
"""
Train Sentence-BERT retriever for RAG
Designed for Colab with GPU

Usage in Colab:
1. Upload project folder to Colab
2. Install dependencies: !pip install -r requirements.txt
3. Run: !python scripts/train_retriever.py
"""

import json
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import RetrieverTrainer


def load_data_splits(data_dir: Path = Path('data/splits')):
    """Load train/val/test splits"""
    print("📖 Loading data splits...")
    
    with open(data_dir / 'train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(data_dir / 'val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    with open(data_dir / 'test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Convert to DataFrames
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)
    
    print(f"   Train: {len(train_df)} examples")
    print(f"   Val: {len(val_df)} examples")
    print(f"   Test: {len(test_df)} examples")
    
    return train_df, val_df, test_df


def main():
    print("="*60)
    print("🚀 Training Sentence-BERT Retriever")
    print("="*60 + "\n")
    
    # Paths
    data_dir = Path('data/splits')
    output_dir = Path('models/retriever-finetuned')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_df, val_df, test_df = load_data_splits(data_dir)
    
    # Verify required columns
    if 'question' not in train_df.columns or 'answer' not in train_df.columns:
        print("❌ Error: train.json must have 'question' and 'answer' columns")
        print(f"   Available columns: {train_df.columns.tolist()}")
        return
    
    # Initialize trainer
    print("\n🔧 Initializing RetrieverTrainer...")
    trainer = RetrieverTrainer(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=None  # Auto-detect (will use CUDA if available)
    )
    
    # Prepare training data with hard negatives
    print("\n📊 Preparing training data...")
    train_examples = trainer.prepare_training_data(
        train_df,
        num_hard_negatives=3  # Number of hard negatives per question
    )
    
    print(f"   Total training examples: {len(train_examples)}")
    
    # Train
    print("\n🏋️  Starting training...")
    trainer.train(
        train_examples,
        output_path=str(output_dir),
        epochs=5,  # Adjust based on your needs
        batch_size=16,  # Adjust based on GPU memory
        warmup_steps=100
    )
    
    # Evaluate on validation set
    print("\n📈 Evaluating on validation set...")
    metrics = trainer.evaluate(
        val_df,
        k_values=[1, 3, 5]
    )
    
    # Print summary
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"\n📊 Validation Metrics:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    print(f"\n💾 Model saved to: {output_dir}")
    print("\n📝 Next steps:")
    print("   1. Download the model folder from Colab")
    print("   2. Use it to index documents for retrieval")
    print("   3. Train the generator with retrieved context")


if __name__ == '__main__':
    main()

