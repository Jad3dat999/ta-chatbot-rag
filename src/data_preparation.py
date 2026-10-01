"""
Data preparation module for TA Chatbot
Handles data loading, augmentation, and splitting
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
import random
from pathlib import Path


class DataPreparer:
    """Handles data loading, augmentation, and splitting for the TA chatbot"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.splits_dir = self.data_dir / "splits"
        
        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.processed_dir, self.splits_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_piazza_data(self, filepath: str) -> pd.DataFrame:
        """
        Load Piazza Q&A data from JSON or CSV
        
        Expected format:
        {
            "question": "How do I...",
            "answer": "You can...",
            "tags": ["homework", "python"],
            "url": "piazza.com/post/123"
        }
        """
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif filepath.endswith('.csv'):
            data = pd.read_csv(filepath).to_dict('records')
        else:
            raise ValueError("Unsupported file format. Use JSON or CSV")
        
        return pd.DataFrame(data)
    
    def load_lecture_slides(self, filepath: str) -> pd.DataFrame:
        """
        Load lecture slide content
        
        Expected format:
        {
            "slide_number": 15,
            "lecture": "Lecture 3",
            "content": "Backpropagation is...",
            "source": "lecture_3.pdf"
        }
        """
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif filepath.endswith('.csv'):
            data = pd.read_csv(filepath).to_dict('records')
        else:
            raise ValueError("Unsupported file format. Use JSON or CSV")
        
        return pd.DataFrame(data)
    
    def load_assignment_docs(self, filepath: str) -> pd.DataFrame:
        """
        Load assignment documentation
        
        Expected format:
        {
            "assignment": "HW1",
            "section": "Instructions",
            "content": "Implement...",
            "source": "hw1_instructions.pdf"
        }
        """
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif filepath.endswith('.csv'):
            data = pd.read_csv(filepath).to_dict('records')
        else:
            raise ValueError("Unsupported file format. Use JSON or CSV")
        
        return pd.DataFrame(data)
    
    def augment_question(self, question: str) -> List[str]:
        """
        Augment a question by creating variations
        
        Techniques:
        - Paraphrasing (simple template-based)
        - Adding question markers
        - Reformatting
        """
        augmented = [question]  # Original question
        
        # Add variations
        variations = [
            f"How do I {question.lower()}",
            f"Can you explain {question.lower()}",
            f"What is the best way to {question.lower()}",
            f"I need help with {question.lower()}",
        ]
        
        # Add only if they make sense (basic check)
        for var in variations:
            if len(var.split()) > 3 and len(var) < 200:
                augmented.append(var)
        
        return augmented[:3]  # Return max 3 variations
    
    def augment_data(self, df: pd.DataFrame, augmentation_factor: int = 2) -> pd.DataFrame:
        """
        Augment the dataset by creating variations of questions
        
        Args:
            df: DataFrame with 'question' and 'answer' columns
            augmentation_factor: How many augmented versions per original
        
        Returns:
            Augmented DataFrame
        """
        augmented_data = []
        
        for _, row in df.iterrows():
            # Add original
            augmented_data.append(row.to_dict())
            
            # Add augmented versions
            question_variations = self.augment_question(row['question'])
            for var in question_variations[1:augmentation_factor+1]:
                aug_row = row.to_dict()
                aug_row['question'] = var
                aug_row['is_augmented'] = True
                augmented_data.append(aug_row)
        
        return pd.DataFrame(augmented_data)
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Important: Test set should only contain original (non-augmented) data
        """
        # First, separate original and augmented data
        original_data = df[~df.get('is_augmented', False)].copy()
        augmented_data = df[df.get('is_augmented', False)].copy()
        
        # Split original data for val and test
        train_orig, temp = train_test_split(
            original_data, 
            test_size=(val_ratio + test_ratio),
            random_state=random_seed
        )
        
        val_data, test_data = train_test_split(
            temp,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_seed
        )
        
        # Combine training original with all augmented data
        train_data = pd.concat([train_orig, augmented_data], ignore_index=True)
        
        return train_data, val_data, test_data
    
    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        prefix: str = "data"
    ):
        """Save train/val/test splits to disk"""
        train_df.to_json(
            self.splits_dir / f"{prefix}_train.json",
            orient='records',
            indent=2
        )
        val_df.to_json(
            self.splits_dir / f"{prefix}_val.json",
            orient='records',
            indent=2
        )
        test_df.to_json(
            self.splits_dir / f"{prefix}_test.json",
            orient='records',
            indent=2
        )
        
        print(f"Saved splits to {self.splits_dir}/")
        print(f"  Train: {len(train_df)} examples")
        print(f"  Val: {len(val_df)} examples")
        print(f"  Test: {len(test_df)} examples")
    
    def load_splits(self, prefix: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train/val/test splits from disk"""
        train_df = pd.read_json(self.splits_dir / f"{prefix}_train.json")
        val_df = pd.read_json(self.splits_dir / f"{prefix}_val.json")
        test_df = pd.read_json(self.splits_dir / f"{prefix}_test.json")
        
        return train_df, val_df, test_df
    
    def create_sample_data(self, num_samples: int = 122):
        """
        Create sample data for testing purposes
        This helps you get started before you have real Piazza data
        """
        sample_questions = [
            "How do I implement backpropagation?",
            "What is the difference between SGD and Adam optimizer?",
            "How do I handle overfitting in my neural network?",
            "Can you explain the concept of attention mechanisms?",
            "What is the best way to initialize weights?",
            "How do I choose the right learning rate?",
            "What is batch normalization and why is it useful?",
            "How do I implement dropout in PyTorch?",
            "What is the difference between CNN and RNN?",
            "How do I debug gradient vanishing problems?",
        ]
        
        sample_answers = [
            "Backpropagation is computed using the chain rule...",
            "SGD updates weights using a single batch, while Adam uses adaptive learning rates...",
            "To handle overfitting, try: 1) Add dropout, 2) Use L2 regularization, 3) Get more data...",
            "Attention mechanisms allow the model to focus on relevant parts of the input...",
            "Use Xavier/He initialization depending on your activation function...",
            "Start with 1e-3 and use learning rate scheduling...",
            "Batch normalization normalizes activations to improve training stability...",
            "Use nn.Dropout(p=0.5) in PyTorch...",
            "CNNs are for spatial data, RNNs are for sequential data...",
            "Try: 1) Use ReLU activations, 2) Use skip connections, 3) Check weight initialization...",
        ]
        
        data = []
        for i in range(num_samples):
            q_idx = i % len(sample_questions)
            data.append({
                'question': sample_questions[q_idx],
                'answer': sample_answers[q_idx],
                'source': f'piazza_post_{i}',
                'tags': ['deep-learning', 'homework'],
                'is_augmented': False
            })
        
        df = pd.DataFrame(data)
        df.to_json(
            self.raw_dir / "sample_piazza_data.json",
            orient='records',
            indent=2
        )
        print(f"Created {num_samples} sample Q&A pairs at {self.raw_dir}/sample_piazza_data.json")
        return df


if __name__ == "__main__":
    # Example usage
    preparer = DataPreparer()
    
    # Create sample data
    print("Creating sample data...")
    sample_df = preparer.create_sample_data(122)
    
    # Augment data
    print("\nAugmenting data...")
    augmented_df = preparer.augment_data(sample_df, augmentation_factor=2)
    
    # Split data
    print("\nSplitting data...")
    train_df, val_df, test_df = preparer.split_data(augmented_df)
    
    # Save splits
    print("\nSaving splits...")
    preparer.save_splits(train_df, val_df, test_df, prefix="sample")
    
    print("\n✓ Data preparation complete!")

