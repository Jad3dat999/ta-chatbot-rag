"""
Retriever module using fine-tuned Sentence-BERT
Implements Multiple Negatives Ranking Loss with BM25 hard negatives
"""

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple, Dict, Optional
import pandas as pd
from pathlib import Path
import os
import chromadb
from chromadb.config import Settings

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'


class RetrieverTrainer:
    """Trains Sentence-BERT model for retrieval with hard negatives"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None
    ):
        """
        Initialize retriever trainer
        
        Args:
            model_name: Pre-trained Sentence-BERT model
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Model loaded on {self.device}")
    
    def create_bm25_index(self, documents: List[str]) -> BM25Okapi:
        """
        Create BM25 index for hard negative mining
        
        Args:
            documents: List of answer texts
        
        Returns:
            BM25 index
        """
        tokenized_docs = [doc.lower().split() for doc in documents]
        return BM25Okapi(tokenized_docs)
    
    def mine_hard_negatives(
        self,
        query: str,
        positive_idx: int,
        bm25: BM25Okapi,
        documents: List[str],
        k: int = 5
    ) -> List[int]:
        """
        Mine hard negatives using BM25
        
        Hard negatives are documents that:
        - Look similar (high BM25 score) but are not the correct answer
        - Help the model learn fine-grained distinctions
        
        Args:
            query: Question text
            positive_idx: Index of correct answer
            bm25: BM25 index
            documents: All answer documents
            k: Number of hard negatives to return
        
        Returns:
            List of indices for hard negative documents
        """
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k excluding the positive document
        top_indices = np.argsort(scores)[::-1]
        hard_negatives = [idx for idx in top_indices if idx != positive_idx][:k]
        
        return hard_negatives
    
    def prepare_training_data(
        self,
        train_df: pd.DataFrame,
        num_hard_negatives: int = 3
    ) -> List[InputExample]:
        """
        Prepare training data with hard negatives
        
        Format: (anchor=question, positive=correct_answer, negative=wrong_answer)
        
        Args:
            train_df: Training DataFrame with 'question' and 'answer' columns
            num_hard_negatives: Number of hard negatives per query
        
        Returns:
            List of InputExample for training
        """
        questions = train_df['question'].tolist()
        answers = train_df['answer'].tolist()
        
        # Create BM25 index
        print("Creating BM25 index for hard negative mining...")
        bm25 = self.create_bm25_index(answers)
        
        # Create training examples
        train_examples = []
        
        print(f"Preparing training examples with {num_hard_negatives} hard negatives each...")
        for i, (question, answer) in enumerate(zip(questions, answers)):
            # Mine hard negatives
            hard_neg_indices = self.mine_hard_negatives(
                question, i, bm25, answers, k=num_hard_negatives
            )
            
            # Create InputExample for each hard negative
            for neg_idx in hard_neg_indices:
                train_examples.append(
                    InputExample(
                        texts=[question, answer, answers[neg_idx]],
                        label=1.0  # Question-Answer pair is positive
                    )
                )
        
        print(f"Created {len(train_examples)} training examples")
        return train_examples
    
    def train(
        self,
        train_examples: List[InputExample],
        output_path: str,
        epochs: int = 3,
        batch_size: int = 16,
        warmup_steps: int = 100,
        val_df: pd.DataFrame = None
    ):
        """
        Train Sentence-BERT with Multiple Negatives Ranking Loss
        
        Args:
            train_examples: List of InputExample
            output_path: Where to save the fine-tuned model
            epochs: Number of training epochs
            batch_size: Training batch size
            warmup_steps: Warmup steps for learning rate
            val_df: Optional validation DataFrame for epoch evaluation
        """
        # Create DataLoader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size
        )
        
        # Calculate steps per epoch
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * epochs
        
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"{'='*60}")
        print(f"  Total examples: {len(train_examples):,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total epochs: {epochs}")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Warmup steps: {warmup_steps}")
        if val_df is not None:
            print(f"  Validation examples: {len(val_df)}")
        print(f"{'='*60}\n")
        
        # Define loss function
        # MultipleNegativesRankingLoss: Given (query, positive, negative),
        # it pulls query close to positive and pushes away from negative
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # Train with epoch-by-epoch evaluation
        print(f"{'='*60}")
        print(f"Starting Training...")
        print(f"{'='*60}\n")
        
        best_score = -1
        best_epoch = 0
        training_history = []
        
        for epoch in range(epochs):
            print(f"\n{'─'*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'─'*60}")
            print(f"Training...")
            
            # Train for one epoch
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,
                warmup_steps=warmup_steps if epoch == 0 else 0,  # Only warmup in first epoch
                output_path=None,  # Don't save after each epoch
                show_progress_bar=True,
                use_amp=True  # Use automatic mixed precision for faster training
            )
            
            # Evaluate on validation set if available
            epoch_metrics = {'epoch': epoch + 1}
            
            if val_df is not None:
                print(f"\nEvaluating on validation set...")
                
                # Compute detailed metrics
                detailed_metrics = self._compute_detailed_metrics(val_df)
                epoch_metrics.update(detailed_metrics)
                
                # Use Precision@3 as main score for tracking best model
                val_score = detailed_metrics['precision@3']
                
                # Print metrics
                print(f"\n{'─'*60}")
                print(f"Epoch {epoch + 1} Results:")
                print(f"{'─'*60}")
                print(f"  Precision@1: {detailed_metrics['precision@1']:.4f}")
                print(f"  Precision@3: {detailed_metrics['precision@3']:.4f}")
                print(f"  Precision@5: {detailed_metrics['precision@5']:.4f}")
                print(f"  MRR (Mean Reciprocal Rank): {detailed_metrics['mrr']:.4f}")
                print(f"{'─'*60}")
                
                # Track best model
                if val_score > best_score:
                    best_score = val_score
                    best_epoch = epoch + 1
                    print(f"  ✅ New best model! (Precision@3: {best_score:.4f})")
                    # Save best model
                    self.model.save(str(output_path))
                    print(f"  💾 Model saved to {output_path}")
                else:
                    print(f"  (Best Precision@3: {best_score:.4f} at epoch {best_epoch})")
            else:
                # No validation, save after last epoch
                if epoch == epochs - 1:
                    self.model.save(str(output_path))
                    print(f"\n✓ Model saved to {output_path}")
            
            training_history.append(epoch_metrics)
            print(f"\nEpoch {epoch + 1}/{epochs} complete ✓")
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        if val_df is not None:
            print(f"\n📊 Training Summary:")
            print(f"  Best Precision@3: {best_score:.4f} (Epoch {best_epoch})")
            print(f"\n  Training History:")
            for i, metrics in enumerate(training_history, 1):
                print(f"    Epoch {i}:")
                for key, value in metrics.items():
                    if key != 'epoch':
                        print(f"      {key}: {value:.4f}")
        print(f"\n  Final model saved to: {output_path}")
        print(f"{'='*60}\n")
        
        return training_history
    
    def _compute_detailed_metrics(self, val_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute detailed validation metrics (Precision@k, MRR)
        
        Args:
            val_df: Validation DataFrame
            
        Returns:
            Dictionary of metrics
        """
        questions = val_df['question'].tolist()
        answers = val_df['answer'].tolist()
        
        # Encode all questions and answers
        question_embeddings = self.model.encode(questions, convert_to_tensor=True, show_progress_bar=False)
        answer_embeddings = self.model.encode(answers, convert_to_tensor=True, show_progress_bar=False)
        
        # Compute similarities
        similarities = torch.mm(question_embeddings, answer_embeddings.T)
        
        metrics = {}
        
        # Compute Precision@k
        for k in [1, 3, 5]:
            correct = 0
            for i in range(len(questions)):
                # Get top-k indices
                top_k = torch.topk(similarities[i], k=min(k, len(answers))).indices
                if i in top_k:
                    correct += 1
            
            precision_k = correct / len(questions)
            metrics[f'precision@{k}'] = precision_k
        
        # Compute MRR (Mean Reciprocal Rank)
        reciprocal_ranks = []
        for i in range(len(questions)):
            # Get sorted indices by similarity
            sorted_indices = torch.argsort(similarities[i], descending=True)
            # Find rank of correct answer (add 1 because rank starts at 1)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            reciprocal_ranks.append(1.0 / rank)
        
        mrr = np.mean(reciprocal_ranks)
        metrics['mrr'] = mrr
        
        return metrics
    
    def evaluate(
        self,
        val_df: pd.DataFrame,
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        Evaluate retriever using Precision@k and MRR
        
        Args:
            val_df: Validation DataFrame
            k_values: List of k values for Precision@k
        
        Returns:
            Dictionary of metrics
        """
        questions = val_df['question'].tolist()
        answers = val_df['answer'].tolist()
        
        # Encode all questions and answers
        print("Encoding questions and answers...")
        question_embeddings = self.model.encode(questions, convert_to_tensor=True)
        answer_embeddings = self.model.encode(answers, convert_to_tensor=True)
        
        # Compute similarities
        similarities = torch.mm(question_embeddings, answer_embeddings.T)
        
        metrics = {}
        
        # Compute Precision@k
        for k in k_values:
            correct = 0
            for i in range(len(questions)):
                # Get top-k indices
                top_k = torch.topk(similarities[i], k=k).indices
                if i in top_k:
                    correct += 1
            
            precision_k = correct / len(questions)
            metrics[f'Precision@{k}'] = precision_k
            print(f"Precision@{k}: {precision_k:.4f}")
        
        # Compute MRR (Mean Reciprocal Rank)
        reciprocal_ranks = []
        for i in range(len(questions)):
            # Get sorted indices by similarity
            sorted_indices = torch.argsort(similarities[i], descending=True)
            # Find rank of correct answer (add 1 because rank starts at 1)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            reciprocal_ranks.append(1.0 / rank)
        
        mrr = np.mean(reciprocal_ranks)
        metrics['MRR'] = mrr
        print(f"MRR: {mrr:.4f}")
        
        return metrics


class Retriever:
    """Production retriever for inference"""
    
    def __init__(
        self, 
        model_path: str, 
        device: str = None,
        use_chromadb: bool = False,
        chromadb_path: str = "data/chromadb"
    ):
        """
        Load fine-tuned retriever model
        
        Args:
            model_path: Path to fine-tuned model
            device: 'cuda', 'cpu', or None (auto-detect)
            use_chromadb: Whether to use ChromaDB for storage
            chromadb_path: Path to ChromaDB directory
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading retriever from {model_path}")
        self.model = SentenceTransformer(model_path, device=self.device)
        
        # Storage options
        self.use_chromadb = use_chromadb
        self.documents = []
        self.metadatas = []
        self.document_embeddings = None
        
        # Initialize ChromaDB if requested
        if use_chromadb:
            print(f"Initializing ChromaDB at {chromadb_path}...")
            Path(chromadb_path).mkdir(parents=True, exist_ok=True)
            
            # Create custom embedding function using fine-tuned model
            class CustomEmbeddingFunction:
                def __init__(self, model):
                    self.model = model
                
                def __call__(self, texts):
                    return self.model.encode(texts, convert_to_numpy=True).tolist()
            
            self.client = chromadb.Client(Settings(
                persist_directory=chromadb_path,
                anonymized_telemetry=False
            ))
            
            self.collection = self.client.get_or_create_collection(
                name="retriever_docs",
                embedding_function=CustomEmbeddingFunction(self.model),
                metadata={"description": "Fine-tuned retriever documents"}
            )
            print("✓ ChromaDB initialized with fine-tuned embeddings")
        else:
            self.client = None
            self.collection = None
    
    def index_documents(
        self, 
        documents: List[str], 
        metadatas: List[Dict] = None,
        ids: List[str] = None
    ):
        """
        Index documents for retrieval
        
        Args:
            documents: List of text documents to index
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of document IDs (for ChromaDB)
        """
        print(f"Indexing {len(documents)} documents...")
        self.documents = documents
        self.metadatas = metadatas if metadatas else [{}] * len(documents)
        
        if self.use_chromadb:
            # Store in ChromaDB (embeddings created automatically)
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            self.collection.add(
                documents=documents,
                metadatas=self.metadatas,
                ids=ids
            )
            print("✓ Documents indexed in ChromaDB")
        else:
            # Store embeddings in memory (original behavior)
            self.document_embeddings = self.model.encode(
                documents,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            print("✓ Documents indexed in memory")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_metadata: bool = True,
        use_smart_ranking: bool = True,
        filter_tags: List[str] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve top-k most relevant documents with smart re-ranking
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            return_metadata: Whether to return metadata with results
            use_smart_ranking: Whether to boost assignment/slide docs based on query (in-memory only)
            filter_tags: If provided, only retrieve docs with matching tags (e.g., ['assignment1'])
        
        Returns:
            List of (document, score, metadata) tuples
        """
        # ChromaDB path - with tag filtering support
        if self.use_chromadb:
            if self.collection is None:
                raise ValueError("ChromaDB not initialized")
            
            # Build where clause for tag filtering
            where = None
            if filter_tags:
                # Filter docs that have ANY of the specified tags
                where = {"tags": {"$in": filter_tags}}
            
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where
            )
            
            documents = results['documents'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0] if return_metadata else [{}] * len(documents)
            
            # Convert distances to similarity scores
            scores = [1 / (1 + d) for d in distances]
            
            return list(zip(documents, scores, metadatas))
        
        # In-memory path - with smart ranking
        if self.document_embeddings is None:
            raise ValueError("No documents indexed. Call index_documents() first.")
        
        # TAG-BASED FILTERING (highest priority if specified)
        assignment_doc_indices = []  # Track assignment document indices
        
        if filter_tags:
            valid_indices = []
            for idx, metadata in enumerate(self.metadatas):
                doc_tags = metadata.get('tags', [])
                # Check if doc has ANY of the filter tags
                if any(tag in doc_tags for tag in filter_tags):
                    valid_indices.append(idx)
                    if metadata.get('doc_type') == 'assignment':
                        assignment_doc_indices.append(idx)
            
            if not valid_indices:
                # Fallback: no docs match tags, use all docs
                valid_indices = list(range(len(self.documents)))
            
            # Skip smart ranking when using explicit tag filter
            # (the user knows exactly what they want)
            assignment_filter = None
            problem_filter = None
            is_assignment_query = False
        else:
            # Original smart filtering logic (auto-detect from query)
            # Detect specific assignment/problem numbers in query
            assignment_filter, problem_filter = self._detect_filters(query)
            
            # Detect if this is an assignment-type query (even without specific number)
            query_lower = query.lower()
            assignment_keywords = [
                'assignment', 'homework', 'hw', 'assign',
                'requirements', 'require', 'due', 'deadline', 'due date',
                'submit', 'submission', 'turn in', 'hand in',
                'problem', 'task', 'part'
            ]
            is_assignment_query = any(keyword in query_lower for keyword in assignment_keywords)
            
            # Filter documents BEFORE computing similarities (more efficient)
            valid_indices = []
            
            for idx, metadata in enumerate(self.metadatas):
                doc_type = metadata.get('doc_type', '')
                doc_assignment = metadata.get('assignment_number', '')
                
                # HARD FILTER: For assignment queries, exclude slide documents entirely
                if is_assignment_query and doc_type == 'slide':
                    continue  # Skip all slides for assignment queries
                
                # If query mentions specific assignment number, filter to that assignment
                if assignment_filter:
                    if doc_type == 'assignment':
                        if doc_assignment == assignment_filter:
                            # Keep this assignment's documents
                            valid_indices.append(idx)
                            assignment_doc_indices.append(idx)
                        # Skip other assignments
                    else:
                        # Keep piazza (slides already excluded above)
                        valid_indices.append(idx)
                else:
                    # No specific assignment number, keep all (except slides if assignment query)
                    valid_indices.append(idx)
                    if doc_type == 'assignment':
                        assignment_doc_indices.append(idx)
        
        if not valid_indices:
            # Fallback: no valid docs after filtering, use all docs
            valid_indices = list(range(len(self.documents)))
        
        # For assignment queries, increase top_k to get comprehensive coverage
        # This gives the LLM more context to work with
        original_top_k = top_k
        if assignment_filter and len(assignment_doc_indices) > 0:
            # Retrieve ALL sections of the filtered assignment + some related docs
            top_k = min(len(assignment_doc_indices) + 5, len(valid_indices))
        
        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True
        )
        
        # Compute similarities only for valid documents
        valid_embeddings = self.document_embeddings[valid_indices]
        similarities = torch.mm(
            query_embedding.unsqueeze(0),
            valid_embeddings.T
        ).squeeze(0)
        
        # Smart re-ranking: boost document types based on query
        if use_smart_ranking:
            similarities = self._apply_smart_boosting(query, similarities, valid_indices, problem_filter)
        
        # Get top-k
        top_k_scores, top_k_indices = torch.topk(similarities, k=min(top_k, len(valid_indices)))
        
        # Map back to original indices
        if return_metadata:
            results = [
                (self.documents[valid_indices[idx]], score.item(), self.metadatas[valid_indices[idx]])
                for idx, score in zip(top_k_indices, top_k_scores)
            ]
        else:
            results = [
                (self.documents[valid_indices[idx]], score.item(), {})
                for idx, score in zip(top_k_indices, top_k_scores)
            ]
        
        return results
    
    def _detect_filters(self, query: str) -> tuple:
        """
        Detect specific assignment and problem numbers mentioned in query
        
        Args:
            query: Query text
            
        Returns:
            (assignment_number, problem_number) tuple, e.g. ('1', '2') for "assignment 1 problem 2"
            Returns (None, None) if no specific filters detected
        """
        import re
        query_lower = query.lower()
        
        # Detect assignment number (e.g., "assignment 1", "assignment 2", "hw 3")
        assignment_patterns = [
            r'assignment\s+(\d+)',
            r'hw\s+(\d+)',
            r'homework\s+(\d+)',
            r'assign\s+(\d+)'
        ]
        
        assignment_num = None
        for pattern in assignment_patterns:
            match = re.search(pattern, query_lower)
            if match:
                assignment_num = match.group(1)
                break
        
        # Detect problem/part number (e.g., "problem 1", "part 2", "question 3")
        problem_patterns = [
            r'problem\s+(\d+)',
            r'part\s+(\d+)',
            r'question\s+(\d+)',
            r'task\s+(\d+)'
        ]
        
        problem_num = None
        for pattern in problem_patterns:
            match = re.search(pattern, query_lower)
            if match:
                problem_num = match.group(1)
                break
        
        return assignment_num, problem_num
    
    def _apply_smart_boosting(self, query: str, similarities: torch.Tensor, 
                             valid_indices: List[int], problem_filter: str = None) -> torch.Tensor:
        """
        Apply query-aware boosting to prioritize relevant document types
        
        For example:
        - "assignment requirements" → boost assignment docs
        - "lecture about" → boost slide docs
        - "problem 2" → boost sections containing problem 2
        - general questions → no boosting (rely on semantic similarity)
        
        Args:
            query: Query text (lowercase)
            similarities: Similarity scores for valid documents
            valid_indices: Indices of valid documents in the original list
            problem_filter: Specific problem number to boost (e.g., '2' for problem 2)
            
        Returns:
            Boosted similarity scores
        """
        query_lower = query.lower()
        
        # Detect query intent with comprehensive keywords
        assignment_keywords = [
            'assignment', 'homework', 'hw', 'assign',
            'requirements', 'require', 'due', 'deadline', 'due date',
            'submit', 'submission', 'turn in', 'hand in',
            'problem', 'task', 'part', 'question'
        ]
        slide_keywords = ['lecture', 'slide', 'explain', 'what is', 'how does', 'define']
        
        is_assignment_query = any(keyword in query_lower for keyword in assignment_keywords)
        is_slide_query = any(keyword in query_lower for keyword in slide_keywords) and not is_assignment_query
        
        # Apply boosting
        boosted_similarities = similarities.clone()
        
        for idx, orig_idx in enumerate(valid_indices):
            metadata = self.metadatas[orig_idx]
            doc_type = metadata.get('doc_type', '')
            section_title = metadata.get('section_title', '').lower()
            
            # === MODERATE PRIORITIZATION (hard filter already excludes slides) ===
            if is_assignment_query:
                if doc_type == 'assignment':
                    # Moderate boost (hard filter already removed slides)
                    boosted_similarities[idx] *= 2.0
                    
                    # Extra boost if specific problem is mentioned and matches
                    if problem_filter and f'problem {problem_filter}' in section_title:
                        boosted_similarities[idx] *= 1.5  # Total: 3x boost
                elif doc_type in ['piazza_note', 'piazza_qa']:
                    # Slight penalty (assignments should rank higher)
                    boosted_similarities[idx] *= 0.7
            
            # === MODERATE PRIORITIZATION FOR SLIDE QUERIES ===
            elif is_slide_query:
                if doc_type == 'slide':
                    boosted_similarities[idx] *= 1.5  # Moderate boost for slides
                elif doc_type in ['piazza_note', 'piazza_qa']:
                    boosted_similarities[idx] *= 0.8  # Slight penalty
        
        return boosted_similarities


if __name__ == "__main__":
    # Example usage
    from data_preparation import DataPreparer
    
    print("=== Retriever Training Example ===\n")
    
    # Load data
    preparer = DataPreparer()
    
    # Check if sample data exists
    sample_path = preparer.splits_dir / "sample_train.json"
    if not sample_path.exists():
        print("Creating sample data first...")
        sample_df = preparer.create_sample_data(122)
        augmented_df = preparer.augment_data(sample_df, augmentation_factor=2)
        train_df, val_df, test_df = preparer.split_data(augmented_df)
        preparer.save_splits(train_df, val_df, test_df, prefix="sample")
    
    # Load splits
    train_df, val_df, test_df = preparer.load_splits(prefix="sample")
    
    # Initialize trainer
    trainer = RetrieverTrainer()
    
    # Prepare training data
    train_examples = trainer.prepare_training_data(train_df, num_hard_negatives=2)
    
    # Train (small example - use more epochs in production)
    output_path = "models/retriever-finetuned"
    trainer.train(
        train_examples,
        output_path=output_path,
        epochs=1,  # Use 3-5 in production
        batch_size=16
    )
    
    # Evaluate
    print("\n=== Evaluation on Validation Set ===")
    metrics = trainer.evaluate(val_df, k_values=[1, 3, 5])
    
    print("\n✓ Retriever training complete!")

