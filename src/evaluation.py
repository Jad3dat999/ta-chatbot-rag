"""
Evaluation metrics for both retrieval and generation
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
from collections import defaultdict


class RetrievalEvaluator:
    """Evaluator for retrieval systems"""
    
    @staticmethod
    def precision_at_k(
        retrieved_indices: List[int],
        relevant_index: int,
        k: int
    ) -> float:
        """
        Compute Precision@k
        
        Args:
            retrieved_indices: List of retrieved document indices
            relevant_index: Index of the relevant document
            k: Cutoff position
        
        Returns:
            1.0 if relevant doc is in top-k, else 0.0
        """
        top_k = retrieved_indices[:k]
        return 1.0 if relevant_index in top_k else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(
        retrieved_indices: List[int],
        relevant_index: int
    ) -> float:
        """
        Compute Mean Reciprocal Rank
        
        Args:
            retrieved_indices: List of retrieved document indices
            relevant_index: Index of the relevant document
        
        Returns:
            Reciprocal rank (1/rank) if found, else 0.0
        """
        try:
            rank = retrieved_indices.index(relevant_index) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0
    
    @staticmethod
    def evaluate_retrieval(
        queries: List[str],
        relevant_docs: List[str],
        all_docs: List[str],
        retriever,
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance
        
        Args:
            queries: List of query texts
            relevant_docs: List of relevant document texts (one per query)
            all_docs: All documents in the corpus
            retriever: Retriever object with retrieve() method
            k_values: K values for Precision@k
        
        Returns:
            Dictionary of metrics
        """
        metrics = defaultdict(list)
        
        # Index documents
        retriever.index_documents(all_docs)
        
        for query, relevant_doc in zip(queries, relevant_docs):
            # Find index of relevant document
            try:
                relevant_idx = all_docs.index(relevant_doc)
            except ValueError:
                print(f"Warning: Relevant doc not found for query: {query[:50]}...")
                continue
            
            # Retrieve documents
            retrieved_results = retriever.retrieve(query, top_k=max(k_values))
            retrieved_docs = [doc for doc, score in retrieved_results]
            
            # Get indices
            retrieved_indices = []
            for doc in retrieved_docs:
                try:
                    retrieved_indices.append(all_docs.index(doc))
                except ValueError:
                    continue
            
            # Compute Precision@k for each k
            for k in k_values:
                p_k = RetrievalEvaluator.precision_at_k(
                    retrieved_indices, relevant_idx, k
                )
                metrics[f'Precision@{k}'].append(p_k)
            
            # Compute MRR
            mrr = RetrievalEvaluator.mean_reciprocal_rank(
                retrieved_indices, relevant_idx
            )
            metrics['MRR'].append(mrr)
        
        # Average metrics
        avg_metrics = {
            key: np.mean(values) for key, values in metrics.items()
        }
        
        return avg_metrics


class GenerationEvaluator:
    """Evaluator for generation systems"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize evaluator
        
        Args:
            model_name: Sentence embedding model for semantic similarity
        """
        self.embedding_model = SentenceTransformer(model_name)
    
    def semantic_similarity(
        self,
        generated: str,
        reference: str
    ) -> float:
        """
        Compute semantic similarity between generated and reference text
        
        Args:
            generated: Generated text
            reference: Reference text
        
        Returns:
            Cosine similarity score (0-1)
        """
        embeddings = self.embedding_model.encode([generated, reference])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def accuracy_threshold(
        self,
        similarities: List[float],
        threshold: float = 0.7
    ) -> float:
        """
        Compute accuracy based on similarity threshold
        
        Args:
            similarities: List of similarity scores
            threshold: Similarity threshold for "correct"
        
        Returns:
            Accuracy (proportion above threshold)
        """
        correct = sum(1 for sim in similarities if sim >= threshold)
        return correct / len(similarities)
    
    def evaluate_generation(
        self,
        generated_answers: List[str],
        reference_answers: List[str],
        accuracy_threshold: float = 0.7
    ) -> Dict[str, float]:
        """
        Evaluate generation quality
        
        Args:
            generated_answers: List of generated answers
            reference_answers: List of reference answers
            accuracy_threshold: Threshold for accuracy metric
        
        Returns:
            Dictionary of metrics
        """
        similarities = []
        
        for gen, ref in zip(generated_answers, reference_answers):
            sim = self.semantic_similarity(gen, ref)
            similarities.append(sim)
        
        metrics = {
            'mean_similarity': np.mean(similarities),
            'median_similarity': np.median(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'std_similarity': np.std(similarities),
            'accuracy': self.accuracy_threshold(similarities, accuracy_threshold)
        }
        
        return metrics
    
    def evaluate_source_citation(
        self,
        generated_answers: List[str],
        expected_sources: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate if sources are cited in generated answers
        
        Args:
            generated_answers: List of generated answers
            expected_sources: List of expected source citations for each answer
        
        Returns:
            Citation metrics
        """
        citation_scores = []
        
        for answer, sources in zip(generated_answers, expected_sources):
            answer_lower = answer.lower()
            
            # Count how many expected sources are mentioned
            mentioned = sum(
                1 for source in sources
                if source.lower() in answer_lower
            )
            
            score = mentioned / len(sources) if sources else 0.0
            citation_scores.append(score)
        
        metrics = {
            'mean_citation_score': np.mean(citation_scores),
            'citation_accuracy': sum(1 for s in citation_scores if s > 0) / len(citation_scores)
        }
        
        return metrics


class ComprehensiveEvaluator:
    """Complete evaluation pipeline for the TA chatbot"""
    
    def __init__(self):
        self.retrieval_eval = RetrievalEvaluator()
        self.generation_eval = GenerationEvaluator()
    
    def evaluate_rag_system(
        self,
        test_df: pd.DataFrame,
        rag_pipeline,
        accuracy_threshold: float = 0.8
    ) -> Dict[str, Dict]:
        """
        Comprehensive evaluation of RAG system
        
        Args:
            test_df: Test DataFrame with questions and answers
            rag_pipeline: RAGPipeline object
            accuracy_threshold: Threshold for accuracy metric
        
        Returns:
            Dictionary with all metrics
        """
        print("=== Evaluating RAG System ===\n")
        
        results = {
            'retrieval': {},
            'generation': {},
            'overall': {}
        }
        
        # Evaluate retrieval
        print("1. Evaluating retrieval...")
        queries = test_df['question'].tolist()
        reference_answers = test_df['answer'].tolist()
        
        # Get all documents from the knowledge base
        # In practice, this would be your indexed documents
        
        # Evaluate generation
        print("2. Evaluating generation...")
        generated_answers = []
        
        for query in queries:
            response = rag_pipeline.ask(query)
            generated_answers.append(response['answer'])
        
        gen_metrics = self.generation_eval.evaluate_generation(
            generated_answers,
            reference_answers,
            accuracy_threshold=accuracy_threshold
        )
        
        results['generation'] = gen_metrics
        
        # Print results
        print("\n=== Results ===")
        print("\nGeneration Metrics:")
        for metric, value in gen_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Check if accuracy target is met
        if gen_metrics['accuracy'] >= accuracy_threshold:
            print(f"\n✓ Accuracy target met: {gen_metrics['accuracy']:.2%} >= {accuracy_threshold:.2%}")
        else:
            print(f"\n✗ Accuracy target not met: {gen_metrics['accuracy']:.2%} < {accuracy_threshold:.2%}")
        
        return results
    
    def compare_ablation_results(
        self,
        rag_only_results: List[Dict],
        lora_only_results: List[Dict],
        hybrid_results: List[Dict]
    ):
        """
        Compare ablation study results
        
        Args:
            rag_only_results: Results from RAG-only approach
            lora_only_results: Results from LoRA-only approach
            hybrid_results: Results from hybrid approach
        """
        print("\n=== Ablation Study Comparison ===\n")
        
        all_results = {
            'RAG-only': rag_only_results,
            'LoRA-only': lora_only_results,
            'Hybrid': hybrid_results
        }
        
        comparison = {}
        
        for method, results in all_results.items():
            generated = [r['generated_answer'] for r in results]
            references = [r['true_answer'] for r in results]
            
            metrics = self.generation_eval.evaluate_generation(
                generated,
                references
            )
            
            comparison[method] = metrics
        
        # Print comparison table
        print(f"{'Metric':<25} {'RAG-only':<12} {'LoRA-only':<12} {'Hybrid':<12}")
        print("-" * 61)
        
        for metric in ['mean_similarity', 'accuracy']:
            print(
                f"{metric:<25} "
                f"{comparison['RAG-only'][metric]:<12.4f} "
                f"{comparison['LoRA-only'][metric]:<12.4f} "
                f"{comparison['Hybrid'][metric]:<12.4f}"
            )
        
        # Determine best method
        best_method = max(
            comparison.items(),
            key=lambda x: x[1]['accuracy']
        )[0]
        
        print(f"\n✓ Best performing method: {best_method}")
        
        return comparison


if __name__ == "__main__":
    print("=== Evaluation Module ===\n")
    
    # Example: Evaluate semantic similarity
    evaluator = GenerationEvaluator()
    
    generated = "Backpropagation is an algorithm for training neural networks using gradient descent."
    reference = "Backpropagation computes gradients using the chain rule to update neural network weights."
    
    similarity = evaluator.semantic_similarity(generated, reference)
    print(f"Semantic similarity: {similarity:.4f}")
    
    # Example: Evaluate a batch
    generated_answers = [
        "Use dropout to prevent overfitting",
        "Adam optimizer uses adaptive learning rates",
        "CNN is for image processing"
    ]
    
    reference_answers = [
        "Dropout is a regularization technique that prevents overfitting",
        "Adam uses adaptive learning rates for each parameter",
        "Convolutional Neural Networks process spatial data like images"
    ]
    
    metrics = evaluator.evaluate_generation(
        generated_answers,
        reference_answers,
        accuracy_threshold=0.7
    )
    
    print("\nBatch evaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n✓ Evaluation module ready!")

