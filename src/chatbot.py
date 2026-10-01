"""
RAG Pipeline: Combines retriever and generator
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
from pathlib import Path
import chromadb
from chromadb.config import Settings


class ChromaDBManager:
    """Manages ChromaDB for document storage and retrieval"""
    
    def __init__(self, persist_directory: str = "data/chromadb", embedding_function=None):
        """
        Initialize ChromaDB
        
        Args:
            persist_directory: Where to store the database
            embedding_function: Optional custom embedding function (e.g., fine-tuned Sentence-BERT)
        """
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        self.collection = None
        self.embedding_function = embedding_function
    
    def create_collection(self, collection_name: str = "ta_chatbot_docs"):
        """Create or get collection"""
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "TA Chatbot knowledge base"},
            embedding_function=self.embedding_function
        )
        print(f"Collection '{collection_name}' ready")
        if self.embedding_function:
            print(f"  Using custom embedding function: {type(self.embedding_function).__name__}")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to collection
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dicts (e.g., source, tags)
            ids: List of document IDs
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(documents)} documents to collection")
    
    def query(
        self,
        query_text: str,
        n_results: int = 3,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict:
        """
        Query the collection with optional metadata filtering
        
        Args:
            query_text: Query string
            n_results: Number of results to return
            where: Metadata filter (e.g., {"doc_type": "slide"})
            where_document: Document content filter
        
        Returns:
            Query results with documents and metadata
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        return results
    
    def delete_collection(self):
        """Delete the collection"""
        if self.collection:
            self.client.delete_collection(self.collection.name)
            print(f"Deleted collection '{self.collection.name}'")


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation"""
    
    def __init__(
        self,
        retriever_path: Optional[str] = None,
        generator_path: Optional[str] = None,
        chromadb_path: str = "data/chromadb",
        use_chromadb: bool = True
    ):
        """
        Initialize RAG pipeline
        
        Args:
            retriever_path: Path to fine-tuned retriever (if provided, ChromaDB will use it for embeddings)
            generator_path: Path to fine-tuned generator
            chromadb_path: Path to ChromaDB
            use_chromadb: Whether to use ChromaDB for retrieval
        """
        self.use_chromadb = use_chromadb
        self.retriever = None
        
        # If retriever_path provided, load it first (for ChromaDB embedding function)
        if retriever_path:
            from sentence_transformers import SentenceTransformer
            print(f"Loading fine-tuned retriever from {retriever_path}...")
            # Create a custom embedding function for ChromaDB
            model = SentenceTransformer(retriever_path)
            
            # Create embedding function compatible with ChromaDB
            class CustomEmbeddingFunction:
                def __init__(self, model):
                    self.model = model
                
                def __call__(self, texts):
                    # ChromaDB expects list of embeddings
                    return self.model.encode(texts, convert_to_numpy=True).tolist()
            
            embedding_fn = CustomEmbeddingFunction(model)
            print("✓ Fine-tuned model loaded for embeddings")
        else:
            embedding_fn = None
            print("No fine-tuned retriever provided, using default embeddings")
        
        # Initialize ChromaDB with custom embedding function
        if use_chromadb:
            print("Initializing ChromaDB...")
            self.db_manager = ChromaDBManager(chromadb_path, embedding_function=embedding_fn)
            self.db_manager.create_collection()
        
        # Initialize generator
        self.generator = None
        if generator_path:
            from generator import Generator
            print("Loading generator...")
            self.generator = Generator(lora_path=generator_path)
    
    def index_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Index documents for retrieval
        
        Args:
            documents: List of document texts
            metadatas: List of metadata (source, tags, etc.)
            ids: Document IDs
        """
        # Index in ChromaDB (uses fine-tuned embeddings if provided)
        if self.use_chromadb:
            self.db_manager.add_documents(documents, metadatas, ids)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filter_tags: Optional[List[str]] = None,
        filter_doc_type: Optional[str] = None,
        boost_by_tags: bool = True
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve relevant documents using ChromaDB (with fine-tuned embeddings if provided)
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            filter_tags: Filter by tags (if provided, only return docs with these tags)
            filter_doc_type: Filter by document type (e.g., "slide", "piazza_qa")
            boost_by_tags: Boost scores for documents matching query keywords in tags
        
        Returns:
            List of (document, score, metadata) tuples
        """
        if not self.use_chromadb:
            raise ValueError("ChromaDB not initialized")
        
        # Build metadata filter
        where = {}
        if filter_doc_type:
            where['doc_type'] = filter_doc_type
        
        # Get more results initially for re-ranking
        fetch_k = top_k * 3 if boost_by_tags else top_k
        
        results = self.db_manager.query(
            query, 
            n_results=fetch_k,
            where=where if where else None
        )
        
        documents = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        
        # Convert distances to similarity scores (lower distance = higher similarity)
        base_scores = [1 / (1 + d) for d in distances]
        
        # Apply tag-based boosting
        if boost_by_tags and metadatas:
            query_terms = set(query.lower().split())
            final_scores = []
            
            for i, (score, metadata) in enumerate(zip(base_scores, metadatas)):
                boost = 1.0
                doc_tags = metadata.get('tags', [])
                
                if doc_tags:
                    # Check if any query terms match document tags
                    doc_tags_lower = [str(tag).lower() for tag in doc_tags]
                    matching_tags = sum(
                        1 for term in query_terms 
                        if any(term in tag or tag in term for tag in doc_tags_lower)
                    )
                    
                    if matching_tags > 0:
                        boost = 1.0 + (matching_tags * 0.15)  # 15% boost per matching tag
                
                # Boost high importance docs
                if metadata.get('importance') == 'high':
                    boost *= 1.1
                
                # Boost definitions for "what is" type questions
                if metadata.get('is_definition') and any(
                    phrase in query.lower() for phrase in ['what is', 'define', 'definition of']
                ):
                    boost *= 1.2
                
                final_scores.append(score * boost)
        else:
            final_scores = base_scores
        
        # Combine and sort
        results_with_metadata = list(zip(documents, final_scores, metadatas))
        results_with_metadata.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return results_with_metadata[:top_k]
    
    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        max_new_tokens: int = 256
    ) -> str:
        """
        Generate answer
        
        Args:
            query: Question
            context: Retrieved context
            max_new_tokens: Max tokens to generate
        
        Returns:
            Generated answer
        """
        if not self.generator:
            raise ValueError("Generator not initialized")
        
        return self.generator.generate(
            query,
            context=context,
            max_new_tokens=max_new_tokens
        )
    
    def ask(
        self,
        query: str,
        top_k: int = 3,
        include_context: bool = True,
        return_sources: bool = True,
        filter_tags: Optional[List[str]] = None,
        filter_doc_type: Optional[str] = None
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve + generate
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            include_context: Whether to include retrieved context in generation
            return_sources: Whether to return source documents
            filter_tags: Filter by tags (optional)
            filter_doc_type: Filter by document type (optional)
        
        Returns:
            Dictionary with answer and sources
        """
        # Retrieve relevant documents (uses fine-tuned embeddings if provided)
        retrieved_docs = self.retrieve(
            query, 
            top_k=top_k,
            filter_tags=filter_tags,
            filter_doc_type=filter_doc_type
        )
        
        # Format context
        context = None
        if include_context and retrieved_docs:
            context = "\n\n".join([
                f"[Source {i+1}]: {doc}"
                for i, (doc, score, metadata) in enumerate(retrieved_docs)
            ])
        
        # Generate answer
        answer = self.generate(query, context=context)
        
        # Prepare response
        response = {
            "question": query,
            "answer": answer,
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "text": doc[:200] + "..." if len(doc) > 200 else doc,
                    "score": float(score),
                    "source": metadata.get('source', 'unknown'),
                    "doc_type": metadata.get('doc_type', 'unknown'),
                    "tags": metadata.get('tags', [])
                }
                for doc, score, metadata in retrieved_docs
            ]
        
        return response


class AblationStudy:
    """Run ablation studies to compare different approaches"""
    
    def __init__(self, data_path: str = "data/splits"):
        """
        Initialize ablation study
        
        Args:
            data_path: Path to data splits
        """
        self.data_path = Path(data_path)
    
    def run_rag_only(self, test_df: pd.DataFrame) -> List[Dict]:
        """
        Test A: RAG-only (no fine-tuning)
        Uses base Sentence-BERT + ChromaDB
        """
        print("\n=== Test A: RAG-only (Baseline) ===")
        
        # Initialize RAG with base models
        rag = RAGPipeline(
            retriever_path=None,  # Use ChromaDB default embeddings
            generator_path=None,  # Use base Mistral without LoRA
            use_chromadb=True
        )
        
        # Index documents (answers from training data)
        # In practice, load from training set
        
        results = []
        for _, row in test_df.iterrows():
            response = rag.ask(row['question'])
            results.append({
                "question": row['question'],
                "true_answer": row['answer'],
                "generated_answer": response['answer'],
                "method": "RAG-only"
            })
        
        return results
    
    def run_lora_only(self, test_df: pd.DataFrame, lora_path: str) -> List[Dict]:
        """
        Test B: LoRA-only (no retrieval)
        Uses fine-tuned Mistral without retrieved context
        """
        print("\n=== Test B: LoRA-only ===")
        
        from generator import Generator
        
        generator = Generator(lora_path=lora_path)
        
        results = []
        for _, row in test_df.iterrows():
            answer = generator.generate(row['question'], context=None)
            results.append({
                "question": row['question'],
                "true_answer": row['answer'],
                "generated_answer": answer,
                "method": "LoRA-only"
            })
        
        return results
    
    def run_hybrid(
        self,
        test_df: pd.DataFrame,
        retriever_path: str,
        generator_path: str
    ) -> List[Dict]:
        """
        Test C: Hybrid RAG + LoRA
        Uses fine-tuned retriever + fine-tuned generator
        """
        print("\n=== Test C: Hybrid (RAG + LoRA) ===")
        
        rag = RAGPipeline(
            retriever_path=retriever_path,
            generator_path=generator_path,
            use_chromadb=True
        )
        
        results = []
        for _, row in test_df.iterrows():
            response = rag.ask(row['question'], retrieval_method="sbert")
            results.append({
                "question": row['question'],
                "true_answer": row['answer'],
                "generated_answer": response['answer'],
                "method": "Hybrid",
                "sources": response.get('sources', [])
            })
        
        return results
    
    def compare_results(
        self,
        rag_only_results: List[Dict],
        lora_only_results: List[Dict],
        hybrid_results: List[Dict]
    ):
        """
        Compare results from all three approaches
        
        Metrics:
        - Accuracy (semantic similarity)
        - Response quality
        - Source citation accuracy
        """
        print("\n=== Ablation Study Results ===")
        
        all_results = {
            "RAG-only": rag_only_results,
            "LoRA-only": lora_only_results,
            "Hybrid": hybrid_results
        }
        
        for method, results in all_results.items():
            print(f"\n{method}:")
            print(f"  Total questions: {len(results)}")
            # Add more metrics here
            # - BLEU score
            # - ROUGE score
            # - Semantic similarity
            # - Human evaluation scores


if __name__ == "__main__":
    print("=== RAG Pipeline Example ===\n")
    
    # Example: Initialize RAG with ChromaDB only
    rag = RAGPipeline(
        retriever_path=None,
        generator_path=None,
        use_chromadb=True
    )
    
    # Index some sample documents
    sample_docs = [
        "Backpropagation is an algorithm used to train neural networks by computing gradients using the chain rule.",
        "SGD (Stochastic Gradient Descent) updates weights using mini-batches, while Adam uses adaptive learning rates.",
        "Dropout is a regularization technique that randomly drops neurons during training to prevent overfitting.",
    ]
    
    rag.index_documents(
        sample_docs,
        metadatas=[
            {"source": "Lecture 3, Slide 15"},
            {"source": "Lecture 5, Slide 22"},
            {"source": "Lecture 7, Slide 10"}
        ]
    )
    
    # Query
    print("Querying: 'How does backpropagation work?'\n")
    results = rag.retrieve("How does backpropagation work?", top_k=2)
    
    for i, (doc, score) in enumerate(results):
        print(f"Result {i+1} (score: {score:.3f}):")
        print(f"  {doc}\n")
    
    print("✓ RAG pipeline ready!")
    print("\nFor full RAG pipeline with generation, initialize with generator_path.")

