"""
TA Chatbot - Source Package
"""

__version__ = "1.0.0"

from .data_preparation import DataPreparer
from .retriever import RetrieverTrainer, Retriever
from .generator import GeneratorTrainer, Generator
from .chatbot import RAGPipeline, ChromaDBManager, AblationStudy
from .evaluation import RetrievalEvaluator, GenerationEvaluator, ComprehensiveEvaluator
from .utils import (
    is_piazza_document,
    count_documents_by_type,
    print_document_stats,
    filter_documents_for_training,
    filter_documents_for_retrieval
)

__all__ = [
    'DataPreparer',
    'RetrieverTrainer',
    'Retriever',
    'GeneratorTrainer',
    'Generator',
    'RAGPipeline',
    'ChromaDBManager',
    'AblationStudy',
    'RetrievalEvaluator',
    'GenerationEvaluator',
    'ComprehensiveEvaluator',
    'is_piazza_document',
    'count_documents_by_type',
    'print_document_stats',
    'filter_documents_for_training',
    'filter_documents_for_retrieval',
]

