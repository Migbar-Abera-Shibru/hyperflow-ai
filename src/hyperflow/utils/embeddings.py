"""
Embeddings module for HyperFlow AI.

Provides semantic embeddings for:
- Tool descriptions (for task-tool matching)
- Schema descriptions (for node anchoring)
- Dependency inference (semantic similarity between schemas)

Supports multiple backends:
- Sentence Transformers (local, free)
- OpenAI embeddings (higher quality, paid)
- Groq embeddings (free, fast)

Production considerations:
- Caching embeddings for performance
- Batch processing for efficiency
- Fallback to simpler similarity when embeddings unavailable
- Configurable embedding model

Usage:
    embedder = EmbeddingManager()
    
    # Single embedding
    vec = embedder.embed("Get user details")
    
    # Batch embedding
    vecs = embedder.embed_batch(["tool1 desc", "tool2 desc"])
    
    # Similarity
    sim = embedder.similarity("user_id", "recipient_id")
"""



from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """ configuration for embedding models."""
    model_type: str = "sentence_transformers" # sentence_transofmers, openai, groq
    model_name: str = "all_MiniLM-L6-v2"
    cache_size: int = 1000
    cache_dir: Optional[str] = None
    batch_size: int = 32
    device: str = "cpu" # cpu, cuda
    use_cache: bool = True

class EmbeddingCache:
    """
    LRU cache for embeddings.
    
    Stores embeddings in memory and optionally persists to disk.
    """
    def __init__(self, max_size: int = 10000, persist_dir: Optional[str] = None):
        self.max_size = max_size
        self.persist_dir = persist_dir
        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []

        if persist_dir:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[np.ndarray]:
        """get embedding from cache."""
        if key in self._cache:
            # update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]

        # try to load from disk
        if self.persist_dir:
            cache_file = Path(self.persist_dir) / f"{hashlib.md5(key.encode()).hexdigest()}.npy"

            if cache_file.exists():
                embedding = np.load(cache_file)
                self._cache[key] = embedding
                self._access_order.append(key)
                return embedding

        return None


    def set(self, key: str, embedding: np.ndarray) -> None:
        """ store embedding in cache."""
        # evict if needed
        if len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = embedding
        self._access_order.append(key)

        # persist to disk
        if self.persist_dir:
            cache_file = Path(self.persist_dir) / f"{hashlib.md5(key.encode()).hexdigest()}.npy"
            np.save(cache_file, embedding)

    def clear(self) -> None:
        """ clear the cache."""
        self._cache.clear()
        self._access_order.clear()

        if self.persist_dir:
            import shutil
            shutil.rmtree(self.persist_dir)
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)


class EmbeddingManager:
    """
    Manages embeddings for the HyperFlow system.
    
    Provides a unified interface for:
    - Text embedding
    - Similarity computation
    - Batch processing
    - Caching
    
    Example:
        embedder = EmbeddingManager()
        
        # Embed a single text
        vec = embedder.embed("Get user information")
        
        # Compute similarity between two texts
        sim = embedder.similarity("user_id", "recipient_id")
        
        # Batch embed texts
        vecs = embedder.embed_batch(["text1", "text2", "text3"])
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedding manager.
        
        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()
        self.model = None
        self.cache = EmbeddingCache(
            max_size=self.config.cache_size,
            persist_dir=self.config.cache_dir
        )

        self._init_model()

    def _init_model(self) -> None:
        """ initialize the embedding model"""
        if self.config.model_type == "sentence_transformers":
            self._init_sentence_transformers()
        elif self.config.model_type == "openai":
            self._initopenai()
        elif self.config.model_type == "groq":
            self._init_groq()
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        logger.info(f"Initialized embedding model: {self.config.model_type} - {self.config.model_name}")

    def _init_sentence_transformers(self) -> None:
        """ Initialize sentence transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
        except ImportError:
            raise ImportError(
                "sentence-transformer not installed"
                "install with: pip install sentence-transformers"
            )
    def _init_openai(self) -> None:
        """ initialize OpenAI embeddings."""
        try:
            import openai
            self.model = openai.Embedding
            self.api_key = openai.api_key
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")

    def _init_groq(self) -> None:
        """ initialize Groq embeddings."""
        try:
            from groq import Groq
            self.model = Groq()
            self.api_key = self.model.api_key
        except ImportError:
            raise ImportError("groq not installed. Install with: pip install groq")

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text:
            return np.zero(self.get_embedding_dimension())

        # check cache
        cache_key = f"{self.config.model_name}:{text}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # generate embedding
        if self.config.model_type == "sentence_transformers":
            embedding = self._embed_sentence_transformers(text)
        elif self.config.model_type == "openai":
            embedding = self._embed_openai(text)
        elif self.config.model_type == "groq":
            embedding = self._embed_groq(text)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        # cache
        self.cache.set(cache_key, embedding)

        return embedding
