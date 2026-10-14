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
from typing import Dict, List, Optional, Tuple

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

    def _embed_sentence_transformers(self, text: str) -> np.ndarray:
        """ embed using setence transformers."""
        return self.model.encode(text, normalize_embeddings=True)

    def _embed_openai(self, text: str) -> np.ndarray:
        """ embed using openai"""
        response = self.model.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding)

    def _embed_groq(self, text: str) -> np.ndarray:
        """ embed using groq"""
        # groq doesn't have native embeddings, use a local fallback
        # or use a differen model
        logger.warning(" groq doesn't support embeddings natively. using sentence-transformers fallback.")
        return self._embed_sentence_transformers(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # check cache first
        results = []
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            cache_key = f"{self.config.model_name}:{text}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                results.append((i, cached))
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                results.append((i, None)) # placeholder

        # embed uncaced texts in batch
        if uncached_texts:
            if self.config.model_type == "sentence-transformers":
                batch_embeddings = self.model.encode(
                    uncached_texts,
                    normalize_embeddings=True,
                    batch_size=self.config.batch_size
                )
            elif self.config.model_type == "openai":
                batch_embeddings = self._embed_openai_batch(uncached_texts)
            else:
                # fallback to individual
                batch_embeddings = [self.embed(text) for text in uncached_texts]

            # store results
            for idx, embedding in zip(uncached_indices, batch_embeddings):
                cache_key = f"{self.config.model_name}:{uncached_texts[idx]}"
                self.cache.set(cache_key, embedding)
                results[idx] = (idx, embedding)
        #sort by original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def _embed_openai_batch(self, texts: List[str]) -> List[np.ndarray]:
        """ embed batch using openai"""
        response = self.model.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [np.array(item.embedding) for item in response.data]

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        vec1 = self.embed(text1)
        vec2 = self.embed(text2)
        return np.dot(vec1, vec2)

    def similarity_batch(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """
        Compute pairwise similarities between two lists of texts.
        
        Args:
            texts1: First list of texts
            texts2: Second list of texts
            
        Returns:
            List of similarity scores
        """

        if len(texts1) != len(texts2):
            raise ValueError("Lists must have the same length")

        vecs1 = self.embed_batch(texts1)
        vecs2 = self.embed_batch(texts2)

        return[np.dot(v1, v2) for v1, v2 in zip(vecs1, vecs2)]

    def similarity_matrix(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """
        Compute full similarity matrix between two sets of texts.
        
        Args:
            texts1: First set of texts
            texts2: Second set of texts
            
        Returns:
            Similarity matrix (len(texts1) x len(texts2))
        """
        vecs1 = self.embed_batch(texts1)
        vecs2 = self.embed_batch(texts2)

        # stack vectors for efficient computation
        mat1 = np.vstack(vecs1)
        mat2 = np.vstack(vecs2)

        return np.dot(mat1, mat2.T)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self.config.model_type == "sentence_transformers":
            return self.model.get_sentence_embedding_dimension()
        elif self.config.model_type == "openai":
            return 1536 # text-embedding-ada-002 dimension
        else:
            return 384 # default

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")


class SimilarityScorer:
    """
    Utility for computing semantic similarities with caching.
    
    Optimized for repeated similarity computations in the
    context extraction and dependency inference.
    """

    def __init__(self, embedder: EmbeddingManager):
        self.embedder = embedder
        self._similarity_cache: Dict[Tuple[str, str], float] = {}

    def score(self, text1: str, text2: str) -> float:
        """
        Compute similarity with caching.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score
        """
        # normalize texts for caching
        key1 = text1.lower().strip()
        key2 = text2.lower().strip()

        # sort to make cache lookup symmetric
        if key1 > key2:
            key1, key2 = key2, key1

        cache_key = (key1, key2)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        # compute similarity
        similarity = self.embedder.similarity(text1, text2)

        #cache
        self._similarity_cache[cache_key] = similarity

        return similarity

    def score_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Compute similarities for multiple pairs.
        
        Args:
            pairs: List of (text1, text2) pairs
            
        Returns:
            List of similarity scores
        """
        # separate cached and uncached
        cached_scores = []
        uncached_indices = []
        uncached_pairs = []

        for i, (text1, text2) in enumerate(paris):
            key1 = text1.lower().strip()
            key2 = text2.lower().strip()
            if key1 > key2:
                key1, key2 = key2, key1

            cache_key = (key1, key2)
            if cache_key in self._similarity_cache:
                cached_scores.append((i, self._similarity_cache[cache_key]))
            else:
                uncached_indices.append(i)
                uncached_pairs.append((text1, text2))
                cached_scores.append((i, None))

        # compute uncached similarities
        if uncached_pairs:
            texts1 = [p[0] for p in uncached_pairs]
            texts2 = [p[1] for p in uncached_pairs]
            scores = self.embedder.similarity_batch(texts1, texts2)

            # cache and store 
            for idx, score in zip(uncached_indices, scores):
                text1, text2 = uncached_pairs[idx]
                key1 = text1.lower().strip()
                key2 = text2.lower().strip()
                if key1 > key2:
                    key1, key2 = key2, key1
                cache_key = (key1, key2)
                self._similarity_cache[cache_key] = score
                cached_scores[idx] = (idx, score)


        # sort and return
        cached_scores.sort(key=lambda x: x[0])
        return [s for _, s in cached_scores]


