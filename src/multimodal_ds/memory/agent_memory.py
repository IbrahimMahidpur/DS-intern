import logging
import time
import uuid
from typing import Optional
from datetime import datetime

import httpx
import chromadb

from multimodal_ds.config import CHROMA_DIR, OLLAMA_BASE_URL, EMBED_MODEL

logger = logging.getLogger(__name__)

class AgentMemory:
    def __init__(self, collection_name: str = "agent_memory", ttl_seconds: int = 86400):
        self.collection_name = collection_name
        self.ttl_seconds = ttl_seconds
        self._client = None
        self._collection = None
        self._last_purge_time = 0
        self._init_chroma()

    def _init_chroma(self):
        def _create_collection(client):
            return client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        def _validate_dimensions(client, collection) -> bool:
            """Return True if collection dimensions match current embed model."""
            test_embed = self._get_embedding("dimension_check")
            if not test_embed:
                return True  # Ollama not running — assume OK, fail later on store
            try:
                collection.upsert(
                    ids=["__dim_check__"],
                    documents=["dimension check"],
                    embeddings=[test_embed],
                    metadatas=[{"type": "dim_check"}],
                )
                try:
                    collection.delete(ids=["__dim_check__"])
                except Exception:
                    pass
                return True
            except Exception as e:
                if "dimension" in str(e).lower():
                    return False
                return True  # Unknown error — don't delete on unknown errors

        try:
            self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))

            existing_names = [c.name for c in self._client.list_collections()]
            if self.collection_name in existing_names:
                collection = self._client.get_collection(name=self.collection_name)
                if not _validate_dimensions(self._client, collection):
                    logger.warning(
                        f"[Memory] Collection '{self.collection_name}' has wrong "
                        f"dimensions — deleting and recreating"
                    )
                    self._client.delete_collection(name=self.collection_name)
                    collection = _create_collection(self._client)
                    logger.info("[Memory] Collection recreated with correct dimensions")
                self._collection = collection
            else:
                self._collection = _create_collection(self._client)

            logger.info("[Memory] ChromaDB initialized (persistent mode)")

        except Exception as e:
            logger.warning(f"[Memory] Persistent ChromaDB init failed: {e}")
            try:
                self._client = chromadb.EphemeralClient()
                self._collection = _create_collection(self._client)
                logger.info("[Memory] ChromaDB initialized (in-memory mode)")
            except Exception as e2:
                logger.warning(f"[Memory] In-memory ChromaDB init failed: {e2}")
                self._collection = None

    def count(self) -> int:
        """Return number of entries in the collection."""
        if not self._collection:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def store(self, content: str, metadata: dict = None, doc_id: str = None) -> str:
        entry_id = doc_id or str(uuid.uuid4())
        meta = {"timestamp": datetime.utcnow().isoformat(), **(metadata or {})}
        meta = {k: str(v) for k, v in meta.items()}
        if self._collection:
            embedding = self._get_embedding(content)
            try:
                if embedding:
                    self._collection.upsert(
                        ids=[entry_id],
                        documents=[content],
                        embeddings=[embedding],
                        metadatas=[meta],
                    )
                else:
                    # No embedding available — use ChromaDB's built-in embedding
                    # by passing documents only (ChromaDB uses its default embedder)
                    self._collection.upsert(
                        ids=[entry_id],
                        documents=[content],
                        metadatas=[meta],
                    )
            except Exception as e:
                if "dimension" in str(e).lower():
                    logger.warning(
                        f"[Memory] Dimension mismatch — recreating collection '{self.collection_name}'"
                    )
                    try:
                        self._client.delete_collection(name=self.collection_name)
                        self._collection = self._client.get_or_create_collection(
                            name=self.collection_name,
                            metadata={"hnsw:space": "cosine"},
                        )
                        # Retry without custom embedding to avoid dimension issues
                        self._collection.upsert(
                            ids=[entry_id],
                            documents=[content],
                            metadatas=[meta],
                        )
                        logger.info("[Memory] Collection recreated and store succeeded")
                    except Exception as e2:
                        logger.warning(f"[Memory] Store after recreation failed: {e2}")
                else:
                    logger.warning(f"[Memory] Store failed: {e}")
        if time.time() - getattr(self, "_last_purge_time", 0) > 3600:
            self._purge_expired()
        return entry_id

    def retrieve(self, query: str, n_results: int = 5, where: dict = None) -> list:
        if not self._collection:
            return []
        try:
            embedding = self._get_embedding(query)
            count = self._collection.count()
            if count == 0:
                return []
            kwargs = {"n_results": min(n_results, count)}
            if embedding:
                kwargs["query_embeddings"] = [embedding]
            else:
                kwargs["query_texts"] = [query]
            if where:
                # Chroma >=0.4.x requires explicit operator syntax for ALL filters,
                # including single-key ones. Passing a raw {key: value} dict worked
                # in older versions but silently returns empty results in newer ones.
                # Normalize: single key → {key: {"$eq": value}}, multi-key → $and
                if len(where) == 1:
                    k, v = next(iter(where.items()))
                    kwargs["where"] = {k: {"$eq": v}}
                else:
                    kwargs["where"] = {
                        "$and": [{k: {"$eq": v}} for k, v in where.items()]
                    }
            results = self._collection.query(**kwargs)
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            # Filter out expired entries based on timestamp TTL
            filtered = []
            cutoff = datetime.utcnow().timestamp() - self.ttl_seconds
            for d, m in zip(docs, metas):
                ts_str = m.get("timestamp")
                try:
                    ts = datetime.fromisoformat(ts_str).timestamp()
                except Exception:
                    ts = 0
                if ts >= cutoff:
                    filtered.append({"content": d, "metadata": m})
            return filtered
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning(
                    f"[Memory] Retrieve dimension mismatch — returning empty. "
                    f"Ensure EMBED_MODEL matches existing collection. Error: {e}"
                )
            else:
                logger.warning(f"[Memory] Retrieve failed: {e}")
            return []

    def store_analysis_step(self, step_name: str, result: str, session_id: str = "default"):
        return self.store(
            content=f"[Step: {step_name}]\n{result}",
            metadata={"step": step_name, "session_id": session_id, "type": "analysis_step"}
        )

    def get_session_history(self, session_id: str) -> list:
        return self.retrieve(query="analysis step result", n_results=20, where={"session_id": session_id})

    def _get_embedding(self, text: str) -> Optional[list]:
        try:
            model_name = EMBED_MODEL.replace("ollama/", "")
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": model_name, "prompt": text[:2000]},
                timeout=10,  # Fast fail — don't block pipeline for embeddings
            )
            if response.status_code == 200:
                embedding = response.json().get("embedding")
                if embedding:
                    logger.debug(
                        f"[Memory] Embedding OK — model={model_name}, dims={len(embedding)}"
                    )
                return embedding
            else:
                logger.warning(
                    f"[Memory] Embedding HTTP {response.status_code} for model '{model_name}'. "
                    f"Response: {response.text[:200]}"
                )
        except Exception as e:
            logger.warning(f"[Memory] Embedding failed for model '{model_name}': {e}")
        return None

    def _purge_expired(self):
        """Delete entries older than TTL from the Chroma collection.

        Uses paginated fetching (500 IDs at a time) to avoid loading the entire
        collection into memory — critical when the collection has 100k+ entries.
        Chroma's .get() supports limit/offset for exactly this use case.
        """
        if not self._collection:
            return
        cutoff = datetime.utcnow().timestamp() - self.ttl_seconds
        to_delete = []
        _PAGE_SIZE = 500
        offset = 0

        try:
            total = self._collection.count()
            if total == 0:
                return

            while offset < total:
                try:
                    page = self._collection.get(
                        include=["metadatas"],
                        limit=_PAGE_SIZE,
                        offset=offset,
                    )
                except TypeError:
                    # Older chromadb versions (<0.4.x) don't support limit/offset —
                    # fall back to single full fetch but log a warning
                    logger.warning(
                        "[Memory] chromadb does not support paginated .get() — "
                        "loading all IDs for TTL purge. Upgrade chromadb>=0.4.0."
                    )
                    page = self._collection.get(include=["metadatas"])
                    offset = total  # exit loop after this iteration

                ids = page.get("ids", [])
                metas = page.get("metadatas", [])

                for entry_id, meta in zip(ids, metas):
                    ts_str = meta.get("timestamp") if meta else None
                    if not ts_str:
                        continue
                    try:
                        ts = datetime.fromisoformat(ts_str).timestamp()
                    except Exception:
                        continue
                    if ts < cutoff:
                        to_delete.append(entry_id)

                offset += _PAGE_SIZE

            if to_delete:
                # Delete in batches to avoid hitting Chroma's internal limits
                _DELETE_BATCH = 200
                for i in range(0, len(to_delete), _DELETE_BATCH):
                    self._collection.delete(ids=to_delete[i:i + _DELETE_BATCH])
                logger.info(f"[Memory] Purged {len(to_delete)} expired entries")
                self._last_purge_time = time.time()

        except Exception as e:
            logger.debug(f"[Memory] Purge expired failed: {e}")

