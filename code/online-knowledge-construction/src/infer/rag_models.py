# src.infer.rag_models.py

import os
import logging
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

BASE_CHATBOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_CHATBOT_DIR))

VECTOR_DB_PATH = os.path.join(BASE_CHATBOT_DIR, "kb/tasi113")
EMBED_MODEL = os.path.join(ROOT_DIR, "models/Qwen3-Embedding-0.6B")

TOP_K_RETRIEVAL = 1

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model...")
        _embedding_model = SentenceTransformer(
            EMBED_MODEL,
            trust_remote_code=True
        )
    return _embedding_model


class RAGRetriever:
    def __init__(self):
        try:
            self.retriever = get_embedding_model()
            self.db_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
            self.collection = self.db_client.get_collection(
                name="text2sql_questions"
            )
        except Exception as e:
            raise RuntimeError(f"Gagal init RAG: {e}")

    def retrieve(self, question: str, evidence: str = "") -> tuple[str, str, list[dict]]:
        try:
            query_vector = self.retriever.encode(
                [question],
                normalize_embeddings=True
            ).tolist()

            raw_results = self.collection.query(
                query_embeddings=query_vector,
                n_results=TOP_K_RETRIEVAL,
                include=["metadatas", "distances"]
            )

            retrieved = []
            retrieved_items = []
            seen = set()

            if raw_results["metadatas"]:
                for meta, dist in zip(
                    raw_results["metadatas"][0],
                    raw_results["distances"][0]
                ):
                    sid = meta.get("source_id", "")
                    if sid in seen:
                        continue
                    seen.add(sid)

                    if meta.get("question") and meta.get("sql"):
                        retrieved.append(
                            f"-- Example Question: {meta['question']}\n"
                            f"-- Example SQL: {meta['sql']}"
                        )
                        retrieved_items.append({
                            "question": meta["question"],
                            "sql": meta["sql"]
                        })

            if not retrieved:
                return "(No similar examples found)", question, []

            return "\n\n".join(retrieved), question, retrieved_items

        except Exception as e:
            logger.error(f"RAG error: {e}")
            return "(No similar examples found)", question, []