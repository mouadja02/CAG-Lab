import os

_vector_db = os.getenv("VECTOR_DB", "pinecone").lower()

if _vector_db == "qdrant":
    from cag_lab.retrieval.qdrant_retriever import Chunk, Retriever  # noqa: F401
else:
    from cag_lab.retrieval.pinecone_retriever import Chunk, Retriever  # noqa: F401
