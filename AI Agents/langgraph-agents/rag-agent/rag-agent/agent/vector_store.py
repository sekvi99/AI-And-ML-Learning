import os
from langchain_chroma import Chroma

class VectorStoreManager:
    """Manages the creation and retrieval of a Chroma vector store."""
    def __init__(self, persist_directory: str, collection_name: str, embeddings):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embeddings = embeddings

    def create_vector_store(self, documents):
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )