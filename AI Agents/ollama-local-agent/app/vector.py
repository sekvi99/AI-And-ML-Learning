from typing import List, Optional, Any
from pydantic import BaseModel, ConfigDict
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

class VectorStoreManager(BaseModel):
    """
    Manages the creation and retrieval of a vector store for restaurant reviews.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    db_location: str = "./chrome_langchain_db"
    collection_name: str = "restaurant_reviews"
    embedding_model: str = "mxbai-embed-large"
    reviews_csv_path: str = "./resources/reviews.csv"
    vector_store: Optional[Chroma] = None
    retriever: Optional[Any] = None

    def setup(self) -> None:
        """
        Loads reviews, creates embeddings, and initializes the vector store.
        """
        # Load reviews from CSV
        df = pd.read_csv(self.reviews_csv_path)
        embeddings = OllamaEmbeddings(model=self.embedding_model)

        # Ensure the database directory exists
        add_documents = not os.path.exists(self.db_location)

        if add_documents:
            documents: List[Document] = []
            ids: List[str] = []
            for i, row in df.iterrows():
                # Combine title and review for page content
                document = Document(
                    page_content=f"{row['Title']} {row['Review']}",
                    metadata={"rating": row["Rating"], "date": row["Date"]},
                    id=str(i)
                )
                ids.append(str(i))
                documents.append(document)

        # Initialize Chroma vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.db_location,
            embedding_function=embeddings
        )

        # Add documents if needed
        if add_documents:
            self.vector_store.add_documents(documents=documents, ids=ids)

        # Create retriever for querying similar documents
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

    def get_retriever(self) -> Any:
        """
        Returns the retriever for querying the vector store.
        """
        if self.retriever is None:
            self.setup()
        return self.retriever