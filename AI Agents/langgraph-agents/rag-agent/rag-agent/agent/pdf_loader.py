from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFLoader:
    """Loads and splits a PDF document into chunks."""
    def __init__(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split(self):
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(pages)