from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        # 1. Start with a base metadata dictionary
        # Use .get() with a default to avoid errors
        # Define metadata keys
        raw_metadata = {
            "source": doc.metadata.get("source"),
            "region": doc.metadata.get("region"),
            "country": doc.metadata.get("country")
        }
        # 2. Clean the dictionary: Remove any keys where the value is None
        # Pinecone only accepts: string, number, boolean, or list of strings
        clean_metadata = {k: v for k, v in raw_metadata.items() if v is not None}

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata=clean_metadata
            )
        )
    return minimal_docs



#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
    return embeddings