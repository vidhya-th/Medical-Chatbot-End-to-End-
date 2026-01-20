import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from src.helper import (
    load_pdf_file, 
    filter_to_minimal_docs, 
    text_split, 
    download_hugging_face_embeddings
)

# 1. Configuration & Security
# Load environment variables from a .env file to keep API keys secure
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate that essential API keys are present
if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("API Keys missing. Please check your .env file.")

# Set environment variables globally for LangChain integration
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY 
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 2. Data Processing
# Load PDF files from the local directory
extracted_data = load_pdf_file(data='data/')

# Remove unnecessary metadata to save memory and reduce DB noise
filter_data = filter_to_minimal_docs(extracted_data)

# Split documents into smaller chunks (e.g., 500 characters) for better retrieval
text_chunks = text_split(filter_data)

# 3. Define Ambulance Documents (Manual Data)
ambulance_docs = [
    # EUROPE
    Document(page_content="EU & UK Ambulance: 112.", metadata={"region": "Europe"}),
    # AFRICA
    Document(page_content="South Africa Ambulance: 10177.", metadata={"region": "South Africa"}),
    Document(page_content="Egypt Ambulance: 123.", metadata={"region": "Egypt"}),
    # ASIA
    Document(page_content="India Ambulance: 102 or 108.", metadata={"region": "India"}),
    Document(page_content="China Ambulance: 120.", metadata={"region": "China"}),
    # AMERICAS
    Document(page_content="USA & Canada Ambulance: 911.", metadata={"region": "North America"}),
    Document(page_content="Brazil Ambulance: 192.", metadata={"region": "Brazil"})
]

# 4. Combine PDF chunks with Ambulance Data
# This ensures everything is searchable in one index
all_documents = text_chunks + ambulance_docs

# 5. Embedding Initialization
# Download the 'all-MiniLM-L6-v2' model to convert text into 384-dimensional vectors
embeddings = download_hugging_face_embeddings()

# 6. Pinecone Vector DB Setup
# Initialize the Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

# Create the index if it does not already exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,     # Must match the embedding model output
        metric="cosine",    # Similarity search type
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the specific index
index = pc.Index(index_name)

# 7. Document Ingestion
# Convert text chunks into vectors and upload them to Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=all_documents,
    index_name=index_name,
    embedding=embeddings, 
)

print("Ingestion complete. Your vector store is ready for querying.")