from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

#-------------------------------
from langchain_groq import ChatGroq
#-------------------------------

# Initialize Flask app
app = Flask(__name__)


# Load environment variables (API Keys)
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
#OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

#-------------------------------
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
#-------------------------------

# Inject keys into environment for LangChain components
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#-------------------------------
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
#-------------------------------

# 1. Setup Retrieval Components
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot" 

# Connect to the existing Pinecone index populated during ingestion
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Configure the retriever to fetch top 3 relevant chunks
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 2. Setup Language Model & Chain
"""chatModel = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt), # Loaded from src/prompt.py
    ("human", "{input}"),
])"""

#-------------------------------
# Setup Language Model (Llama 3 via Groq)
chatModel = ChatGroq(
    groq_api_key=os.environ.get('GROQ_API_KEY'),
    model_name="llama-3.3-70b-versatile",
    temperature=0.4
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt), 
    ("human", "{input}"),
])
#-------------------------------



# Assemble the RAG chain
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 3. Routes
@app.route("/")
def index():
    #Renders the main chat interface.
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    #Handles user messages and returns the AI response
    msg = request.form["msg"]
    print(f"User Input: {msg}")

    if not msg:
        return "Please enter a message."
    
    # Process query through the RAG pipeline
    response = rag_chain.invoke({"input": msg})
    print("Response: ", response["answer"])
    
    return str(response["answer"])

# Start the server on port 8080
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
