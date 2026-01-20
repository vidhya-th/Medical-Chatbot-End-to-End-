# Medical RAG Chatbot (Llama 3 + Pinecone)

A **Retrieval-Augmented Generation (RAG)** system designed to provide reliable medical information and emergency ambulance assistance using real-time document retrieval.  
This chatbot leverages **Llama 3.3 via Groq** for ultra-fast inference and **Pinecone** for high-accuracy vector search.

---

## Key Features

- **Smart Medical Retrieval**  
  Answers medical questions using private PDF-based medical documentation.

- **Global Ambulance Support**  
  Detects emergency scenarios and returns localized ambulance numbers  
  (e.g., `911` for the US, `112` for the EU, `102/108` for India).

- **High Performance Inference**  
  Powered by **Groq LPU** for near-instant LLM responses.

- **Safety-First Design**  
  Built-in medical disclaimers and emergency detection logic.

---

## Technology Stack

| Component        | Technology |
|------------------|------------|
| LLM              | Llama 3.3-70B-Versatile (via Groq) |
| Orchestration    | LangChain |
| Vector Database  | Pinecone |
| Embeddings       | HuggingFace (`all-MiniLM-L6-v2`) |
| Backend          | Flask |
| Frontend         | HTML, CSS, JavaScript |

---


##  Prerequisites

- Python **3.10+**
- Pinecone API Key
- Groq API Key

---

## Environmemt setup and executions

### STEP 01 — Create and Activate Conda Environment

After opening the repository, create a new Conda environment:

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

### STEP 02 — Install Dependencies

Install all required Python packages:+

```bash
pip install -r requirements.txt
```

### STEP 03 — Environment variables

Create a .env file in the root directory and add your API credentials:

```bash
PINECONE_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### STEP 04 — Store Embeddings in Pinecone

Run the ingestion script to process documents and store embeddings:

```bash
python store_index.py
```

### STEP 05 — Run the Application

Start the Flask application

```bash
python app.py
```
### STEP 06 — Open in Browser

Open your browser and navigate to:

http://localhost:8080

You should now see the Medical RAG Chatbot running locally 

## Repository Structure
```

├── data/               # Medical PDF documents
├── src/
│   ├── helper.py       # Data loading, splitting, and embedding logic
│   └── prompt.py       # System prompts and safety protocols
├── templates/          # HTML files (chat interface)
├── static/             # CSS and JS assets
├── .env                # API Keys (gitignored)
├── app.py              # Flask server and RAG chain
├── store_index.py      # Vector DB ingestion script
└── requirements.txt    # Project dependencies
```
## AWS CI/CD Deployment with GitHub Actions

This section explains how to deploy the Medical RAG Chatbot on **AWS EC2** using **Docker, Amazon ECR**, and **GitHub Actions (Self-Hosted Runner)**.

---

## STEP 01 — AWS IAM Setup

1. Log in to the **AWS Console**
2. Create a new **IAM User** for deployment with **programmatic access**
3. Attach the following policies:

### Required IAM Policies
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

These permissions allow:
- Pushing Docker images to **ECR**
- Managing **EC2 instances**

---

## STEP 02 — Deployment Flow (Overview)

The CI/CD pipeline performs the following steps:

1. Build Docker image from source code
2. Push Docker image to **Amazon ECR**
3. Launch an **EC2 instance**
4. Pull the Docker image from **ECR** onto EC2
5. Run the Docker container on EC2

---

## STEP 03 — Create Amazon ECR Repository

1. Go to **Amazon ECR** in AWS Console
2. Create a new repository

Example repository URI:

```text
315865595366.dkr.ecr.us-east-1.amazonaws.com/medicalbot
```

## Medical Disclaimer

This AI is for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.