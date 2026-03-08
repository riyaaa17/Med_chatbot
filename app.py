from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt

import os
import certifi

# Fix SSL certificate error
os.environ["SSL_CERT_FILE"] = certifi.where()

# Initialize Flask
app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if API keys exist
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing in .env file")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing in .env file")

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Load embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index name
index_name = "medical-chatbot"

# Load existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Load OpenAI model
chatModel = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Create chains
question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


# Home page
@app.route("/")
def index():
    return render_template("chat.html")


# Chat endpoint
@app.route("/get", methods=["POST"])
def chat():

    msg = request.form["msg"]

    print("User input:", msg)

    response = rag_chain.invoke({
        "input": msg
    })

    answer = response["answer"]

    print("Response:", answer)

    return str(answer)


# Run Flask
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8888,
        debug=True
    )