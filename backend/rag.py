import os

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "./data/chroma_db"

documents = [
    "Docker is a container platform used to package applications.",
    "Kubernetes manages container orchestration.",
    "MLflow tracks machine learning experiments.",
    "Jenkins is used for CI/CD pipelines.",
    "Terraform is Infrastructure as Code tool.",
]

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def create_vector_db():

    os.makedirs("./data", exist_ok=True)

    if os.path.exists(DB_PATH):
        print("Vector DB already exists")
        return

    vectordb = Chroma.from_texts(
        texts=documents,
        embedding=embedding,
        persist_directory=DB_PATH
    )

    vectordb.persist()

    print("Vector DB created successfully")

