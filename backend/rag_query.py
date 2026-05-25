from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# SAME DB PATH (IMPORTANT)
# ==========================================

DB_PATH = "./data/chroma_db"

# ==========================================
# Embedding model (must be SAME as before)
# ==========================================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# LOAD VECTOR DATABASE
# ==========================================

def load_vector_db():

    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding
    )

    return vectordb

# ==========================================
# SEMANTIC QUESTION ANSWERING
# ==========================================

def ask_question(question):

    vectordb = load_vector_db()

    # STEP 1: semantic search
    docs = vectordb.similarity_search(question, k=2)

    # STEP 2: build context
    context = "\n".join([doc.page_content for doc in docs])

    

    # STEP 3: simple reasoning (no LLM yet)
    answer = f"""
    Question: {question}

    Retrieved Context:
    {context}

    Final Answer: ok i will get 
    """

    return answer

