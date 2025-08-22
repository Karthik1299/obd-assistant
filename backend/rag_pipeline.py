from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Paths
KNOWLEDGE_BASE_DIR = "../knowledge_base"
VECTOR_STORE_DIR = "../vector_store"
PDF_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "dtc_list.pdf")

# Prompt template
prompt_template = """
You are an expert automotive technician specializing in OBD-II diagnostics. Your task is to provide a detailed, accurate, and structured response for the given OBD-II code based on the provided context from diagnostic manuals. If the context is insufficient, use your general knowledge but note any limitations. Format the response in markdown with the following sections: Code Definition, Symptoms, Possible Causes, Diagnostic Steps, Repair Procedures, Safety Precautions, Related Codes, Source, and Additional Notes.

**Input OBD-II Code**: {query}

**Context from Knowledge Base**:
{context}

**Instructions**:
1. Use the context to extract relevant information about the code, ensuring accuracy.
2. If the context lacks details, supplement with general OBD-II knowledge and include a disclaimer.
3. Structure the response in markdown with the specified sections.
4. Keep answers concise, actionable, and technician-friendly (e.g., use bullet points, clear steps).
5. For the Source section, cite the document and page from the context metadata if available.
6. If no relevant context is found, state: "Code not found in knowledge base" and provide general advice.

**Response**:
"""

def ingest_pdf():
    """Process PDF and store in ChromaDB."""
    loader = PyPDFLoader(PDF_FILE)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_STORE_DIR)
    vectorstore.persist()
    print("Ingestion complete!")

def get_rag_chain():
    """Load vectorstore and create RAG chain."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Use local gpt2 via transformers (no API key needed)
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = PromptTemplate(template=prompt_template, input_variables=["query", "context"])
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt})
    return chain

# For testing
if __name__ == "__main__":
    ingest_pdf()  # Run once
    # Test: chain = get_rag_chain(); print(chain.run("P0301"))