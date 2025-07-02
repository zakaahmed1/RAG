from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# Step 1: Load and chunk the document
loader = TextLoader("ExampleCompanyPolicy.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Step 2: Embed chunks and create vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
retriever = vectorstore.as_retriever()

# Step 3: Load a small, fast Hugging Face model
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Step 4: Simple function to run prompt with retrieved context
def simple_llm(prompt: str) -> str:
    result = generator(prompt, max_new_tokens=150)
    return result[0]["generated_text"]

# Step 5: Interactive question loop
while True:
    query = input("\nAsk a question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"Answer the question based only on the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = simple_llm(prompt)
    
    print("\n--- Answer ---")
    print(answer.strip())
