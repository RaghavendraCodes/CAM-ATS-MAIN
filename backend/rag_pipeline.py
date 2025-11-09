import os
import pickle
from operator import itemgetter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 🔹 Load once globally
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", task="text-generation")
chat_model = ChatHuggingFace(llm=llm)

# ✅ Prompt (from your example)
qa_prompt = PromptTemplate(
    template="""
    You are a helpful AI assistant.
    Answer only using the provided context.
    If the context is insufficient, say "I don't know based on this video."

    Context:
    {context}

    Question: {question}
    """,
    input_variables=["context", "question"]
)

def get_vector_store(video_id: str, transcript: str):
    """Cache & reuse FAISS DB for each video."""
    os.makedirs("cache", exist_ok=True)
    cache_path = f"cache/{video_id}_faiss.pkl"

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = text_splitter.create_documents([transcript])
    vector_store = FAISS.from_documents(docs, embedding=embedding_model)

    with open(cache_path, "wb") as f:
        pickle.dump(vector_store, f)

    return vector_store


def get_answer(transcript: str, question: str, video_id: str) -> str:
    """RAG Chain: Retrieve → Prompt LLM → Return Answer."""
    vector_store = get_vector_store(video_id, transcript)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel({
            "context": itemgetter("question") | retriever | RunnableLambda(format_docs),
            "question": itemgetter("question")
        })
        | qa_prompt
        | chat_model
        | StrOutputParser()
    )

    return chain.invoke({"question": question})
