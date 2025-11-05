from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
import os

app = FastAPI()

# CORS (Next.js থেকে API কলের অনুমতি)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FAISS + Embeddings লোড
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# LLM সেটআপ (OpenAI / HuggingFace model)
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(api_key=os.getenv("API_KEY"), model="gpt-4"),
    retriever=retriever
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query")
    answer = qa.run(query)
    return {"answer": answer}
