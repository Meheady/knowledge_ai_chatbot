from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = DirectoryLoader("data/", glob="*.json")
docs = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(docs, embeddings)
db.save_local("vectorstore")

print("✅ Embeddings তৈরি ও সংরক্ষণ সম্পন্ন (চাঙ্কিং ছাড়া)!")
