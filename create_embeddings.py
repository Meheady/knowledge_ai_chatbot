from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. ডেটা লোড করো
loader = DirectoryLoader("data/", glob="*.json")
docs = loader.load()

# 2. চাঙ্কিং (বড় টেক্সটকে ভাগ করা)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(docs)

# 3. এমবেডিং
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. FAISS ডাটাবেজে সেভ করা
db = FAISS.from_documents(texts, embeddings)
db.save_local("vectorstore")

print("✅ Embeddings তৈরি ও সংরক্ষণ সম্পন্ন!")
