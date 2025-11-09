# --------------------------------------------------------------
#  No Document import → works with any LangChain version
# --------------------------------------------------------------
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import json

# 1. Load the JSON file
with open("data/products.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Prepare the two lists FAISS expects
texts = []          # <-- pure strings
metadatas = []      # <-- list of dicts (exactly the original items)

for item in data:
    text = f"{item['title']}: {item['description']} Category: {item['category']}"
    texts.append(text)
    metadatas.append(item)          # keep the whole record as metadata

# 3. Choose an embedding model
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")


# 5. Create the FAISS index
db = FAISS.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,        # optional but super handy for retrieval
)

# 6. Persist locally
db.save_local("vectorstore")

print("Embeddings তৈরি ও সংরক্ষণ সম্পন্ন (চাঙ্কিং ছাড়া)!")