from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load embeddings + FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# Example query
def query_product(query, top_k=2):
    # Retrieve top-k similar products
    results = db.similarity_search(query, k=top_k)

    # Collect metadata info
    output = []
    for res in results:
        info = {
            "title": res.metadata.get("title"),
            "price": res.metadata.get("price"),
            "stock": res.metadata.get("stock"),
            "category": res.metadata.get("category")
        }
        output.append(info)
    return output

# -----------------------------
# 3. Test queries
# -----------------------------
query1 = "লেডিজ কটন শার্ট"
query2 = "Women Shirts"

print("Query 1 →", query_product(query1))
print("Query 2 →", query_product(query2))
