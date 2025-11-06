import os
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
# 1️⃣ Load .env file
load_dotenv()


token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)


# Load embeddings + FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# 1️⃣ FAISS retrieval
query = "পাঞ্জাবি (পুরুষ) price কত?"
results = db.similarity_search(query, k=3)  # top 3 context

# Combine context into one string
context_text = "\n".join([
    f"- {r.metadata['title']}: {r.metadata['description']} | Price: {r.metadata['price']} | Stock: {r.metadata['stock']}"
    for r in results
])

print("Context for LLM:\n", context_text)
# 2️⃣ Prepare messages for GitHub GPT-5
messages = [
    {"role": "system", "content": "You are a helpful assistant for an e-commerce chatbot."},
    {"role": "user", "content": f"Answer the question using this context:\n{context_text}\n\nQuestion: {query}"}
]

# 3️⃣ Call GitHub GPT-5
response = client.chat.completions.create(
    model=model,
    messages=messages
)

answer = response.choices[0].message.content
print(answer)

