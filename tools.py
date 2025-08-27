import datasets
import torch
from langchain_core.documents import Document
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

# Load the dataset
guest_dataset = datasets.load_dataset(
    "agents-course/unit3-invitees",
    split="train",
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Convert dataset entries into Document objects
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset
]

@tool
def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    corpus_embeddings = embedder.encode_document(
        [doc.page_content for doc in docs],
        convert_to_tensor=True,
    )
    top_k = min(3, len(docs))
    query_embedding = embedder.encode_query(query, convert_to_tensor=True)

    similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
    scores, indices = torch.topk(similarity_scores, k=top_k)
    results = [docs[i] for i in indices]
    if results:
        return "\n\n".join([doc.page_content for doc in results])
    else:
        return "No matching guest information found."
