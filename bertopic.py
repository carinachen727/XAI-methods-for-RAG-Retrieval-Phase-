
pip install bertopic lxml[html_clean] newspaper3k sentence-transformers torch

from bertopic import BERTopic
from newspaper import Article
from sentence_transformers import SentenceTransformer, util
# List of URLs (your document links)
urls = [
    "https://www.cdc.gov/reproductive-health/depression/",
    "https://medlineplus.gov/depression.html",
    "https://medlineplus.gov/ency/article/000945.htm",
    "https://www.cdc.gov/tobacco/campaign/tips/diseases/depression-anxiety.html",
    "https://pubmed.ncbi.nlm.nih.gov/10318745/"
    "https://www.ncbi.nlm.nih.gov/books/NBK279282/"
]
documents = []  # This will store the article texts
for url in urls:
    try:
        article = Article(url)
        article.download()
        article.parse()
        documents.append(article.text)
    except Exception as e:
        print(f"Failed to process {url}: {e}")
topics = ["symptoms", "depression", "causes"]
# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Encode documents and topics
doc_embeddings = model.encode(documents, convert_to_tensor=True)
topic_embeddings = model.encode(topics, convert_to_tensor=True)
# Compute similarity
cosine_scores = util.cos_sim(doc_embeddings, topic_embeddings)
# Define how many top topics you want to return
top_n = len(topics)

# Display all matched topics for each document
for idx, doc in enumerate(documents):
    print(f"\nDocument {idx+1}:")
    topic_scores = cosine_scores[idx]

    # Sort topics based on cosine similarity (descending order)
    top_indices = topic_scores.argsort(descending=True)

    # Display the top N topics for each document
    for i in range(top_n):
        top_idx = top_indices[i]
        print(f"Topic {i+1}: '{topics[top_idx]}' (Score: {topic_scores[top_idx]:.2f})")
