# vectorstore.py - Local vector database operations using PyMuPDF and Scikit-Learn (TF-IDF Vectorizer)
# This file handles document loading, text chunking, local vector representations, and cosine similarity queries.

import os
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    """
    Handles PDF loading, text chunking, generating vector representations locally
    using TF-IDF, and performing Cosine Similarity queries.
    This runs completely locally, requires no PyTorch/TensorFlow, and operates instantly.
    """
    def __init__(self, persist_directory=None):
        # We use a TfidfVectorizer to create vector representations based on term frequencies
        self.vectorizer = TfidfVectorizer()
        self.chunks = []
        self.tfidf_matrix = None

    def load_pdf(self, pdf_path):
        """
        Extracts all raw text from a PDF file page-by-page.
        """
        text = ""
        try:
            with fitz.open(pdf_path) as pdf:
                for page_num in range(pdf.page_count):
                    page = pdf.load_page(page_num)
                    text += page.get_text("text")
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
        return text

    def chunk_text(self, text, chunk_size=1000, overlap=100):
        """
        Splits text into smaller, overlapping chunks to ensure semantic continuity.
        Splits on sentence boundaries first to avoid slicing sentences in half.
        """
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding the sentence exceeds chunk_size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip() + ".")
                
                # Introduce simple sentence-level overlap
                words = current_chunk.split()
                overlap_words = words[-max(1, int(overlap/6)):] if len(words) > 10 else []
                current_chunk = " ".join(overlap_words) + " " + sentence + ". "
            else:
                current_chunk += sentence + ". "

        # Add the remaining text if any
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def embed_and_index(self, pdf_name, chunks):
        """
        Generates TF-IDF vectors for the text chunks locally and indexes them in memory.
        """
        self.chunks = chunks
        if not chunks:
            return 0
            
        # Fit the TF-IDF model on our document chunks and transform them into a term-document matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)
        return len(chunks)

    def retrieve(self, query, top_k=3):
        """
        Given a query, converts it to a TF-IDF vector, computes Cosine Similarity
        against all document chunks, and retrieves the top_k most similar chunks.
        """
        if not self.chunks or self.tfidf_matrix is None:
            return []

        # Convert query to the same TF-IDF space
        query_vector = self.vectorizer.transform([query])

        # Compute cosine similarity between query vector and all chunk vectors
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Sort similarity scores and get top indices
        top_indices = similarities.argsort()[::-1][:top_k]

        retrieved_chunks = []
        for idx in top_indices:
            # Only retrieve if there's a positive match score
            if similarities[idx] > 0.0:
                retrieved_chunks.append(self.chunks[idx])

        # Fallback to first few chunks if no similarity matches
        if not retrieved_chunks and self.chunks:
            retrieved_chunks = self.chunks[:top_k]

        return retrieved_chunks
