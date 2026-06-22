# chatbot.py - Handles generation using Groq's LLM API
# Takes retrieved context chunks and queries Groq's fast llama3 model.

import os
from groq import Groq

class Chatbot:
    """
    Chatbot client using the Groq API to run ultra-fast inference
    and answer questions using retrieved document context.
    """
    def __init__(self, groq_api_key=None):
        # Read API key from parameter or environment variables
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API Key is missing. Please provide it via constructor or set GROQ_API_KEY.")
        
        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        # Using llama-3.1-8b-instant for high speed and solid reasoning
        self.model = "llama-3.1-8b-instant"

    def respond(self, query, retrieved_chunks):
        """
        Creates a grounded chat completion prompt using retrieved context documents,
        sends it to Groq, and returns the response.
        """
        if not retrieved_chunks:
            return "Sorry, I couldn't find any relevant details in the document to answer your question."

        # Format retrieved context
        context_block = "\n\n".join([f"[Context block {i+1}]:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])

        # Construct system and user prompts
        system_prompt = (
            "You are a helpful and factual document question-answering assistant. "
            "Your task is to answer the user's question using ONLY the provided document context. "
            "If the document context does not contain the answer, politely state that you cannot find the answer in the document. "
            "Do not make up facts or use external knowledge. Keep answers clear, accurate, and concise."
        )

        user_prompt = (
            f"Here is the context retrieved from the document:\n"
            f"==================================================\n"
            f"{context_block}\n"
            f"==================================================\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        try:
            # Query Groq API
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.2,  # Low temperature for strict adherence to facts
                max_tokens=512
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error while calling Groq API: {str(e)}"
