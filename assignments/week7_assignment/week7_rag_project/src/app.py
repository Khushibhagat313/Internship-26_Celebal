# app.py - Main Streamlit web application
# Run with: streamlit run app.py

import streamlit as st
from vectorstore import VectorStore
from chatbot import Chatbot
import tempfile
import os

def main():
    # Page configuration
    st.set_page_config(page_title="Local RAG QA Bot", page_icon="📄", layout="wide")

    st.title("Local RAG Document QA Bot 📄🤖")
    st.write(
        "Upload a PDF document, enter your Groq API Key, and query the document contents. "
        "This application uses local TF-IDF Cosine Similarity vector search "
        "and Groq LLM API for answers."
    )

    # Setup session state parameters
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = None
    if "doc_processed" not in st.session_state:
        st.session_state["doc_processed"] = False
    if "chunks_count" not in st.session_state:
        st.session_state["chunks_count"] = 0

    # Sidebar settings
    with st.sidebar:
        st.header("Settings & API Key 🔑")
        st.write("An API Key is needed only for Groq's fast LLM response generation.")
        
        # input for Groq API key
        groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your gsk_... key from Groq Console")
        st.markdown("[Get your Groq API key here](https://console.groq.com/keys)")

        st.markdown("---")
        st.write("### Pipeline details:")
        st.info(
            "• **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)\n"
            "• **Similarity**: Cosine Similarity Math\n"
            "• **Database**: In-Memory local storage\n"
            "• **LLM Generator**: Groq (`llama3-8b-8192`)"
        )

        # Clear & Start Over button
        if st.session_state["doc_processed"]:
            if st.button("🗑️ Clear & Start Over"):
                st.session_state["vectorstore"] = None
                st.session_state["chatbot"] = None
                st.session_state["chat_history"] = []
                st.session_state["doc_processed"] = False
                st.session_state["chunks_count"] = 0
                st.rerun()

    # File upload section
    uploaded_file = st.file_uploader("📄 Upload a PDF document", type="pdf")

    # Process the PDF file
    if uploaded_file and not st.session_state["doc_processed"]:
        if not groq_api_key:
            st.warning("⚠️ Please enter your Groq API key in the sidebar first!")
        else:
            with st.spinner("Analyzing document... Extracted text, chunking & vectorizing locally. Please wait."):
                try:
                    # Write uploaded PDF content to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    # Initialize local vector store
                    vs = VectorStore()
                    
                    # 1. Load PDF Text
                    raw_text = vs.load_pdf(tmp_path)
                    
                    # 2. Chunk text
                    chunks = vs.chunk_text(raw_text)
                    
                    if not chunks:
                        raise ValueError("No text could be extracted from the PDF. It might be scanned or empty.")
                        
                    # 3. Create vector matrix
                    count = vs.embed_and_index(uploaded_file.name, chunks)
                    
                    # Initialize Groq chatbot client
                    bot = Chatbot(groq_api_key=groq_api_key)

                    # Update session state
                    st.session_state["vectorstore"] = vs
                    st.session_state["chatbot"] = bot
                    st.session_state["doc_processed"] = True
                    st.session_state["chunks_count"] = count

                    # Remove temporary file
                    os.unlink(tmp_path)

                    st.success(f"✅ Document successfully parsed and indexed! Split into {count} semantic chunks.")
                except Exception as e:
                    st.error(f"❌ Failed to parse PDF: {str(e)}")
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    # Document details banner
    if st.session_state["doc_processed"]:
        st.info(f"📚 Loaded document: `{uploaded_file.name}` | total chunks indexed: {st.session_state['chunks_count']}")

    st.markdown("---")
    st.subheader("💬 Document Chatroom")

    # Display chat logs
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # Chat text box
    user_query = st.chat_input("Ask a question about the uploaded document...")

    if user_query:
        if not st.session_state["doc_processed"]:
            st.warning("Please upload a PDF document and add your API key first!")
        else:
            # Display user query
            st.chat_message("user").write(user_query)
            st.session_state["chat_history"].append({"role": "user", "content": user_query})

            with st.spinner("Retrieving facts & writing answer..."):
                # 4. Search local vector database
                retrieved_chunks = st.session_state["vectorstore"].retrieve(user_query, top_k=3)
                
                # 5. Call LLM for grounded answer
                response = st.session_state["chatbot"].respond(user_query, retrieved_chunks)

            # Display response
            st.chat_message("assistant").write(response)
            st.session_state["chat_history"].append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
