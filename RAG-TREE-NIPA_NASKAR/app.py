"""
app.py
------
Streamlit frontend for the CMaaP AI FAQ Assistant.
Users can upload a PDF, which becomes the vector store database,
then ask questions in a conversational chat interface.

Run with: streamlit run app.py
"""

import streamlit as st
from vector_store import VectorStore
from pdf_loader import PDFLoader
from retriever import Retriever, GuardrailException
from llm_client import LLMClient


# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="CMaaP AI Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===== INITIALIZE SESSION STATE =====
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm_client" not in st.session_state:
    st.session_state.llm_client = LLMClient(provider="openai")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ===== SIDEBAR =====
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/rocket.png", width=60)
    st.title("📚 CMaaP AI Assistant")
    st.markdown("---")

    # PDF Upload Section
    st.header("📄 Upload Your Document")
    st.caption("Upload a PDF file to use as your knowledge base.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF document. The system will extract text, "
             "chunk it, and create a searchable vector database."
    )

    # Chunking settings
    st.markdown("---")
    st.subheader("⚙️ Settings")
    chunk_size = st.slider("Chunk Size (words)", 100, 500, 200, step=50,
                           help="Number of words per chunk. Smaller = more precise, Larger = more context.")
    chunk_overlap = st.slider("Chunk Overlap (words)", 0, 100, 30, step=10,
                              help="Overlap between consecutive chunks to avoid losing info at boundaries.")
    top_k = st.slider("Number of Results (k)", 1, 10, 3,
                      help="How many relevant chunks to retrieve per question.")

    # Process PDF button
    if uploaded_file is not None:
        if st.button("🔄 Process PDF", type="primary", use_container_width=True):
            with st.spinner("📖 Reading and processing PDF..."):
                # Load and chunk the PDF
                loader = PDFLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                pdf_bytes = uploaded_file.read()
                chunks = loader.load_pdf_from_bytes(pdf_bytes, uploaded_file.name)

                # Create vector store and add chunks
                store = VectorStore("pdf_store")
                store.add_bulk(chunks)

                # Create retriever
                retriever = Retriever(
                    vector_store=store,
                    k=top_k,
                    theta_collapse=0.98,
                    pick_threshold=0.05
                )

                # Save to session state
                st.session_state.vector_store = store
                st.session_state.retriever = retriever
                st.session_state.pdf_loaded = True
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chunk_count = len(chunks)
                st.session_state.chat_history = []  # Reset chat on new PDF

            st.success(f"✅ PDF processed successfully!")
            st.info(f"📊 Created {len(chunks)} chunks from '{uploaded_file.name}'")

    # Display current status
    st.markdown("---")
    st.subheader("📊 Status")
    if st.session_state.pdf_loaded:
        st.success(f"📄 **Loaded:** {st.session_state.pdf_name}")
        st.info(f"🧩 **Chunks:** {st.session_state.chunk_count}")
        st.info(f"💬 **Questions asked:** {len(st.session_state.chat_history) // 2}")
    else:
        st.warning("No document loaded. Please upload a PDF.")

    # Clear chat button
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Reset everything button
    if st.session_state.pdf_loaded:
        if st.button("🔄 Reset Everything", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.retriever = None
            st.session_state.pdf_loaded = False
            st.session_state.pdf_name = ""
            st.session_state.chunk_count = 0
            st.session_state.chat_history = []
            st.rerun()


# ===== MAIN CHAT AREA =====
st.title("🚀 CMaaP - AI Document Q&A Assistant")
st.caption("Upload a PDF in the sidebar, then ask questions about it here!")

# Display instructions if no PDF loaded
if not st.session_state.pdf_loaded:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.markdown("Upload a PDF document using the sidebar.")
    with col2:
        st.markdown("### 2️⃣ Process")
        st.markdown("Click 'Process PDF' to create the knowledge base.")
    with col3:
        st.markdown("### 3️⃣ Ask")
        st.markdown("Type your questions and get instant answers!")

    st.markdown("---")
    st.info("👈 Start by uploading a PDF file in the sidebar.")

else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document..."):
                # Retrieve relevant chunks
                result = st.session_state.retriever.retrieve_with_context(prompt)

                if result.get('blocked'):
                    response_text = f"🚫 **Blocked:** {result['llm_context']}"
                elif result['num_results'] == 0:
                    response_text = (
                        "😕 I couldn't find relevant information in the uploaded document "
                        "for your question. Try rephrasing or ask something else."
                    )
                else:
                    # Generate LLM response
                    llm_response = st.session_state.llm_client.generate(
                        prompt, result['llm_context']
                    )
                    response_text = llm_response

                    # Show sources in expander
                    with st.expander(f"📚 Sources ({result['num_results']} chunks found)"):
                        for i, r in enumerate(result['results'], 1):
                            score_pct = r['score'] * 100
                            st.markdown(f"**Chunk {i}** (relevance: {score_pct:.1f}%)")
                            st.caption(r['text'][:300] + "..." if len(r['text']) > 300 else r['text'])
                            st.markdown("---")

                st.markdown(response_text)

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

    # Continue/End session prompt
    if len(st.session_state.chat_history) > 0 and len(st.session_state.chat_history) % 2 == 0:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 Ask another question above, or use the sidebar to upload a new document.")
        with col2:
            if st.button("👋 End Session", use_container_width=True):
                num_questions = len(st.session_state.chat_history) // 2
                st.balloons()
                st.success(
                    f"✅ Session complete! You asked {num_questions} question(s). "
                    f"Thank you for using CMaaP AI Assistant! 🌟"
                )


# ===== FOOTER =====
st.markdown("---")
st.caption("🔧 Built with Streamlit | Vector Store: TF-IDF + Cosine Similarity | Novuz CMaaP Project")