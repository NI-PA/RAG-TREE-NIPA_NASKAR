Setup Instructions
Prerequisites

Python 3.9 or higher
pip

----------------------------------------------------------------
1. Clone / copy the project files
----------------------------------------------------------------
Ensure the following files are in the same directory:
app.py
main.py
llm_client.py
pdf_loader.py
retriever.py
vector_store.py
requirement.txt

----------------------------------------------------------------
2. Install dependencies
----------------------------------------------------------------
pip install -r requirement.txt
This installs:

numpy — vector math
scikit-learn — TF-IDF vectorizer
requests — HTTP calls to LLM APIs
PyMuPDF — PDF text extraction
streamlit — web UI
python-docx — optional Word export

----------------------------------------------------
3. Running the Application
----------------------------------------------------
Streamlit Web UI
streamlit run app.py

Then open your browser to http://localhost:8501.

i) Use the sidebar to upload a PDF file
ii) Adjust chunk size, overlap, and retrieval count (k) as needed
iii) Click Process PDF
iv) Ask questions in the chat input at the bottom