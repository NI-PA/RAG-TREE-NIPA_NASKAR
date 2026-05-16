"""
main.py
-------
Command-line version of the system (for testing without Streamlit).
Also serves as a reference/demo showing the full pipeline.

For the Streamlit UI version, run: streamlit run app.py
"""

from __future__ import annotations
import sys
import os

from vector_store import VectorStore
from pdf_loader import PDFLoader
from retriever import Retriever, GuardrailException
from llm_client import LLMClient


def run_cli():
    """
    Command-line interactive version.
    User provides a PDF path, system processes it, then chat begins.
    """

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                                                                  ║")
    print("║          🚀 CMaaP AI Assistant — CLI Mode 🚀                    ║")
    print("║          Upload a PDF & Ask Questions                            ║")
    print("║                                                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    # Step 1: Get PDF path from user
    print("📄 Please provide the path to your PDF file:")
    pdf_path = input("   PDF path: ").strip()

    if not os.path.exists(pdf_path):
        print(f"   ❌ File not found: {pdf_path}")
        print("   Please check the path and try again.")
        return

    # Step 2: Process the PDF
    print(f"\n[⏳] Processing '{pdf_path}'...")
    loader = PDFLoader(chunk_size=200, chunk_overlap=30)
    chunks = loader.load_pdf(pdf_path)
    print(f"[✅] Extracted {len(chunks)} chunks from the PDF.")

    # Step 3: Create vector store
    print("[⏳] Building vector store...")
    store = VectorStore("pdf_store")
    store.add_bulk(chunks)
    print(f"[✅] Vector store ready with {store.size()} entries.")

    # Step 4: Initialize retriever and LLM
    retriever = Retriever(vector_store=store, k=3, theta_collapse=0.98, pick_threshold=0.05)
    llm = LLMClient(provider="openai")
    print("[✅] System initialized!\n")

    # Step 5: Conversation loop
    question_count = 0

    while True:
        print("═" * 50)
        question_count += 1

        user_query = input(f"\n📝 Question #{question_count}: ").strip()

        if not user_query:
            print("   ⚠️  Please enter a valid question.")
            question_count -= 1
            continue

        if user_query.lower() in ('quit', 'exit', 'bye'):
            break

        # Retrieve
        print("\n   [⏳] Searching...")
        result = retriever.retrieve_with_context(user_query)

        if result.get('blocked'):
            print(f"\n   🚫 BLOCKED: {result['llm_context']}")
        elif result['num_results'] == 0:
            print("\n   😕 No relevant information found. Try rephrasing.")
        else:
            print(f"\n   ✅ Found {result['num_results']} relevant chunk(s):")
            for i, r in enumerate(result['results'], 1):
                print(f"      [{i}] (score: {r['score']:.4f}) {r['text'][:80]}...")

            # LLM response
            print("\n   🤖 AI Answer:")
            llm_response = llm.generate(user_query, result['llm_context'])
            print(f"   {llm_response}")

        # Ask to continue
        print()
        print("─" * 50)
        choice = input("\n🔄 Do you have another question? (yes/no): ").strip().lower()
        if choice in ('no', 'n'):
            break

    # End session
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print(f"║   👋 Session complete! You asked {question_count} question(s).          ║")
    print("║   Thank you for using CMaaP AI Assistant! 🌟                    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    run_cli()