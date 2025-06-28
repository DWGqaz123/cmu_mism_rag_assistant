# main.py

import os
import sys
from datetime import datetime

# 确保 sys.path 包含项目根目录，以便导入 utils
# 假设 main.py 在项目根目录
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 从新的共享模块导入 RAG 组件初始化函数
from utils.rag_core_setup import initialize_rag_components_cli

# 从新的共享模块导入 RAG 核心逻辑函数
from utils.rag_pipeline_logic import get_rag_answer

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- Starting CMU MISM RAG Assistant (CLI Mode) ---")

    # 1. 初始化 RAG 组件
    try:
        llm, embeddings, QA_CHAIN_PROMPT, flexible_qa_chain, vectordb, bm25_index, bm25_doc_list, reranker_model, simple_tokenizer_func = initialize_rag_components_cli(verbose=True)
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- RAG Components Initialized Successfully! ---")
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- ERROR: Failed to initialize RAG components: {e} ---")
        print("Please ensure your API key is set (environment variable or keys.json) and knowledge base is built correctly.")
        sys.exit(1)

    # 2. 命令行交互循环
    print("\n--- Ready to answer your questions about CMU MISM. ---")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        question = input("\nYour Question: ").strip()
        if question.lower() in ['exit', 'quit']:
            print("Exiting RAG Assistant. Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing your question...")
        try:
            # 调用 RAG 核心逻辑
            answer_content, source_docs_for_display, metadata_info, retrieved_context_str = get_rag_answer(
                question, vectordb, llm, flexible_qa_chain, bm25_index, bm25_doc_list, reranker_model, simple_tokenizer_func
            )

            print("\n--- Answer ---")
            print(answer_content)

            print("\n--- RAG Processing Details ---")
            print(f"  Query Category: {metadata_info.get('query_category', 'N/A')}")
            print(f"  Retrieved Docs (Before Re-rank): {metadata_info.get('retrieved_doc_count_before_rerank', 'N/A')}")
            print(f"  Retrieved Docs (After Re-rank): {metadata_info.get('retrieved_doc_count_after_rerank', 'N/A')}")
            print("\n  LLM Prompt Context (Truncated):")
            display_context = retrieved_context_str[:1500] + "..." if len(retrieved_context_str) > 1500 else retrieved_context_str
            print(display_context)

            print("\n--- Source Documents ---")
            if source_docs_for_display:
                for i, doc in enumerate(source_docs_for_display[:5]): # 限制显示前5个文档
                    source = doc.metadata.get('source', 'N/A')
                    page = doc.metadata.get('page', 'N/A')
                    category = doc.metadata.get('category', 'unknown')
                    print(f"  --- Document {i+1} ---")
                    print(f"  Source: {source}")
                    print(f"  Category: {category}")
                    if page != 'N/A':
                        print(f"  Page: {page}")
                    print(f"  Content Preview: {doc.page_content[:200]}...")
                    print("  ---")
            else:
                print("  No relevant source documents found.")

        except Exception as e:
            print(f"\n--- ERROR: An error occurred while processing your question: {e} ---")
            print("Please check your input or the system's configuration.")

if __name__ == "__main__":
    main()