# utils/rag_core_setup.py

import os
import re
import torch
import json
from datetime import datetime # 用于日志时间戳

# LangChain 核心组件
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredHTMLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from langchain.docstore.document import Document

# 外部库
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# 导入自定义工具函数 (注意：这里的导入是相对路径，因为 rag_core_setup.py 在 utils 目录下)
from .load_key import load_key
from .text_processing.query_classifier import classify_query, CATEGORIES # 导入 CATEGORIES
from .text_processing.mism_doc_loader import filter_documents
from .text_processing.category_map import category_map
from .model_training.train_eval_utils import set_seed

def initialize_rag_components_cli(verbose: bool = True):
    """
    加载并初始化所有 RAG 组件，适用于命令行或非 Streamlit 环境。
    此函数会打印初始化进度。

    Args:
        verbose (bool): 如果为 True，则打印详细的初始化日志。

    Returns:
        tuple: 包含所有初始化好的 RAG 组件的元组，以及文档和分块数量。
    """
    def log_status(message, level="info"):
        """用于命令行输出状态日志"""
        if verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level.upper()}] {message}")

    log_status("Starting RAG component initialization...", level="info")

    # --- 设备设置 ---
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    set_seed(66)
    log_status(f"Device detected: {device} | Random seed set to 66", level="info")

    # --- API 密钥加载 ---
    log_status("Loading OpenAI API Key...", level="info")
    openai_api_key_val = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key_val:
        # 在 CLI 环境中，使用 load_key 函数，它可能会通过 getpass 提示用户输入
        try:
            openai_api_key_val = load_key("OPENAI_API_KEY")
            log_status("API Key loaded (from keys.json or user input via getpass).", level="success")
        except ValueError as e:
            log_status(f"ERROR: Failed to load API key: {e}. Please ensure it's set correctly.", level="error")
            raise # 抛出错误以停止初始化
    else:
        log_status("API Key loaded from environment variable.", level="success")

    # --- LLM 初始化 ---
    log_status("Initializing LLM...", level="info")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=openai_api_key_val)
    log_status(f"LLM instantiated: {llm.__class__.__name__} with model: {llm.model_name if hasattr(llm, 'model_name') else llm.model}", level="success")

    # --- 嵌入模型初始化 ---
    log_status("Loading embedding model...", level="info")
    embeddings_model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model_name,
        model_kwargs={'device': str(device)}
    )
    log_status(f"Embedding model '{embeddings_model_name}' loaded.", level="success")

    # --- QA 链 Prompt 定义 ---
    QA_CHAIN_PROMPT = ChatPromptTemplate.from_template("""You are an assistant that answers questions only based on provided context about the CMU MISM program.
When the question is about application requirements, look specifically for information in admissions pages or files and **provide a comprehensive and detailed list of all relevant requirements**.\nWhen the question is about career paths, refer to student handbooks or career services sections and **describe them thoroughly**.\nDo not infer from general knowledge if the answer is not in the documents.\n\n{context}\n\nQuestion: {question}\nHelpful Answer:""")
    log_status("QA Prompt template defined.", level="success")

    # --- 灵活问答链构建 ---
    flexible_qa_chain = (
        QA_CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )
    log_status("Flexible QA chain built.", level="success")

    # --- 重排序模型加载 ---
    log_status("Loading re-ranker model...", level="info")
    try:
        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        log_status("Re-ranker model loaded.", level="success")
    except Exception as e:
        log_status(f"ERROR: Failed to load re-ranker model: {e}. Please check network connection.", level="error")
        raise # 抛出错误

    # --- 知识库文档加载与分块 ---
    KNOWLEDGE_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'cmu_mism_docs'))
    CHROMA_PERSIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vectorstore', 'cmu_mism_chroma'))

    log_status(f"Loading documents from: {KNOWLEDGE_BASE_DIR}", level="info")
    documents = []

    # load_and_tag_documents 函数 (复制自 Notebook)
    def load_and_tag_documents(path, glob_pattern, loader_cls, category_key):
        if not os.path.exists(path):
            log_status(f"WARNING: Directory not found: {path}.", level="warning")
            return []
        loader = DirectoryLoader(path, glob=glob_pattern, loader_cls=loader_cls)
        try:
            raw_docs = loader.load()
            for doc in raw_docs:
                rel_path = os.path.relpath(doc.metadata.get('source', ''), KNOWLEDGE_BASE_DIR).replace("\\", "/")
                doc.metadata["source"] = rel_path
                doc.metadata["category"] = category_map.get(rel_path, category_map.get(category_key, "unknown"))
            return raw_docs
        except Exception as e:
            log_status(f"WARNING: Failed to load {category_key.upper()} documents from {path}. Error: {e}", level="warning")
            return []

    html_docs = load_and_tag_documents(os.path.join(KNOWLEDGE_BASE_DIR, 'websites_pages'), "**/*.html", UnstructuredHTMLLoader, 'websites_pages')
    documents.extend(html_docs)
    pdf_docs = load_and_tag_documents(os.path.join(KNOWLEDGE_BASE_DIR, 'handbooks'), "**/*.pdf", PyPDFLoader, 'handbooks')
    documents.extend(pdf_docs)
    forums_txt_docs = load_and_tag_documents(os.path.join(KNOWLEDGE_BASE_DIR, 'forums'), "**/*.txt", TextLoader, 'forums')
    documents.extend(forums_txt_docs)
    text_info_docs = load_and_tag_documents(os.path.join(KNOWLEDGE_BASE_DIR, 'text_info'), "**/*.txt", TextLoader, 'text_info')
    documents.extend(text_info_docs)
    
    documents, _ = filter_documents(documents)
    log_status(f"Original documents loaded and cleaned: {len(documents)} valid documents.", level="success")

    log_status("Performing semantic chunking...", level="info")
    def simple_tokenizer(text): # 确保 simple_tokenizer 在此作用域内可见
        return re.findall(r'\w+', text.lower())
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    chunks = text_splitter.split_documents(documents)
    log_status(f"Documents chunked: {len(chunks)} chunks.", level="success")

    # --- BM25 索引构建 ---
    log_status("Building BM25 index...", level="info")
    tokenized_corpus = [simple_tokenizer(doc.page_content) for doc in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    bm25_doc_list = chunks # BM25 索引对应的原始文档列表
    log_status("BM25 index built.", level="success")

    # --- ChromaDB 加载/创建 ---
    log_status(f"Loading or creating vector database at: {CHROMA_PERSIST_DIR}...", level="info")
    try:
        if os.path.exists(CHROMA_PERSIST_DIR) and len(os.listdir(CHROMA_PERSIST_DIR)) > 0:
            vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
            try: # Test if usable
                _ = vectordb.similarity_search("test query", k=1)
                log_status("Vector database loaded successfully.", level="success")
            except Exception as e:
                log_status(f"WARNING: Vector database found but unusable ({e}), attempting to rebuild.", level="warning")
                import shutil
                shutil.rmtree(CHROMA_PERSIST_DIR)
                os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
                vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PERSIST_DIR)
                log_status("Vector database rebuilt successfully.", level="success")
        else:
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PERSIST_DIR)
            log_status("Vector database created successfully.", level="success")
    except Exception as e:
        log_status(f"ERROR: Vector database operation failed: {e}.", level="error")
        raise # 抛出错误

    log_status("All RAG components initialized!", level="success")
    # 返回所有组件，包括 simple_tokenizer，以及 documents 和 chunks
    return llm, embeddings, QA_CHAIN_PROMPT, flexible_qa_chain, vectordb, bm25_index, bm25_doc_list, reranker_model, simple_tokenizer, documents, chunks