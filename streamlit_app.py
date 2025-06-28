# streamlit_app.py

# ==============================================================================
# 1. 必要的库导入
# ==============================================================================
import streamlit as st
import os
import re
import torch
import json
import time
from datetime import datetime

# LangChain 核心组件
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredHTMLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 语义分块器
from langchain_experimental.text_splitter import SemanticChunker

# 重排序模型
from sentence_transformers import CrossEncoder

# BM25 稀疏检索
from rank_bm25 import BM25Okapi

# LangChain Document 类
from langchain.docstore.document import Document

# 导入自定义工具函数
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils')))

# 导入自定义工具
try:
    from utils.load_key import load_key
    from utils.text_processing.query_classifier import classify_query, CATEGORIES
    from utils.text_processing.mism_doc_loader import filter_documents
    from utils.text_processing.category_map import category_map
    from utils.model_training.train_eval_utils import set_seed
except ImportError as e:
    st.error(f"❌ 无法导入自定义工具模块：{e}。请检查 'utils' 目录及其子目录是否存在于正确路径。")
    st.stop()

# ==============================================================================
# 2. 页面配置和样式设置
# ==============================================================================

# 页面配置
st.set_page_config(
    page_title="CMU MISM RAG Assistant", 
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 主题色彩定义 */
    :root {
        --cmu-red: #c41e3a;
        --cmu-dark-red: #8b0000;
        --cmu-gold: #ffb81c;
        --dark-bg: #0e1117;
        --card-bg: #1e1e1e;
        --text-primary: #ffffff;
        --text-secondary: #b0b0b0;
        --accent-blue: #4a90e2;
        --success-green: #00d4aa;
        --warning-orange: #ff6b35;
    }

    /* 隐藏默认的Streamlit样式 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 主容器样式 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* 标题区域美化 */
    .main-header {
        background: linear-gradient(135deg, var(--cmu-red) 0%, var(--cmu-dark-red) 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(196, 30, 58, 0.3);
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    /* 卡片样式 */
    .info-card {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* 聊天消息样式 */
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, var(--accent-blue) 0%, #357abd 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: var(--card-bg);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-right: 2rem;
    }
    
    /* 状态指示器 */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .status-success {
        background: rgba(0, 212, 170, 0.2);
        color: var(--success-green);
        border: 1px solid var(--success-green);
    }
    
    .status-info {
        background: rgba(74, 144, 226, 0.2);
        color: var(--accent-blue);
        border: 1px solid var(--accent-blue);
    }
    
    .status-warning {
        background: rgba(255, 107, 53, 0.2);
        color: var(--warning-orange);
        border: 1px solid var(--warning-orange);
    }
    
    /* 侧边栏美化 */
    .sidebar .sidebar-content {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* 按钮美化 */
    .stButton > button {
        background: linear-gradient(135deg, var(--cmu-red) 0%, var(--cmu-dark-red) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(196, 30, 58, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(196, 30, 58, 0.4);
    }
    
    /* 展开器美化 */
    .streamlit-expanderHeader {
        background: var(--card-bg);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* 文本区域美化 */
    .stTextArea > div > div > textarea {
        background: var(--card-bg);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: var(--text-primary);
    }
    
    /* 进度条美化 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--cmu-red) 0%, var(--cmu-gold) 100%);
    }
    
    /* 指标卡片 */
    .metric-card {
        background: var(--card-bg);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--cmu-red);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* 动画效果 */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .user-message {
            margin-left: 0;
        }
        
        .assistant-message {
            margin-right: 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. RAG 组件加载与缓存
# ==============================================================================
@st.cache_resource
def load_rag_components():
    """加载并初始化所有RAG组件"""
    
    # 创建加载进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(value, text):
        progress_bar.progress(value)
        status_text.markdown(f"<div class='status-indicator status-info'>🔄 {text}</div>", unsafe_allow_html=True)
    
    update_progress(0.1, "初始化设备和环境...")
    
    # 设备设置
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    set_seed(66)
    
    update_progress(0.2, "加载API密钥...")
    
    # API密钥加载
    openai_api_key_val = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key_val:
        if "OPENAI_API_KEY" in st.secrets:
            openai_api_key_val = st.secrets["OPENAI_API_KEY"]
        else:
            keys_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'keys.json'))
            if os.path.exists(keys_file_path):
                try:
                    with open(keys_file_path, "r", encoding="utf-8") as f:
                        keys_data = json.load(f)
                        if "OPENAI_API_KEY" in keys_data and keys_data["OPENAI_API_KEY"]:
                            openai_api_key_val = keys_data["OPENAI_API_KEY"]
                except Exception as e:
                    st.error(f"❌ 从 'keys.json' 加载密钥失败：{e}")
                    st.stop()
            else:
                st.error("❌ 'OPENAI_API_KEY' 未设置")
                st.stop()
    
    if not openai_api_key_val:
        st.error("❌ 无法获取 OpenAI API 密钥")
        st.stop()
    
    update_progress(0.3, "初始化语言模型...")
    
    # LLM初始化
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=openai_api_key_val)
    
    update_progress(0.4, "加载嵌入模型...")
    
    # 嵌入模型
    embeddings_model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model_name,
        model_kwargs={'device': str(device)}
    )
    
    update_progress(0.5, "设置问答链...")
    
    # QA链设置
    QA_CHAIN_PROMPT = ChatPromptTemplate.from_template("""You are an assistant that answers questions only based on provided context about the CMU MISM program.
When the question is about application requirements, look specifically for information in admissions pages or files and **provide a comprehensive and detailed list of all relevant requirements**.
When the question is about career paths, refer to student handbooks or career services sections and **describe them thoroughly**.
Do not infer from general knowledge if the answer is not in the documents.

{context}

Question: {question}
Helpful Answer:""")
    
    flexible_qa_chain = (
        QA_CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )
    
    update_progress(0.6, "加载重排序模型...")
    
    # 重排序模型
    try:
        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    except Exception as e:
        st.error(f"❌ 重排序模型加载失败：{e}")
        st.stop()
    
    update_progress(0.7, "加载知识库文档...")
    
    # 知识库文档加载
    KNOWLEDGE_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'cmu_mism_docs'))
    CHROMA_PERSIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'vectorstore', 'cmu_mism_chroma'))
    
    documents = []
    
    def load_and_tag_documents(path, glob_pattern, loader_cls, category_key):
        doc_list = []
        if os.path.exists(path):
            loader = DirectoryLoader(path, glob=glob_pattern, loader_cls=loader_cls)
            try:
                raw_docs = loader.load()
                for doc in raw_docs:
                    rel_path = os.path.relpath(doc.metadata.get('source', ''), KNOWLEDGE_BASE_DIR).replace("\\", "/")
                    doc.metadata["source"] = rel_path
                    doc.metadata["category"] = category_map.get(rel_path, category_map.get(category_key, "unknown"))
                return raw_docs
            except Exception as e:
                return []
        return []
    
    # 加载各类文档
    html_docs_path = os.path.join(KNOWLEDGE_BASE_DIR, 'websites_pages')
    html_docs = load_and_tag_documents(html_docs_path, "**/*.html", UnstructuredHTMLLoader, 'websites_pages')
    documents.extend(html_docs)
    
    pdf_docs_path = os.path.join(KNOWLEDGE_BASE_DIR, 'handbooks')
    pdf_docs = load_and_tag_documents(pdf_docs_path, "**/*.pdf", PyPDFLoader, 'handbooks')
    documents.extend(pdf_docs)
    
    forums_txt_path = os.path.join(KNOWLEDGE_BASE_DIR, 'forums')
    forums_txt_docs = load_and_tag_documents(forums_txt_path, "**/*.txt", TextLoader, 'forums')
    documents.extend(forums_txt_docs)
    
    text_info_path = os.path.join(KNOWLEDGE_BASE_DIR, 'text_info')
    text_info_docs = load_and_tag_documents(text_info_path, "**/*.txt", TextLoader, 'text_info')
    documents.extend(text_info_docs)
    
    # 文档清洗
    documents, _ = filter_documents(documents)
    
    update_progress(0.8, "执行语义分块...")
    
    # 语义分块
    def simple_tokenizer(text):
        return re.findall(r'\w+', text.lower())
    
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    chunks = text_splitter.split_documents(documents)
    
    update_progress(0.85, "构建BM25索引...")
    
    # BM25索引
    tokenized_corpus = [simple_tokenizer(doc.page_content) for doc in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    bm25_doc_list = chunks
    
    update_progress(0.9, "构建向量数据库...")
    
    # 向量数据库
    try:
        if os.path.exists(CHROMA_PERSIST_DIR) and len(os.listdir(CHROMA_PERSIST_DIR)) > 0:
            vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
            try:
                _ = vectordb.similarity_search("test", k=1)
            except Exception as e:
                import shutil
                shutil.rmtree(CHROMA_PERSIST_DIR)
                os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
                vectordb = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=CHROMA_PERSIST_DIR
                )
        else:
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
    except Exception as e:
        st.error(f"❌ 向量数据库操作失败：{e}")
        st.stop()
    
    update_progress(1.0, "完成初始化!")
    
    # 清理进度指示器
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    return llm, embeddings, QA_CHAIN_PROMPT, flexible_qa_chain, vectordb, bm25_index, bm25_doc_list, reranker_model, simple_tokenizer, len(documents), len(chunks)

# ==============================================================================
# 4. RAG 核心逻辑函数
# ==============================================================================
def get_rag_answer(question: str, vectordb, llm, flexible_qa_chain, bm25_index, bm25_doc_list, reranker_model, simple_tokenizer_func):
    """执行RAG流程并返回答案"""
    response_metadata = {}
    
    # 查询分类
    predicted_category = classify_query(question)
    response_metadata['query_category'] = predicted_category
    
    # 构建过滤条件
    filter_condition = {}
    if predicted_category != "unknown":
        filter_condition = {"category": predicted_category}
    
    # 混合检索
    vector_results = vectordb.similarity_search(
        query=question,
        k=10,
        filter=filter_condition
    )
    
    # BM25检索
    tokenized_query = simple_tokenizer_func(question)
    bm25_results = []
    if tokenized_query:
        doc_scores = bm25_index.get_scores(tokenized_query)
        sorted_doc_indices_and_scores = sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True)
        top_n_bm25 = 10
        for i, (doc_original_index, score_value) in enumerate(sorted_doc_indices_and_scores):
            if i >= top_n_bm25:
                break
            bm25_results.append(bm25_doc_list[doc_original_index])
    
    # 合并和去重
    unique_docs_map = {}
    for doc in vector_results:
        unique_docs_map[doc.page_content] = doc
    for doc in bm25_results:
        if doc.page_content not in unique_docs_map:
            unique_docs_map[doc.page_content] = doc
    retrieved_docs = list(unique_docs_map.values())
    
    response_metadata['retrieved_doc_count_before_rerank'] = len(retrieved_docs)
    
    # 重排序
    if retrieved_docs:
        pairs = [(question, doc.page_content) for doc in retrieved_docs]
        rerank_scores = reranker_model.predict(pairs)
        ranked_docs_with_scores = sorted(zip(retrieved_docs, rerank_scores), key=lambda x: x[1], reverse=True)
        retrieved_docs = [doc for doc, score in ranked_docs_with_scores]
    
    response_metadata['retrieved_doc_count_after_rerank'] = len(retrieved_docs)
    
    # 准备上下文
    context_strings = [doc.page_content for doc in retrieved_docs]
    context_str = "\n\n".join(context_strings)
    
    # 调用LLM
    if not context_str.strip():
        response_content = "抱歉，根据我的知识库，未能找到相关信息来回答您的问题。"
        source_docs_info = []
    else:
        response_content = flexible_qa_chain.invoke({
            "context": context_str,
            "question": question
        })
        source_docs_info = retrieved_docs
    
    return response_content, source_docs_info, response_metadata, context_str

# ==============================================================================
# 5. 主界面
# ==============================================================================

# 主标题区域
st.markdown("""
<div class="main-header fade-in-up">
    <h1>🎓 CMU MISM RAG 智能问答助手</h1>
    <p>欢迎使用卡内基梅隆大学信息系统管理项目智能问答系统</p>
</div>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown("### 📊 系统状态")
    
    # 加载RAG组件
    with st.spinner("🚀 正在初始化系统..."):
        try:
            llm, embeddings, QA_CHAIN_PROMPT, flexible_qa_chain, vectordb, bm25_index, bm25_doc_list, reranker_model, simple_tokenizer_func, total_docs, total_chunks = load_rag_components()
            
            # 显示系统指标
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_docs}</div>
                    <div class="metric-label">文档总数</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_chunks}</div>
                    <div class="metric-label">文档块数</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<div class='status-indicator status-success'>✅ 系统就绪</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ 系统初始化失败: {e}")
            st.stop()
    
    st.markdown("---")
    st.markdown("### 📚 知识库覆盖")
    st.markdown("""
    - 🌐 **官方网站**: 项目介绍、课程信息
    - 📖 **学生手册**: 详细指导文档
    - 💬 **论坛讨论**: 学生经验分享
    - 📄 **文本资料**: 补充信息材料
    """)
    
    st.markdown("---")
    st.markdown("### 💡 使用建议")
    st.markdown("""
    - 询问**申请要求**和**截止日期**
    - 了解**课程设置**和**学习内容**
    - 探索**就业前景**和**职业发展**
    - 查询**学费费用**和**奖学金**
    """)

# 主聊天界面
# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state['message_count'] = 0

# 显示欢迎消息
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="info-card fade-in-up">
        <h3>👋 欢迎使用CMU MISM智能助手</h3>
        <p>我可以帮助您了解卡内基梅隆大学信息系统管理硕士项目的相关信息。请随时向我提问！</p>
        <p><strong>示例问题：</strong></p>
        <ul>
            <li>CMU MISM项目的申请要求是什么？</li>
            <li>这个项目的课程设置如何？</li>
            <li>毕业后的就业前景怎样？</li>
            <li>学费和生活费大概需要多少？</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 显示聊天历史
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message fade-in-up">
            <strong>🙋‍♂️ 您：</strong><br>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message fade-in-up">
            <strong>🤖 助手：</strong><br>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)

# 用户输入
if prompt := st.chat_input("💬 请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 显示用户消息
    st.markdown(f"""
    <div class="chat-message user-message fade-in-up">
        <strong>🙋‍♂️ 您：</strong><br>
        {prompt}
    </div>
    """, unsafe_allow_html=True)
    
    # 生成回答
    with st.spinner("🤔 正在思考中..."):
        try:
            answer_content, source_docs_for_ui, metadata_info, retrieved_context_str = get_rag_answer(
                prompt, vectordb, llm, flexible_qa_chain, bm25_index, bm25_doc_list, reranker_model, simple_tokenizer_func
            )
            
            # 添加助手消息
            st.session_state.messages.append({"role": "assistant", "content": answer_content})
            
            # 显示助手回答
            st.markdown(f"""
            <div class="chat-message assistant-message fade-in-up">
                <strong>🤖 助手：</strong><br>
                {answer_content}
            </div>
            """, unsafe_allow_html=True)
            
            # 显示处理详情
            with st.expander("🔍 处理详情", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{metadata_info.get('query_category', 'N/A')}</div>
                        <div class="metric-label">查询类别</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{metadata_info.get('retrieved_doc_count_before_rerank', 'N/A')}</div>
                        <div class="metric-label">检索文档数</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{metadata_info.get('retrieved_doc_count_after_rerank', 'N/A')}</div>
                        <div class="metric-label">重排序后</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("#### 📝 上下文预览")
                if len(retrieved_context_str) > 1000:
                    st.text_area(
                        "上下文内容",
                        retrieved_context_str[:1000] + "...",
                        height=200,
                        disabled=True,
                        key=f"context_{st.session_state.message_count}"
                    )
                else:
                    st.text_area(
                        "上下文内容",
                        retrieved_context_str,
                        height=200,
                        disabled=True,
                        key=f"context_{st.session_state.message_count}"
                    )
            
            # 显示来源文档
            if source_docs_for_ui:
                with st.expander("📚 参考来源", expanded=False):
                    for i, doc in enumerate(source_docs_for_ui[:5]):  # 限制显示前5个文档
                        source = doc.metadata.get('source', 'N/A')
                        page = doc.metadata.get('page', 'N/A')
                        category = doc.metadata.get('category', 'unknown')
                        
                        # 文档卡片
                        st.markdown(f"""
                        <div class="info-card">
                            <h4>📄 文档 {i+1}</h4>
                            <p><strong>来源:</strong> <code>{source}</code></p>
                            <p><strong>类别:</strong> <span class="status-indicator status-info">{category}</span></p>
                            {f'<p><strong>页码:</strong> {page}</p>' if page != 'N/A' else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 内容预览
                        content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                        st.text_area(
                            f"内容预览",
                            content_preview,
                            height=120,
                            disabled=True,
                            key=f"source_doc_preview_{st.session_state.message_count}_{i}"
                        )
                        
                        if i < len(source_docs_for_ui[:5]) - 1:
                            st.markdown("---")
            
        except Exception as e:
            st.error(f"❌ 处理问题时出现错误: {e}")
            st.markdown(f"""
            <div class="chat-message assistant-message fade-in-up">
                <strong>🤖 助手：</strong><br>
                抱歉，处理您的问题时出现了技术错误。请稍后重试，或者尝试重新表述您的问题。
            </div>
            """, unsafe_allow_html=True)
    
    # 增加消息计数器
    st.session_state['message_count'] += 1

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: var(--text-secondary);">
    <p>🎓 <strong>CMU MISM RAG 智能助手</strong> | 基于先进的检索增强生成技术</p>
    <p>💡 如有问题或建议，请联系作者</p>
    <p>email:dwgqaz123@outlook.com</p>
    <p>github: <a href="https://github.com/DWGqaz123/cmu_mism_rag_assistant" target="_blank" style="color: var(--accent-blue);">DWGqaz123/cmu_mism_rag_assistant</a></p>
    <p style="font-size: 0.8rem; opacity: 0.7;">
        ⚠️ 本助手提供的信息仅供参考，具体申请要求请以官方网站为准
    </p>
</div>
""", unsafe_allow_html=True)

# 添加返回顶部按钮
st.markdown("""
<style>
    .back-to-top {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(135deg, var(--cmu-red) 0%, var(--cmu-dark-red) 100%);
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 1.2rem;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(196, 30, 58, 0.3);
        transition: all 0.3s ease;
        z-index: 1000;
    }
    
    .back-to-top:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(196, 30, 58, 0.4);
    }
</style>

<button class="back-to-top" onclick="window.scrollTo({top: 0, behavior: 'smooth'})">
    ↑
</button>
""", unsafe_allow_html=True)