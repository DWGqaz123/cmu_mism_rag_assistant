# streamlit_app.py

# ==============================================================================
# 1. 必要的库导入
# ==============================================================================
import streamlit as st
import os
import re # simple_tokenizer_func 还在用
import torch
import json
import time
from datetime import datetime

# 导入 LangChain Document 类 (因为 get_rag_answer 返回 Document 对象)
from langchain.docstore.document import Document

# 导入新的共享模块
import sys
# 确保 sys.path 包含项目根目录，以便导入 utils
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.rag_core_setup import initialize_rag_components_cli # 用于加载所有组件
from utils.rag_pipeline_logic import get_rag_answer # 用于核心RAG逻辑

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
        margin-top: 1.5rem;
        width: 100%;
        justify-content: center
    }
    
    .status-success {
        background: rgba(0, 212, 170, 0.2);
        color: var(--success-green);
        border: 1px solid var(--success-green);
        justify-content: center
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
        width: 500px;
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
@st.cache_resource(show_spinner=False) # 禁用默认的加载旋转器，我们自己控制
def load_rag_components_for_streamlit(): # 更改函数名以避免混淆
    """加载并初始化所有RAG组件，适用于Streamlit环境。"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(value, text, level="info"):
        progress_bar.progress(value)
        status_text.markdown(f"<div class='status-indicator status-{level}'>🔄 {text}</div>", unsafe_allow_html=True)
    
    update_progress(0.1, "初始化系统组件...")
    
    try:
        # 接收 initialize_rag_components_cli 返回的所有值，包括 documents 和 chunks
        llm, embeddings, QA_CHAIN_PROMPT, flexible_qa_chain, vectordb, bm25_index, bm25_doc_list, reranker_model, simple_tokenizer_func, documents, chunks = initialize_rag_components_cli(verbose=False) # 禁用 CLI 内部打印
        
        update_progress(1.0, "完成初始化!")
        time.sleep(1) # 稍作停留，让用户看到完成状态
        
        # 现在可以安全地使用 len(documents) 和 len(chunks)
        return llm, embeddings, QA_CHAIN_PROMPT, flexible_qa_chain, vectordb, bm25_index, bm25_doc_list, reranker_model, simple_tokenizer_func, len(documents), len(chunks)
    except Exception as e:
        status_text.markdown(f"<div class='status-indicator status-error'>❌ 初始化失败: {e}</div>", unsafe_allow_html=True)
        st.error(f"❌ 系统初始化失败: {e}。请检查您的API密钥和知识库是否正确构建。")
        progress_bar.empty()
        st.stop()
    finally:
        progress_bar.empty()
        status_text.empty() # 确保进度条和状态文本最终被清除

# ==============================================================================
# 4. 主界面
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
    st.title("💡 关于助手")
    st.info(
        "本助手旨在回答关于 **CMU 信息系统管理 (MISM)** 项目的各种问题。\n\n"
        "它集成了 **语义分块、查询分类、混合检索和上下文重排序** 等先进 RAG 技术。"
    )
    st.markdown("---")
    st.markdown("### 📊 系统状态")
    
    # 加载RAG组件
    try:
        llm, embeddings, QA_CHAIN_PROMPT, flexible_qa_chain, vectordb, bm25_index, bm25_doc_list, reranker_model, simple_tokenizer_func, total_docs, total_chunks = load_rag_components_for_streamlit()
        
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
        st.stop() # 如果初始化失败，停止应用

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
    st.session_state['message_count'] = 0 # 用于确保 text_area 的 key 唯一

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
