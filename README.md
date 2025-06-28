# 🎓 CMU MISM RAG 智能问答助手

## 🚀 项目简介

本项目旨在构建一个针对**卡内基梅隆大学信息系统管理（CMU MISM）项目**的智能问答助手。通过集成先进的\*\*检索增强生成（Retrieval-Augmented Generation, RAG）\*\*技术，本助手能够从 MISM 项目的官方文档、手册和相关资料中检索信息，并利用大型语言模型（LLM）生成精确、全面的回答，帮助用户快速了解项目的各项细节。

## ✨ 功能亮点

  * **多文档格式支持**：能够处理包括 PDF、HTML 和 TXT 在内的多种格式文档，构建全面的知识库.
  * **语义分块**：采用 `SemanticChunker` 策略，确保每个文本块包含完整的语义单元，提高上下文质量.
  * **智能查询分类**：利用 LLM 对用户查询进行动态分类（例如“申请要求”、“职业路径”），实现精准的预检索过滤.
  * **混合检索 (Hybrid Search)**：结合了**向量相似度检索**（捕捉语义相关性）和 **BM25 关键词检索**（确保精确匹配），显著提升了信息召回率和准确性.
  * **上下文重排序 (Contextual Re-ranking)**：对初步检索到的文档进行二次排序，确保最相关和最关键的信息优先传递给 LLM.
  * **多运行模式**：提供**命令行界面 (CLI)** 和 **Streamlit 交互式 Web UI** 两种使用模式.
  * **知识边界控制**：LLM 能够识别并拒绝回答知识库范围之外的问题，避免“幻觉”现象，保持回答的专业性.
  * **模块化代码结构**：核心 RAG 逻辑已抽离为独立模块，提高了代码复用性和可维护性.

## ⚙️ 安装指南

请按照以下步骤设置并运行本项目：

### 1\. 克隆仓库

首先，将本项目克隆到您的本地机器：

```bash
git clone https://github.com/DWGqaz123/cmu_mism_rag_assistant.git
cd cmu_mism_rag_assistant
```

### 2\. 创建并激活 Python 虚拟环境

强烈建议使用虚拟环境来管理项目依赖：

```bash
# 使用 conda
conda create -n rag_env python=3.10
conda activate rag_env

# 或者使用 venv
python3 -m venv rag_env
source rag_env/bin/activate
```

### 3\. 安装项目依赖

安装 `requirements.txt` 文件中列出的所有依赖：

```bash
pip install -r requirements.txt
```

### 🔑 API Key 配置

本项目需要 OpenAI API 密钥 (`OPENAI_API_KEY`) 来运行大型语言模型(也可使用Gemini API 和本地Ollama模型)。请通过以下任一方式提供您的密钥：

#### 3.1 推荐方式：通过环境变量设置

  * **临时设置 (当前会话有效)**：
    在您的终端中，运行 Streamlit 应用或 `main.py` 之前，执行以下命令：
    ```bash
    export OPENAI_API_KEY="sk-您的OpenAI_API_密钥" 
    ```
  * **永久设置 (每次启动终端都有效)**：
    将上述 `export` 语句添加到您的 Shell 配置文件中（如 `~/.bashrc`、`~/.zshrc` 或 `~/.bash_profile`），然后运行 `source ~/.bashrc` (或相应文件) 使其生效。

#### 3.2 Streamlit Cloud 部署方式：使用 Streamlit Secrets

如果您计划将本项目部署到 Streamlit Community Cloud，您可以在其仪表板中以安全的方式设置您的密钥。

  * **操作步骤**：
    1.  登录 Streamlit Cloud 并部署您的应用。
    2.  在应用的设置界面中，找到“Secrets”或“Manage secrets”选项。
    3.  添加一个新密钥，键为 `OPENAI_API_KEY`，值为您的实际 OpenAI API 密钥。
  * **参考文档**：请参考 [Streamlit Secrets 官方文档](https://www.google.com/search?q=https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/set-up-a-streamlit-app%23set-up-secrets) 获取详细步骤。

#### 3.3 备选方式：通过 `keys.json` 文件 

作为本地开发的备选方案，助手运行时会调用load_key.py提示您可以在项目根目录创建一个名为 `keys.json` 的文件，并提示输入您的 API 密钥：

  * **`keys.json` 文件内容示例**：
    ```json
    {
        "OPENAI_API_KEY": "sk-您的OpenAI_API_密钥" # 请务必替换为您的实际密钥
    }
    ```

## 📚 知识库构建
项目data目录下已包含基本的知识信息，包括student handbooks, website pages, forums_info等。（定期补充）

## ▶️ 使用说明

您的 RAG 助手现在有两种运行模式。请确保在运行前已完成**API Key 配置**。

### 1\. 运行 Streamlit Web UI (推荐)

提供一个交互式图形界面，方便用户提问并查看答案和来源。

1.  在终端中，导航到项目根目录（与 `streamlit_app.py` 文件同级）。
2.  运行：
    ```bash
    streamlit run streamlit_app.py
    ```
3.  应用程序将在您的默认浏览器中打开。

### 2\. 运行命令行界面 (CLI)

提供一个简洁的终端问答体验。

1.  在终端中，导航到项目根目录（与 `main.py` 文件同级）。
2.  运行：
    ```bash
    python main.py
    ```
3.  系统初始化完成后，您可以在终端中输入问题。输入 `exit` 或 `quit` 结束会话。

## 📂 项目结构

```
.
├── .git/                      # Git 版本控制
├── .gitignore                 # Git 忽略文件配置
├── .streamlit/                # Streamlit 应用配置，包含 secrets.toml
├── .vscode/                   # VS Code 编辑器配置
├── data/                      # 原始知识库文档
│   └── cmu_mism_docs/         # CMU MISM 相关文档 (PDF, HTML, TXT)
├── keys.json                  # API 密钥文件 
├── main.py                    # 命令行界面 (CLI) 入口
├── models/                    # 可能用于存放本地模型文件 
├── notebooks/                 # Jupyter Notebooks，用于开发、实验和知识库构建
│   └── cmu_mism_rag_build_and_test.ipynb
├── requirements.txt           # 项目依赖列表
├── streamlit_app.py           # Streamlit UI 应用主文件
├── test_output/               # 测试输出结果
├── utils/                     # 辅助工具函数和模块
│   ├── load_key.py            # API 密钥加载工具
│   ├── model_training/
│   │   └── train_eval_utils.py # 训练/评估工具 (如设置随机种子)
│   ├── rag_core_setup.py      # RAG 核心组件初始化逻辑 (供 main.py 和 streamlit_app.py 调用)
│   └── rag_pipeline_logic.py  # RAG 核心问答逻辑 (供 main.py 和 streamlit_app.py 调用)
│   └── text_processing/
│       ├── category_map.py    # 文档类别映射
│       ├── mism_doc_loader.py # 文档加载器与清洗
│       └── query_classifier.py# 查询分类器
└── vectorstore/               # 持久化存储的向量数据库 (ChromaDB) 和相关索引
    └── cmu_mism_chroma/
```

## 📈 未来优化方向

  * **性能优化**：
      * 研究更高效的文档加载和预处理，尤其是针对大规模数据集。
      * 探索更快的嵌入模型或量化技术以降低推理延迟。
      * 在 Streamlit Cloud 部署时，优化冷启动时间。
  * **检索增强**：
      * 实现 **HyDE (Hypothetical Document Embedding)** 策略，通过 LLM 生成假设文档来改进检索。
      * 集成 **上下文压缩 (Contextual Compression)**，进一步精炼传递给 LLM 的上下文。
  * **LLM 交互**：
      * 引入 **ReAct (Reasoning and Acting)** 或 **Chain-of-Thought (CoT) RAG**，提升 LLM 的推理能力和答案的逻辑性。
      * 实现用户反馈机制，允许用户对答案质量进行评分，以收集数据进行未来的模型改进。
  * **扩展知识库**：
      * 支持更多数据源类型（如视频转录、音频、图片中的文本）。
      * 集成实时数据更新和增量索引功能。

## 👋 联系方式

如果您有任何问题、建议或希望合作，欢迎通过以下方式联系我：

  * **Email**: `dwgqaz123@outlook.com`
  * **GitHub**: [DWGqaz123/cmu\_mism\_rag\_assistant](https://github.com/DWGqaz123/cmu_mism_rag_assistant)