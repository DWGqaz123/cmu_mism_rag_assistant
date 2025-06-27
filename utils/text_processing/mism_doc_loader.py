# utils/text_processing/mism_doc_loader.py
import re
from typing import List, Tuple
from langchain.schema import Document
import os
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredHTMLLoader
from langchain.docstore.document import Document
from typing import List
from .category_map import category_map  

def filter_documents(documents: List[Document], min_length: int = 100) -> Tuple[List[Document], List[Document]]:
    """
    清洗文档：过滤内容过短或包含乱码的 Document 实例。

    Args:
        documents (List[Document]): 原始文档列表。
        min_length (int): 文档内容的最小有效字符数，默认100。

    Returns:
        Tuple[List[Document], List[Document]]: 
            (clean_documents, filtered_documents)
            - clean_documents: 清洗后的有效文档列表；
            - filtered_documents: 被剔除的无效/乱码文档列表。
    """
    clean_documents = []
    filtered_documents = []

    for doc in documents:
        content = doc.page_content.strip()

        # 条件1：内容过短
        too_short = len(content) < min_length

        # 条件2：疑似乱码（如连续20个非字母数字空格）
        has_garbage = re.search(r'[^a-zA-Z0-9\s]{20,}', content[:500]) is not None

        if too_short or has_garbage:
            filtered_documents.append(doc)
        else:
            clean_documents.append(doc)
    
    print(f"\n📊 文档清洗完成：")
    print(f"  - 原始文档数量：{len(documents)}")
    print(f"  - 保留有效文档：{len(clean_documents)}")
    print(f"  - 剔除无效/乱码文档：{len(filtered_documents)}")

    return clean_documents, filtered_documents


# 按类别加载文档
def load_documents_with_category(base_path: str) -> List[Document]:
    all_docs = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, base_path).replace("\\", "/")  # 统一路径分隔符

            # 选择合适的 Loader
            if file.endswith(".txt"):
                loader = TextLoader(full_path, autodetect_encoding=True)
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(full_path)
            elif file.endswith(".html"):
                loader = UnstructuredHTMLLoader(full_path)
            else:
                continue

            try:
                raw_docs = loader.load()
            except Exception as e:
                print(f"⚠️ 跳过加载失败文件：{rel_path}，错误：{e}")
                continue

            # 为每个 document 添加 metadata
            category = category_map.get(rel_path, category_map.get(rel_path.split("/")[0], "unknown"))
            for doc in raw_docs:
                doc.metadata["source"] = rel_path
                doc.metadata["category"] = category

            all_docs.extend(raw_docs)

    # 清洗文档
    clean_docs, filtered_docs = filter_documents(all_docs)
    return clean_docs