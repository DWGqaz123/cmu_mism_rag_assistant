import re
from typing import List, Tuple
from langchain.schema import Document

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