# utils/rag_pipeline_logic.py

import os
import re
from datetime import datetime

# LangChain Document 类，因为 get_rag_answer 返回 Document 对象
from langchain.docstore.document import Document

# 导入自定义工具函数 (注意：这里的导入是相对路径，因为 rag_pipeline_logic.py 在 utils 目录下)
# query_classifier 是 get_rag_answer 内部需要的
from  utils.text_processing.query_classifier import classify_query

def get_rag_answer(question: str, vectordb, llm, flexible_qa_chain, bm25_index, bm25_doc_list, reranker_model, simple_tokenizer_func):
    """
    执行RAG流程并返回答案、源文档及中间处理信息。
    此函数不包含任何Streamlit特定的UI代码，可在任何Python环境中运行。
    """
    response_metadata = {}
    
    # --- 查询分类 ---
    predicted_category = classify_query(question)
    response_metadata['query_category'] = predicted_category
    
    # --- 构建过滤条件 ---
    filter_condition = {}
    if predicted_category != "unknown":
        filter_condition = {"category": predicted_category}
    
    # --- 混合检索 ---
    vector_results = vectordb.similarity_search(
        query=question,
        k=10,
        filter=filter_condition
    )

    tokenized_query = simple_tokenizer_func(question) # 使用传入的分词函数
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

    # --- 上下文重排序 ---
    if retrieved_docs:
        pairs = [(question, doc.page_content) for doc in retrieved_docs]
        rerank_scores = reranker_model.predict(pairs)
        ranked_docs_with_scores = sorted(zip(retrieved_docs, rerank_scores), key=lambda x: x[1], reverse=True)
        retrieved_docs = [doc for doc, score in ranked_docs_with_scores]
    
    response_metadata['retrieved_doc_count_after_rerank'] = len(retrieved_docs)

    # --- 准备上下文给LLM ---
    context_strings = [doc.page_content for doc in retrieved_docs]
    context_str = "\n\n".join(context_strings)

    # --- 调用灵活的LLM链 ---
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