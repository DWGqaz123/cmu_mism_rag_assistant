# utils/text_processing/query_classifier.py
# This module is designed to classify user queries into predefined categories
# using a language model (LLM). It initializes the LLM, defines a prompt template
# for classification, and provides a function to classify queries based on the
# categories defined in the `CATEGORIES` list.
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sys

# 确保sys.path包含项目根目录，以便load_key能被导入
if '..' not in sys.path:
    sys.path.append('..')
from utils.load_key import load_key

# 1. 初始化LLM（与cmu_mism_rag_build_and_test.ipynb中LLM的初始化方式相同）
# 注意：这里是为了模块的独立性而再次初始化，但在实际集成时，
# 你可能会从主程序传递LLM实例进来，避免重复加载。
# 为了简化指导，这里采取了独立的初始化方式。
def initialize_llm():
    openai_api_key_val = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key_val:
        # print("OPENAI_API_KEY not found in environment variables. Attempting to load from keys.json.")
        openai_api_key_val = load_key("OPENAI_API_KEY")
    
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=openai_api_key_val)

llm = initialize_llm()

# 2. 定义分类的Prompt Template
# 我们将列出所有可能的类别，并要求LLM从这些类别中选择
# 你的所有文档类别：
# student_handbook, admission, program_intro, program_pathway, career, curriculum, forum
CATEGORIES = [
    "student_handbook",
    "admission",
    "program_intro",
    "program_pathway",
    "career",
    "curriculum",
    "forum"
]

CLASSIFICATION_PROMPT_TEMPLATE = """
你是一个用于将用户关于CMU MISM项目的问题进行分类的助手。
请根据用户问题的内容，将其归类到以下七个类别之一。
请只返回最适合的类别名称，不要包含任何其他文字、解释或标点符号。
如果问题与任何已知类别都不相关，或者你无法确定类别，请返回 "unknown"。

可用类别: {categories}

用户问题: {query}

分类:
"""

classification_prompt = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT_TEMPLATE)
output_parser = StrOutputParser()

# 3. 构建分类链
classification_chain = (
    classification_prompt
    | llm
    | output_parser
)

# 4. 定义分类函数
def classify_query(query: str) -> str:
    """
    根据预定义的类别对用户查询进行分类。
    """
    try:
        # 使用LLM进行分类
        predicted_category = classification_chain.invoke({
            "query": query,
            "categories": ", ".join(CATEGORIES) # 将类别列表转换为字符串传入Prompt
        }).strip().lower() # 清除空格，转换为小写

        # 验证分类结果是否在允许的类别列表中
        if predicted_category in CATEGORIES:
            return predicted_category
        else:
            print(f"警告：LLM返回了一个未知的类别 '{predicted_category}'。将其归类为 'unknown'。")
            return "unknown"
    except Exception as e:
        print(f"分类查询时发生错误: {e}")
        return "unknown" # 发生错误时返回unknown类别

if __name__ == "__main__":
    # 简单的测试
    print("测试查询分类器...")
    test_queries = [
        "MISM项目的入学要求是什么？", # 预期: admission
        "MISM学生毕业后从事什么类型的工作？", # 预期: career
        "MISM课程设置怎么样？", # 预期: curriculum
        "关于MISM 16个月的项目有什么介绍？", # 预期: program_pathway 或 program_intro
        "最新的学生手册在哪里能找到？", # 预期: student_handbook
        "我想了解一下MISM项目的排名和声誉。", # 预期: program_intro
        "如何申请CMU的某个学位？", # 预期: admission
        "巴黎的首都在哪里？", # 预期: unknown
        "有什么关于学生论坛的信息吗？", # 预期: forum
    ]

    for q in test_queries:
        category = classify_query(q)
        print(f"问题: '{q}' -> 分类: '{category}'")