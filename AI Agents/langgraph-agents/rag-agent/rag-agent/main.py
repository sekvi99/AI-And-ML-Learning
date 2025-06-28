from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from agent.pdf_loader import PDFLoader
from agent.vector_store import VectorStoreManager
from agent.retriever_tool import create_retriever_tool
from agent.prompts import system_prompt
from agent.agent_graph import build_agent_graph
import os

load_dotenv()

# LLM and Embeddings
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# PDF Loading and Splitting
pdf_path = "Stock_Market_Performance_2024.pdf"
pdf_loader = PDFLoader(pdf_path)
pages_split = pdf_loader.load_and_split()

# Vector Store
persist_directory = r"stock_market_persist_data"
collection_name = "stock_market"
vector_manager = VectorStoreManager(persist_directory, collection_name, embeddings)
vectorstore = vector_manager.create_vector_store(pages_split)

# Retriever Tool
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
retriever_tool = create_retriever_tool(retriever)
tools = [retriever_tool]
llm = llm.bind_tools(tools)

# Build Agent Graph
rag_agent = build_agent_graph(llm, tools, system_prompt)

def running_agent():
    print("\n=== RAG AGENT===")
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

if __name__ == "__main__":
    running_agent()