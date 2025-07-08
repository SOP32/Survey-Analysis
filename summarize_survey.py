import os
import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load API keys from .env
load_dotenv()

# --- 1. Load Excel responses ---
df = pd.read_excel("survey_responses.xlsx")
open_ended_column = "response"  # change as needed
df = df[[open_ended_column]].dropna().rename(columns={open_ended_column: "text"})

# --- 2. Create LangChain documents from the dataframe ---
loader = DataFrameLoader(df, page_content_column="text")
docs = loader.load()

# --- 3. Split into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# --- 4. Embed and create vector store ---
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    collection_name="survey_collection",
    persist_directory="./chroma_db"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 30})

# --- 5. Use GPT-4o to summarize retrieved chunks ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

query = """
From the retrieved survey responses, identify the most common themes mentioned as strengths and weaknesses.

Please return:
- Top 3 Strengths (as short themes with 1-2 lines each)
- Top 3 Weaknesses (as short themes with 1-2 lines each)

Avoid duplication and ignore off-topic responses.
"""

# Get relevant chunks via RAG
docs = retriever.invoke(query)
context = "\n\n".join([doc.page_content for doc in docs])

# Send to GPT
final_prompt = f"""You are analyzing student survey responses.

Here are the retrieved responses:
{context}

{query}
"""

response = llm.invoke(final_prompt)
print("\n Summary:\n")
print(response.content)
