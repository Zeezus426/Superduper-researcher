import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
url = os.getenv("NEO4J_URL")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")


loader = TextLoader("/Users/zacharyaldin/Downloads/Pagani Zonda R Specifications and Performance Details.html")
documents = loader.load()
text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_spliter.split_documents(documents)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
db = Neo4jVector.from_documents(docs, url=url, username=username, password=password, embedding=embeddings)
query = "What is the horsepower and weight of the Pagani Zonda R?"
docs_with_score = db.similarity_search_with_score(query, k=2)


for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)