

import warnings
warnings.filterwarnings("ignore")

import os, sys, json, http.client, logging

os.environ["CREWAI_TRACING_ENABLED"] = "false"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
logging.getLogger("litellm").setLevel(logging.CRITICAL)

from dotenv import load_dotenv
from crewai.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()




@tool("Product Web Search")
def product_search_tool(query: str) -> str:
    """Search the web for product reviews and prices.
    Args: query — e.g. 'best earbuds under $50'."""

    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        return "No SERPER_API_KEY found in .env"

    
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query})
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    conn.request("POST", "/search", payload, headers)
    data = json.loads(conn.getresponse().read().decode("utf-8"))
    conn.close()

    
    results = data.get("organic", [])[:5]
    if not results:
        return "No results found."

    output = []
    for r in results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        output.append(f"- {title}\n  {snippet}\n  {link}")

    return "\n\n".join(output)



policies_path = os.path.join(os.path.dirname(__file__), "store_policies.txt")
documents = TextLoader(policies_path, encoding="utf-8").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50).split_documents(documents)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)


@tool("Store Policy Lookup")
def store_policy_rag_tool(question: str) -> str:
    """Look up store policies — returns, warranty, shipping, loyalty.
    Args: question — e.g. 'what is the return policy?'."""

    results = vector_store.similarity_search(question, k=3)
    if not results:
        return "No relevant policy found."

    return "\n\n".join(f"Policy: {doc.page_content}" for doc in results)




if __name__ == "__main__":
    print("=" * 60)
    print("  Tool Tests — Web Search + RAG")
    print("=" * 60)

    
    print("\n--- Tool 1: Product Web Search ---")
    query = "best wireless earbuds under $50"
    print(f"Query: {query}\n")
    print(product_search_tool.run(query))

    
    print("\n--- Tool 2: Store Policy Lookup (RAG) ---")
    question = "What is the return policy?"
    print(f"Question: {question}\n")
    print(store_policy_rag_tool.run(question))

    print("\n" + "=" * 60)
    print("Both tools working! Ready to attach to an agent.")
    print("=" * 60)
    sys.stdout.flush()
    os._exit(0) 