# 🛒 CartAssist

### Agentic AI E-Commerce Assistant

!\[Python](https://img.shields.io/badge/Python-3.10+-blue)
!\[CrewAI](https://img.shields.io/badge/Agent%20Framework-CrewAI-purple)
!\[LLM](https://img.shields.io/badge/LLM-Groq-green)
!\[VectorDB](https://img.shields.io/badge/Vector%20Database-FAISS-orange)
!\[Frontend](https://img.shields.io/badge/UI-Streamlit-red)
!\[License](https://img.shields.io/badge/License-MIT-lightgrey)

CartAssist is an **Agentic AI system designed to automate product
discovery and customer support for e-commerce platforms**.

The system combines **multi-agent collaboration, retrieval-augmented
knowledge systems, and large language models (LLMs)** to interpret user
queries, retrieve relevant product or policy information, and generate
intelligent responses.

Built using **CrewAI, LangChain retrieval pipelines, Groq LLMs, FAISS
vector search, and Streamlit**, CartAssist demonstrates how modern AI
architectures can power **next-generation AI shopping assistants**.

\---

 Key Features

###  Multi-Agent AI Collaboration

CartAssist uses **specialized AI agents that work together to solve user
queries**.

**Product Research Agent** - Finds product recommendations - Analyzes
product specifications - Retrieves relevant items for user needs

**Store Policy Advisor Agent** - Answers questions related to: -
returns - shipping - warranty - store policies - Uses a **vectorized
knowledge base**



### Retrieval-Augmented Intelligence

The system integrates **vector search using FAISS** with **HuggingFace
embeddings** to retrieve store policy knowledge before generating
responses.

Benefits: - higher factual accuracy - contextual responses - scalable
knowledge retrieval

\---

###  Agent Orchestration

Agents are coordinated through **CrewAI orchestration pipelines**,
enabling structured collaboration between agents.

Example workflow:

&#x20;   User Query
       │
       ▼
    Query Intent Analysis
       │
       ├── Product Query → Product Research Agent
       │
       └── Policy Query → Policy Advisor Agent
                │
                ▼
         Vector Knowledge Retrieval
                │
                ▼
            LLM Response


\---

#  System Architecture

&#x20;                   ┌─────────────────────┐
                    │     User Query      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Streamlit UI      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   CrewAI Orchestrator│
                    └──────────┬──────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
    ┌───────────────┐                   ┌────────────────┐
    │ Product Agent │                   │ Policy Agent   │
    └───────┬───────┘                   └───────┬────────┘
            │                                   │
            ▼                                   ▼
     Product Search Tool              FAISS Knowledge Base
            │                                   │
            └──────────────┬────────────────────┘
                           ▼
                     Groq LLM Engine
                           ▼
                     Final Response


\---

# 🛠 Tech Stack

Layer                  Technology

\---

Programming Language   Python
Agent Framework        CrewAI
LLM Provider           Groq
Models                 LLaMA 3.1 / Gemma
Retrieval Framework    LangChain
Vector Database        FAISS
Embeddings             HuggingFace
Frontend               Streamlit
Configuration          dotenv

\---

#  Project Structure

&#x20;   CartAssist
    │
    ├── app.py                  # Streamlit user interface
    ├── 06\_orchestrator.py      # Main agent workflow controller
    ├── 05\_crew.py              # CrewAI agent coordination
    ├── 03\_researcher.py        # Product research agent
    ├── 04\_policy\_advisor.py    # Policy advisor agent
    ├── 02\_tools.py             # Product search tools
    ├── llm\_config.py           # LLM configuration
    ├── store\_policies.txt      # Policy knowledge base
    ├── requirements.txt
    └── README.md


\---

#  Installation \& Setup

## 1️⃣ Clone the repository

``` bash
git clone https://github.com/yourusername/cartassist.git
cd cartassist
```

## 2️⃣ Install dependencies

``` bash
pip install -r requirements.txt
```

## 3️⃣ Configure API Key

Create a `.env` file:

&#x20;   GROQ\_API\_KEY=your\_groq\_api\_key


Optional:

&#x20;   GROQ\_MODEL=groq/llama-3.1-8b-instant


## 4️⃣ Run the application

&#x20;   streamlit run app.py


The assistant will launch in your browser.

\---

#  Demo

(Add screenshots or a demo GIF here once the application is running.)

Example interaction:

&#x20;   User: Recommend a good laptop under $1000
    AI: Here are some laptops that match your budget and requirements...


\---

#  Example Queries

&#x20;   Recommend a smartphone under $500
    What is the return policy?
    Do you offer free shipping?
    Suggest laptops for programming


\---

#  Why This Project Matters

Modern e-commerce platforms require **intelligent, scalable customer
interaction systems**.

CartAssist demonstrates how **agentic AI architectures combined with
retrieval pipelines** can:

* automate customer support
* improve product discovery
* enhance online shopping experiences
* reduce operational costs

\---

#  Future Improvements

* Integration with live e-commerce APIs
* conversational memory
* personalized product recommendations
* multilingual support
* voice-enabled shopping assistants

\---

#  Author

**Guru Adhithya**  
Computer Science Undergraduate  
Focused on AI systems, data analytics, and intelligent product
development

GitHub: https://github.com/adhithyyaa

\---

# ⭐ Support

If you find this project interesting, consider **starring the
repository**.

