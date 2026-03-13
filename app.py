

import warnings
warnings.filterwarnings("ignore")

import os, sys, logging

os.environ["CREWAI_TRACING_ENABLED"] = "false"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
logging.getLogger("litellm").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

import streamlit as st
from dotenv import load_dotenv
from crewai import Task, Crew
from llm_config import build_llm, run_with_retry
from importlib import import_module

load_dotenv()



@st.cache_resource
def load_agents():
    mod_03 = import_module("03_researcher")
    mod_04 = import_module("04_policy_advisor")

    researcher = mod_03.researcher
    advisor = mod_04.advisor

    researcher_llm = build_llm(
        model=os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant"),
        temperature=0.7, max_tokens=500,
    )
    advisor_llm = build_llm(
        model=os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant"),
        temperature=0.7, max_tokens=500, force_react=True,
    )
    researcher.llm = researcher_llm
    researcher.max_iter = 5
    advisor.llm = advisor_llm
    advisor.max_iter = 5

    router_llm = build_llm(
        model=os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant"),
        temperature=0.0, max_tokens=20,
    )

    return researcher, advisor, researcher_llm, advisor_llm, router_llm


researcher, advisor, researcher_llm, advisor_llm, router_llm = load_agents()



ROUTER_PROMPT = """Classify this customer query into exactly ONE category:

- product  → asking about product recommendations, comparisons, prices, reviews
- policy   → asking about returns, warranty, shipping, loyalty, store rules

Reply with ONLY one word: product or policy

Query: {query}
Category:"""


def route(query):
    try:
        response = router_llm.call(
            messages=[{"role": "user", "content": ROUTER_PROMPT.format(query=query)}]
        )
        category = str(response).strip().lower().split()[0]
        if category not in ("product", "policy"):
            category = "product"
    except Exception:
        category = "product"
    return category


def format_history(messages):
    turns = [m for m in messages if m["role"] in ("user", "assistant")]
    if not turns:
        return "(Start of conversation.)"
    lines = []
    for i in range(0, len(turns) - 1, 2):
        user_msg = turns[i]["content"] if i < len(turns) else ""
        agent_msg = turns[i + 1]["content"] if i + 1 < len(turns) else ""
        if len(agent_msg) > 200:
            agent_msg = agent_msg[:200] + "..."
        lines.append(f"Customer: {user_msg}\nAgent: {agent_msg}")
    return "\n".join(lines[-5:])


def ask(user_input, messages):
    category = route(user_input)
    agent = researcher if category == "product" else advisor
    active_llm = researcher_llm if category == "product" else advisor_llm

    if category == "product":
        instruction = (
            "You MUST call the product_web_search tool to find real data. "
            "After getting results, summarize them in plain friendly language."
        )
    else:
        instruction = (
            "You MUST call the store_policy_lookup tool to find the answer. "
            "After getting results, explain the full policy details to the "
            "customer in plain friendly language. Include all relevant rules, "
            "timeframes, fees, and conditions."
        )

    def go():
        task = Task(
            description=(
                f"Conversation so far:\n{format_history(messages)}\n\n"
                f"Customer says: {user_input}\n\n"
                f"{instruction}"
            ),
            expected_output="A helpful, natural-language response to the customer.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        return crew.kickoff()

    result = str(run_with_retry(go, llm=active_llm))
    return result, category




st.set_page_config(page_title="ShopWise Assistant", layout="centered")

st.title("ShopWise Assistant")
st.caption("Ask about products or store policies — the router picks the right agent.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result, category = ask(prompt, st.session_state.messages)
                agent_name = "Product Research Analyst" if category == "product" else "Store Policy Advisor"
                st.caption(f"Routed to: {agent_name}")
                st.markdown(result)
            except Exception as e:
                result = f"Something went wrong — please try again. ({e})"
                st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})
    st.rerun()