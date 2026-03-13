

import warnings
warnings.filterwarnings("ignore")

import os, sys, logging

os.environ["CREWAI_TRACING_ENABLED"] = "false"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
logging.getLogger("litellm").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

from dotenv import load_dotenv
from crewai import Task, Crew
from llm_config import build_llm, run_with_retry

load_dotenv()



from importlib import import_module
mod_03 = import_module("03_researcher")
mod_04 = import_module("04_policy_advisor")

researcher = mod_03.researcher
advisor = mod_04.advisor

researcher_llm = build_llm(
    model=os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant"),
    temperature=0.7,
    max_tokens=500,
)
advisor_llm = build_llm(
    model=os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant"),
    temperature=0.7,
    max_tokens=500,
    force_react=True,
)
researcher.llm = researcher_llm
researcher.max_iter = 5
advisor.llm = advisor_llm
advisor.max_iter = 5


router_llm = build_llm(
    model=os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant"),
    temperature=0.0,
    max_tokens=20,
)



ROUTER_PROMPT = """Classify this customer query into exactly ONE category:

- product  → asking about product recommendations, comparisons, prices, reviews
- policy   → asking about returns, warranty, shipping, loyalty, store rules

Reply with ONLY one word: product or policy

Query: {query}
Category:"""


def route(query):
    """Ask the LLM to classify the query as product or policy."""
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




history = []


def format_history():
    if not history:
        return "(Start of conversation.)"
    lines = []
    for t in history[-5:]:
        agent_text = t["agent"]
        if len(agent_text) > 200:
            agent_text = agent_text[:200] + "..."
        lines.append(f"Customer: {t['user']}\nAgent: {agent_text}")
    return "\n".join(lines)


def ask(user_input):
    category = route(user_input)
    agent = researcher if category == "product" else advisor
    active_llm = researcher_llm if category == "product" else advisor_llm
    print(f"  [Router → {category} → {agent.role}]")

    def go():
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
        task = Task(
            description=(
                f"Conversation so far:\n{format_history()}\n\n"
                f"Customer says: {user_input}\n\n"
                f"{instruction}"
            ),
            expected_output="A helpful, natural-language response to the customer.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        return crew.kickoff()

    return str(run_with_retry(go, llm=active_llm))




if __name__ == "__main__":
    print("=" * 60)
    print("  Orchestrator — Smart Router")
    print(f"  Model: {researcher_llm.model}")
    print(f"  Agent 1: {researcher.role} (web search)")
    print(f"  Agent 2: {advisor.role} (RAG)")
    print("=" * 60)
    print("The router decides which agent handles your query.")
    print("'clear' = reset, 'quit' = exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "clear":
            history.clear()
            print("[Memory cleared]\n")
            continue

        try:
            response = ask(user_input)
        except Exception as e:
            print(f"\n  [Error: {e}]\n  Wait a moment and try again.\n")
            continue
        print(f"\nAgent: {response}\n")

        history.append({"user": user_input, "agent": response})
        history[:] = history[-10:]

    print("Goodbye!")
    os._exit(0)