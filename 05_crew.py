

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
from crewai import Task, Crew, Process
from llm_config import build_llm, run_with_retry

load_dotenv()



from importlib import import_module
mod_03 = import_module("03_researcher")
mod_04 = import_module("04_policy_advisor")

researcher = mod_03.researcher
advisor = mod_04.advisor


llm = build_llm(
    model=os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant"),
    temperature=0.7,
    max_tokens=500,
    force_react=True,
)
researcher.llm = llm
advisor.llm = llm



history = []


def format_history():
    if not history:
        return "(Start of conversation.)"
    return "\n".join(
        f"Customer: {t['user']}\nAgent: {t['agent']}" for t in history
    )


def ask(user_input):
    def go():
        research_task = Task(
            description=(
                f"Conversation so far:\n{format_history()}\n\n"
                f"Customer says: {user_input}\n\n"
                f"Search the web for real product data — names, prices, ratings."
            ),
            expected_output="A short list of products with prices and ratings.",
            agent=researcher,
        )

        policy_task = Task(
            description=(
                f"The researcher found products for the customer.\n"
                f"Now look up our store policies for those products —\n"
                f"return policy, warranty, shipping costs, loyalty benefits.\n"
                f"Combine the research + policies into a final recommendation."
            ),
            expected_output="Final recommendation with product picks and policy details.",
            agent=advisor,
        )

        crew = Crew(
            agents=[researcher, advisor],
            tasks=[research_task, policy_task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()
    return str(run_with_retry(go, llm=llm))




if __name__ == "__main__":
    print("=" * 60)
    print("  Multi-Agent Crew — Researcher + Policy Advisor")
    print(f"  Model: {llm.model}")
    print(f"  Agent 1: {researcher.role} (web search)")
    print(f"  Agent 2: {advisor.role} (RAG)")
    print("=" * 60)
    print("Ask about any product. 'clear' = reset, 'quit' = exit\n")

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
        print(f"\nCrew: {response}\n")

        history.append({"user": user_input, "agent": response})
        history[:] = history[-10:]

    print("Goodbye!")
    os._exit(0)