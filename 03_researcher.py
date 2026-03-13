

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
from crewai import Agent, Task, Crew
from llm_config import build_llm, run_with_retry

load_dotenv()


from importlib import import_module
tools = import_module("02_tools")
product_search_tool = tools.product_search_tool



llm = build_llm(
    model=os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant"),
    temperature=0.7,
    max_tokens=500,
    force_react=True,  
)



researcher = Agent(
    role="Product Research Analyst",
    goal="Find the best products within budget by searching the web "
         "for real reviews, specs, and prices.",
    backstory="You are a meticulous e-commerce researcher who always "
              "searches for real product data before recommending anything.",
    tools=[product_search_tool],
    llm=llm,
    max_iter=3,
    verbose=True,
)



history = []


def format_history():
    if not history:
        return "(Start of conversation.)"
    return "\n".join(
        f"Customer: {t['user']}\nAgent: {t['agent']}" for t in history
    )


def ask(user_input):
    def go():
        task = Task(
            description=(
                f"Conversation so far:\n{format_history()}\n\n"
                f"Customer says: {user_input}\n\n"
                f"Search the web for real product data if needed. "
                f"Use conversation history for follow-ups."
            ),
            expected_output="A helpful response with real product recommendations.",
            agent=researcher,
        )
        crew = Crew(agents=[researcher], tasks=[task], verbose=True)
        return crew.kickoff()
    return str(run_with_retry(go, llm=llm))




if __name__ == "__main__":
    print("=" * 60)
    print("  Researcher Agent — with Web Search tool")
    print(f"  Model: {llm.model}  |  Agent: {researcher.role}")
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
        print(f"\nAgent: {response}\n")

        history.append({"user": user_input, "agent": response})
        history[:] = history[-10:]

    print("Goodbye!")
    os._exit(0)