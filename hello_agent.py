

import os, sys, warnings, logging

os.environ["CREWAI_TRACING_ENABLED"] = "false"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
logging.getLogger("litellm").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from llm_config import build_llm, run_with_retry

load_dotenv()



llm = build_llm(
    model=os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant"),
    temperature=0.7,
    max_tokens=500,
)



agent = Agent(
    role="E-commerce Product Expert",
    goal="Help customers with product recommendations and buying advice.",
    backstory="You are a product analyst with 10 years of experience "
              "in consumer electronics, known for honest practical advice.",
    llm=llm,
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
                f"Respond helpfully. Use conversation history for follow-ups."
            ),
            expected_output="A helpful, concise response.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        return crew.kickoff()
    return str(run_with_retry(go, llm=llm))




if __name__ == "__main__":
    print("=" * 60)
    print("  Hello Agent — Single agent with memory (no tools)")
    print(f"  Model: {llm.model}  |  Agent: {agent.role}")
    print("=" * 60)
    print("Type a message to chat. 'clear' = reset, 'quit' = exit\n")

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
    