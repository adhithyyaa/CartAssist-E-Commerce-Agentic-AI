

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
from crewai import Agent
from llm_config import build_llm

load_dotenv()


llm = build_llm(
    model=os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant"),
    temperature=0.7,
    max_tokens=500
)


advisor = Agent(
    role="Store Policy Advisor",
    goal="Explain store policies like warranty, return policy, shipping and loyalty benefits.",
    backstory="You work for an e-commerce store and help customers understand policies clearly.",
    llm=llm,
    verbose=True
)