"""
Simple ReAct Agent using LangChain
===================================
ReAct (Reasoning + Acting) loops through:
  Thought  → the model reasons about what to do
  Action   → the model calls a tool
  Observation → the tool result is fed back
  ... repeat until the model produces a final Answer.

Dependencies:
  pip install langchain langchain-openai
"""

import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain import hub


# ---------------------------------------------------------------------------
# 1. Define tools
#    Each @tool function becomes an action the agent can call.
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression. Input must be a valid Python expression."""
    try:
        result = eval(expression, {"__builtins__": {}})  # restricted eval
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_word_length(word: str) -> str:
    """Return the number of characters in a word or phrase."""
    return str(len(word))


tools = [calculator, get_word_length]


# ---------------------------------------------------------------------------
# 2. Set up the LLM
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"],
)


# ---------------------------------------------------------------------------
# 3. Pull a standard ReAct prompt from LangChain Hub
#    hwchase17/react is the canonical ReAct prompt template.
#    It expects {tools}, {tool_names}, {input}, and {agent_scratchpad}.
# ---------------------------------------------------------------------------

prompt = hub.pull("hwchase17/react")


# ---------------------------------------------------------------------------
# 4. Build the agent and executor
# ---------------------------------------------------------------------------

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,       # prints each Thought / Action / Observation step
    max_iterations=5,   # safety cap on the reasoning loop
)


# ---------------------------------------------------------------------------
# 5. Run it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    questions = [
        "What is 37 multiplied by 48?",
        "How many characters are in the word 'intelligence'? Then multiply that number by 3.",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("=" * 60)
        result = agent_executor.invoke({"input": question})
        print(f"\nFinal answer: {result['output']}")
