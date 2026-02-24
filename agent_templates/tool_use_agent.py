"""
Tool Use Pattern Agent using LangChain + LangGraph
====================================================
Unlike ReAct (text-based Thought/Action/Observation loop), this agent uses
the model's NATIVE structured tool calling (function calling).

The LLM returns structured tool_call objects (JSON), not free text.
LangGraph wires the orchestration as an explicit state graph:

    [agent node] ──── has tool calls? ────► [tools node]
          ▲                                      │
          └──────────────────────────────────────┘
                       no tool calls
                            │
                           END

Dependencies:
    pip install langchain langchain-openai langgraph
"""

import os
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# 1. Define tools
#    These are given to the LLM as JSON schemas. The model returns structured
#    tool_call objects — no prompt engineering needed.
# ---------------------------------------------------------------------------

@tool
def search_web(query: str) -> str:
    """Search the web for current information on a topic."""
    # Stub — swap in a real search API (Tavily, SerpAPI, etc.)
    results = {
        "weather in London": "Overcast, 12°C, light rain expected this afternoon.",
        "latest Python version": "Python 3.13 was released in October 2024.",
    }
    return results.get(query.lower(), f"No results found for: {query}")


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # Stub — swap in a real weather API (OpenWeatherMap, etc.)
    forecasts = {
        "london": "12°C, overcast with light rain.",
        "new york": "5°C, clear skies.",
        "tokyo": "18°C, partly cloudy.",
    }
    return forecasts.get(city.lower(), f"Weather data unavailable for {city}.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


tools = [search_web, get_weather, calculator]


# ---------------------------------------------------------------------------
# 2. Set up the LLM with bound tools
#    bind_tools() attaches JSON schemas so the model knows what it can call.
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"],
)

llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# 3. Define the graph state
#    MessagesState is a list of messages that grows as the agent runs.
#    add_messages is a reducer that appends new messages to the list.
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# 4. Define graph nodes
#
#    agent_node: calls the LLM — may return tool_calls or a plain response
#    tool_node:  executes all tool_calls in the latest message, returns results
# ---------------------------------------------------------------------------

def agent_node(state: AgentState) -> AgentState:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)


# ---------------------------------------------------------------------------
# 5. Build the graph
#
#    tools_condition: built-in conditional edge that routes to "tools" if the
#    last AIMessage contains tool_calls, otherwise routes to END.
# ---------------------------------------------------------------------------

graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("agent")

graph_builder.add_conditional_edges(
    "agent",
    tools_condition,          # routes to "tools" or END
)
graph_builder.add_edge("tools", "agent")  # always return to agent after tool call

graph = graph_builder.compile()


# ---------------------------------------------------------------------------
# 6. Run it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    questions = [
        "What is the weather like in Tokyo?",
        "If it's 18°C in Tokyo and 5°C in New York, what is the average temperature?",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("=" * 60)

        final_state = graph.invoke({"messages": [HumanMessage(content=question)]})

        # Print each step so the orchestration is visible
        for msg in final_state["messages"]:
            msg.pretty_print()
