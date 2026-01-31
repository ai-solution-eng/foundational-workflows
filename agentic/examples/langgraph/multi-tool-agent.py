"""
LangGraph MCP Agent - With Memory + MCP + Retry Correction

Features:
- MCP tool loading preserved
- Tool schema descriptions preserved
- Manual TOOL_CALL parsing
- Short-term memory (last 3 messages)
- Tool result persistence
- Anti-repeat retry logic
"""

import os
import httpx
import logging
import asyncio
import re
from typing import List, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
from phoenix.otel import register

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from fastmcp import Client
from fastmcp.client import StreamableHttpTransport

# ================= INIT =================

load_dotenv(find_dotenv())
project_name = "langgraph"
register(project_name=project_name, auto_instrument=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

custom_headers = {"X-Model-Authorization": f"{os.getenv('NEMO_MAIN_MODEL_TOKEN')}"}
guardrail_config_id = os.getenv("NEMO_NAMESPACE") + '/' + os.getenv("GUARDRAIL_CONFIG_NAME")

os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ezua-domain-ca-certs/ezua-domain-ca-cert.crt"
os.environ["SSL_CERT_FILE"] = "/etc/ezua-domain-ca-certs/ezua-domain-ca-cert.crt"


def refresh_token():
    """
    Reads and returns the authentication token from a local secret file.

    Returns:
        str: The current authentication token for MCP connections.
    """
    with open("/etc/secrets/ezua/.auth_token", "r") as file:
        return file.read().strip()


BEARER = refresh_token()


def auth_client_factory(headers=None, timeout=None, auth=None, **kwargs):
    """
    Factory function to create an httpx.AsyncClient for MCP connections
    with the current bearer token.

    Args:
        headers (dict, optional): Additional headers. Defaults to None.
        timeout (float, optional): Connection timeout. Defaults to None.
        auth: Authentication object (unused).
        **kwargs: Extra arguments to pass to AsyncClient.

    Returns:
        httpx.AsyncClient: Configured HTTP client.
    """
    return httpx.AsyncClient(
        headers={"Authorization": f"Bearer {BEARER}"},
        timeout=120,
        **kwargs
    )


llm = ChatOpenAI(
    model=os.getenv("NEMO_MAIN_MODEL_ID"),
    base_url=os.getenv("GUARDRAILS_BASE_URL"),
    api_key="dummy",
    default_headers=custom_headers,
    extra_body={"guardrails": {"config_id": guardrail_config_id}},
    streaming=True
)

max_iterations = 15

# ================= MEMORY HELPER =================


def build_messages_with_history(system_message, state, max_history: int = 3):
    """
    Build a list of messages for the LLM including a short session history.

    Args:
        system_message (SystemMessage): System prompt for the agent.
        state (AgentState): Current agent state containing message history.
        max_history (int, optional): Number of previous messages to include. Defaults to 3.

    Returns:
        List[BaseMessage]: Messages to send to LLM including history.
    """
    history = state.messages[-max_history:] if state.messages else []
    return [system_message] + history


# ================= STATE =================

class AgentState(BaseModel):
    """
    Represents the persistent state of an agent workflow.

    Attributes:
        messages (List[BaseMessage]): All messages exchanged by the agent so far.
        topic (str): The main question or subject being analyzed.
        research_result (str): Output of the researcher node.
        sql_result (str): Output of the SQL node.
        summary (str): Final summarized output.
        next_step (str): Name of the next node to execute in the workflow.
    """
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    topic: str = ""
    research_result: str = ""
    sql_result: str = ""
    summary: str = ""
    next_step: str = ""

    class Config:
        arbitrary_types_allowed = True


# ================= MCP TOOL LAYER =================

class SimpleTool:
    """
    Wrapper class for an MCP tool.

    Attributes:
        name (str): Name of the tool.
        description (str): Human-readable description.
        client (Client): MCP client used to execute the tool.
        schema (dict, optional): Optional input schema of the tool.
    """
    def __init__(self, name: str, description: str, client: Client, schema=None):
        self.name = name
        self.description = description
        self.client = client
        self.schema = schema

    async def execute(self, **kwargs):
        try:
            result = await self.client.call_tool(self.name, kwargs)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"


mcp_server_params = [
    {
        "url": os.getenv("MCP_EZPRESTO_SERVER"),
        "transport": "streamable-http",
        "httpx_client_factory": auth_client_factory,
    },
    {
        "url": os.getenv("MCP_FILESYSTEM_SERVER"),
        "transport": "streamable-http",
    },
    {
        "url": os.getenv("MCP_S3_SERVER"),
        "transport": "streamable-http",
    },
]


async def load_mcp_tools():
    """
        Executes the tool using the MCP client.

        Args:
            **kwargs: Tool-specific input arguments.

        Returns:
            str: Tool execution result or error string.
        """
    tools = []
    clients = []

    for server_config in mcp_server_params:
        try:
            client = Client(
                transport=StreamableHttpTransport(
                    url=server_config["url"],
                    httpx_client_factory=server_config.get("httpx_client_factory")
                )
            )
            await client.__aenter__()
            clients.append(client)

            available_tools = await client.list_tools()
            logger.info(f"Found {len(available_tools)} tools from {server_config['url']}")


            for tool_info in available_tools:
                tools.append(
                    SimpleTool(
                        tool_info.name,
                        tool_info.description or "",
                        client,
                        schema=tool_info.inputSchema
                    )
                )
                logger.info(f"Loaded tool: {tool_info.name}")

        except Exception as e:
            logger.error(f"MCP connection error: {e}")

    logger.info(f"Successfully loaded {len(tools)} tools")
    return tools, clients


async def cleanup_mcp_clients(clients):
    """
    Connects to all configured MCP servers and loads available tools.

    Returns:
        Tuple[List[SimpleTool], List[Client]]: Loaded tools and active clients.
    """
    """Disconnect from all MCP servers"""
    logger.info("Disconnecting from MCP servers...")
    for client in clients:
        try:
            await client.__aexit__(None, None, None)
        except Exception as e:
            logger.error(f"Error disconnecting client: {e}")


def create_tool_descriptions(tools):
    """
    Gracefully disconnects all active MCP clients.

    Args:
        clients (List[Client]): MCP clients to disconnect.
    """
    descriptions = ["Available tools:\n"]

    for tool in tools:
        if tool.schema:
            fields = []
            for name, field in tool.schema.get("properties", {}).items():
                required = name in tool.schema.get("required", [])
                field_type = field.get("type", "unknown")
                fields.append(
                    f"  {name}: {field_type}" + (" (required)" if required else "")
                )

            args_block = "\n".join(fields)
            descriptions.append(
                f"{tool.name}(\n{args_block}\n)\nDescription: {tool.description}\n"
            )
        else:
            descriptions.append(f"{tool.name}: {tool.description}\n")

    return "\n".join(descriptions)


# ================= TOOL PARSER =================

def parse_tool_calls(response_text):
    """
    Creates a textual description of all available MCP tools, including their input schema.

    Args:
        tools (List[SimpleTool]): Tools to describe.

    Returns:
        str: Formatted description of all tools for LLM prompts.
    """
    pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
    matches = re.finditer(pattern, response_text, re.DOTALL)
    tool_calls = []

    for match in matches:
        name = match.group(1)
        args_str = match.group(2).strip()
        args = {}
        if args_str:
            try:
                args = eval(f"dict({args_str})")
            except:
                args = {}
        tool_calls.append({"name": name, "args": args})

    return tool_calls


async def execute_tools(response_text, tools):
    """
    Parses tool calls from LLM response text.

    Args:
        response_text (str): LLM output potentially containing TOOL_CALLs.

    Returns:
        List[dict]: List of tool call dictionaries with 'name' and 'args'.
    """
    tool_calls = parse_tool_calls(response_text)
    results = []

    for call in tool_calls:
        tool = next((t for t in tools if t.name == call["name"]), None)
        if tool:
            logger.info(f"Executing tool: {call['name']} with args: {call['args']}")
            result = await tool.execute(**call["args"])
            results.append(f"{tool.name} result:\n{result}")
            logger.info(f"Result: {result}")
        else:
            results.append(f"{call['name']} not found")

    return "\n\n".join(results)


# ================= NODES =================


async def researcher_node(state: AgentState, tools: List[SimpleTool] = None, max_iterations: int = 3) -> AgentState:
    """
    Executes any tool calls found in LLM response text.

    Args:
        response_text (str): LLM output containing TOOL_CALLs.
        tools (List[SimpleTool]): List of available tools.

    Returns:
        str: Combined results of all executed tools.
    """
    """Research agent that analyzes files and data with multiple iterations."""
    logger.info("Researcher agent starting...")

    tools = [tool for tool in tools if tool.name not in ["execute_query", "get_table_schema", "list_catalogs", "list_schemas", "list_tables"]]
    tool_desc = create_tool_descriptions(tools) if tools else "No tools available."

    system_message = SystemMessage(
        content=(
            "You are a competent researcher. Use available tools to analyze files "
            "and data, both remote s3 object stores and local filesystems.\n"
            "When browsing filesystems, look for 'data' folders and then search\n"
            " for specific files with keywords that match the subject matter.\n"
            "Exclude any hidden directories that start with '.' by using escaped regex period.\n"
            "Only make one tool call at a time, avoid multiple calls.\n"
            "Take into account the previous tool result before making a new tool call.\n"
            "Pay close attention to the bucket names when calling s3 resources.\n"
            "To use a tool, write: TOOL_CALL: tool_name(arg1=\"value1\", arg2=\"value2\")\n\n"
            f"{tool_desc}\n\n"
        )
    )

    user_message = HumanMessage(
        content=f"Answer '{state.topic}' based on available data only. Look for any relevant files."
    )

    # Keep a session history with the last few messages
    session_history: List[BaseMessage] = [system_message, user_message]

    for iteration in range(max_iterations):
        logger.info(f"Researcher iteration {iteration + 1}/{max_iterations}")

        # Call the LLM
        response = await llm.ainvoke(session_history)
        response_text = response.content
        state.messages.append(AIMessage(content=response_text, name=f"researcher_iter{iteration+1}"))

        # Check for tool calls
        if "TOOL_CALL:" in response_text and tools:
            tool_results = await execute_tools(response_text, tools)

            # Add tool results to session history for next iteration
            session_history.append(AIMessage(content=response_text))
            session_history.append(HumanMessage(
                content=f"Tool results:\n{tool_results or 'No relevant data'}\n\nChoose the best next course of action based on this output."
            ))
        else:
            # No tool calls -> stop iterating early
            logger.info("No tool calls detected; finishing researcher node early.")
            break

    # Final answer
    final_response = await llm.ainvoke(session_history)
    final_text = final_response.content
    state.research_result = final_text
    state.messages.append(AIMessage(content=final_text, name="researcher_final"))
    state.next_step = "summarizer"
    
    return state


async def sql_agent_node(state: AgentState, tools: List[SimpleTool] = None, max_iterations: int = 3) -> AgentState:
    """
    Research agent node that searches and analyzes files/data.

    Iteratively calls LLM and tools up to `max_iterations`.
    Tracks short-term history to avoid repeating mistakes.

    Args:
        state (AgentState): Current workflow state.
        tools (List[SimpleTool], optional): Tools available for research.
        max_iterations (int, optional): Maximum LLM-tool cycles. Defaults to 3.

    Returns:
        AgentState: Updated state including research_result and message history.
    """
    """SQL agent that queries databases"""
    logger.info("SQL agent starting...")

    tools = [tool for tool in tools if tool.name in ["execute_query", "get_table_schema", "list_catalogs", "list_schemas", "list_tables"]]
    tool_desc = create_tool_descriptions(tools)

    system_message = SystemMessage(
        content=(
            "You are a database specialist.\n"
            "If a query fails, fix it instead of repeating it.\n"
            "Take into account the previous tool result when making the next tool call.\n"
            "Remove semicolons from SQL queries. Use full table names.\n"
            "To use a tool, write: TOOL_CALL: tool_name(arg1=\"value1\", arg2=\"value2\")\n\n"
            f"{tool_desc}\n\n"
        )
    )
    
    instruction = HumanMessage(
        content=f"Answer '{state.topic}' using database tools."
    )

    messages = build_messages_with_history(system_message, state)
    messages.append(instruction)

    response = await llm.ainvoke(messages)
    response_text = response.content
    
    # Keep a session history with the last few messages
    session_history: List[BaseMessage] = [system_message, instruction]

    for iteration in range(max_iterations):
        logger.info(f"SQL agent iteration {iteration + 1}/{max_iterations}")

        # Call the LLM
        response = await llm.ainvoke(session_history)
        response_text = response.content
        state.messages.append(AIMessage(content=response_text, name=f"sql_agent_iter{iteration+1}"))

        # Check for tool calls
        if "TOOL_CALL:" in response_text and tools:
            tool_results = await execute_tools(response_text, tools)

            # Add tool results to session history for next iteration
            session_history.append(AIMessage(content=response_text))
            session_history.append(HumanMessage(
                content=f"Tool results:\n{tool_results or 'No relevant data'}\n\nAnalyze the error messages to improve the next call."
            ))
        else:
            # No tool calls -> stop iterating early
            logger.info("No tool calls detected; finishing sql agent node early.")
            break

    # Final answer
    final_response = await llm.ainvoke(session_history)
    final_text = final_response.content
    state.sql_result = final_text
    state.messages.append(AIMessage(content=response_text))
    state.next_step = "researcher"
    return state


async def summarizer_node(state):
    """
    SQL agent node that queries databases and processes results.

    Iteratively calls LLM and database tools up to `max_iterations`.
    Uses previous tool results to refine queries and avoid repeated errors.

    Args:
        state (AgentState): Current workflow state.
        tools (List[SimpleTool], optional): Tools available for SQL queries.
        max_iterations (int, optional): Maximum LLM-tool cycles. Defaults to 3.

    Returns:
        AgentState: Updated state including sql_result and message history.
    """
    system_message = SystemMessage(content="Provide concise final report.")

    combined = f"""
Research findings:
{state.research_result}

SQL findings:
{state.sql_result}
"""

    messages = build_messages_with_history(system_message, state)
    messages.append(HumanMessage(content=combined))

    response = await llm.ainvoke(messages)

    state.summary = response.content
    state.messages.append(AIMessage(content=response.content))
    state.next_step = "end"
    return state


# ================= WORKFLOW =================

def route_next(state: AgentState):
    """
    Summarizer agent node that compiles research and SQL results into a final report.

    Args:
        state (AgentState): Current workflow state containing research_result and sql_result.

    Returns:
        AgentState: Updated state including the summary.
    """
    return state.next_step


def create_workflow(tools):
    """
    Determines the next node in the workflow based on state.

    Args:
        state (AgentState): Current workflow state.

    Returns:
        str: Name of next node.
    """

    workflow = StateGraph(AgentState)

    async def researcher_wrapper(state: AgentState):
        logger.info("Researcher node executing...")
        return await researcher_node(state, tools, max_iterations)

    async def sql_agent_wrapper(state: AgentState):
        logger.info("SQL agent node executing...")
        return await sql_agent_node(state, tools, max_iterations)

    async def summarizer_wrapper(state: AgentState):
        logger.info("Summarizer node executing...")
        return await summarizer_node(state)

    #
    # Register nodes
    #
    workflow.add_node("sql_agent", sql_agent_wrapper)
    workflow.add_node("researcher", researcher_wrapper)
    workflow.add_node("summarizer", summarizer_wrapper)

    #
    # Edges
    #
    workflow.add_edge(START, "sql_agent")

    workflow.add_conditional_edges(
        "sql_agent",
        route_next,
        {
            "researcher": "researcher",
            "sql_agent": "sql_agent",
            "summarizer": "summarizer",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "researcher",
        route_next,
        {
            "researcher": "researcher",
            "sql_agent": "sql_agent",
            "summarizer": "summarizer",
            "end": END,
        },
    )

    workflow.add_edge("summarizer", END)

    return workflow.compile()


async def run_workflow(topic):
    """
    Builds and compiles the full workflow graph.

    Nodes:
        - researcher
        - sql_agent
        - summarizer

    Edges are configured to follow:
        sql_agent -> researcher -> summarizer -> END
        with conditional loops if necessary.

    Args:
        tools (List[SimpleTool]): Available tools to pass to nodes.

    Returns:
        StateGraph: Compiled workflow ready for execution.
    """
    tools, clients = await load_mcp_tools()
    try:
        state = AgentState(topic=topic, messages=[HumanMessage(content=topic)])
        app = create_workflow(tools)
        return await app.ainvoke(state)
    finally:
        await cleanup_mcp_clients(clients)


async def main(topic: str):
    """
    Executes the workflow for a given topic.

    Args:
        topic (str): Main question or subject for the agents.

    Returns:
        AgentState: Final state after workflow execution.
    """
    """Main entry point"""
    try:
        logger.info(f"Starting workflow for topic: {topic}")
        result = await run_workflow(topic)
        
        logger.info("=" * 80)
        logger.info("FINAL SUMMARY:")
        logger.info("=" * 80)
        logger.info(result['summary'])
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in workflow: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    """
    Main entry point to execute the workflow and log the results.

    Args:
        topic (str): The primary question or task to analyze.

    Returns:
        AgentState: Final agent state including summary.
    """
    TOPIC = "My customer name is John and my flight is A105, what is my status and benefits?"
    asyncio.run(main(TOPIC))
