"""
LangGraph MCP Agent - No Tool Binding Version

This version works with LLM endpoints that don't support native function calling.
Instead of using bind_tools(), it provides tool descriptions in the prompt and
parses the LLM's response to extract tool calls manually.

Use this version if you get errors about tool format incompatibility.
"""
import os
import httpx
import logging
import asyncio
import json
import re
from typing import Dict, Any, List, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
from phoenix.otel import register

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from fastmcp import Client

load_dotenv(find_dotenv())
project_name = "langgraph"
tracer_provider = register(project_name=project_name, auto_instrument=True)

custom_headers = {"X-Model-Authorization": f"{os.getenv('NEMO_MAIN_MODEL_TOKEN')}"}
guardrail_config_id = os.getenv("NEMO_NAMESPACE") + '/' + os.getenv("GUARDRAIL_CONFIG_NAME")

# AIE certs
CA_CERT_PATH = "/etc/ezua-domain-ca-certs/ezua-domain-ca-cert.crt"
SSL_CERT_FILE = "/etc/ezua-domain-ca-certs/ezua-domain-ca-cert.crt"
os.environ["REQUESTS_CA_BUNDLE"] = SSL_CERT_FILE
os.environ["SSL_CERT_FILE"] = SSL_CERT_FILE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def refresh_token():
    with open("/etc/secrets/ezua/.auth_token", "r") as file:
        token = file.read().strip()
    return token


BEARER = refresh_token()


def auth_client_factory(
    headers: Dict[str, Any] | None = None,
    timeout: float | None = None,
    auth: Any | None = None,
) -> httpx.AsyncClient:
    """Factory function to create an httpx.AsyncClient with an Authorization header."""
    return httpx.AsyncClient(
        headers={"Authorization": f"Bearer {BEARER}"}, timeout=120, auth=None
    )


llm = ChatOpenAI(
    model=os.getenv("NEMO_MAIN_MODEL_ID"),
    base_url=os.getenv("GUARDRAILS_BASE_URL"),
    api_key="dummy",
    default_headers=custom_headers,
    extra_body= {
        "guardrails": {
            "config_id": guardrail_config_id,
        }
    },
    streaming=True
)


# Pydantic Models for State Management
class AgentState(BaseModel):
    """State schema for the LangGraph workflow"""
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    topic: str = ""
    research_result: str = ""
    sql_result: str = ""
    summary: str = ""
    next_step: str = ""
    
    class Config:
        arbitrary_types_allowed = True


# MCP Server Configuration
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

tools_selected = []
# tools_selected = [
#     "create_directory",
#     "edit_file",
#     "get_file_info",
#     "list_allowed_directories",
#     "list_directory",
#     "list_directory_with_sizes",
#     "move_file",
#     "read_media_file",
#     "read_multiple_files",
#     "read_text_file",
#     "search_files",
#     "write_file",
#     "execute_query",
#     "get_table_schema",
#     "list_catalogs",
#     "list_schemas",
#     "list_tables",
# ]


class SimpleTool:
    """Simple tool wrapper that stores MCP tool info"""
    def __init__(self, name: str, description: str, client: Client):
        self.name = name
        self.description = description
        self.client = client
    
    async def execute(self, **kwargs) -> str:
        """Execute the tool"""
        try:
            result = await self.client.call_tool(self.name, kwargs)
            if hasattr(result, 'content'):
                if isinstance(result.content, list):
                    return '\n'.join(
                        item.text if hasattr(item, 'text') else str(item)
                        for item in result.content
                    )
                return str(result.content)
            elif hasattr(result, 'data'):
                return str(result.data)
            return str(result)
        except Exception as e:
            logger.error(f"Error calling tool {self.name}: {e}")
            return f"Error: {str(e)}"


async def load_mcp_tools(selected_tools=None):
    """Load tools from MCP servers"""
    if selected_tools is None:
        selected_tools = tools_selected
    
    tools = []
    clients = []
    
    logger.info(f"Loading tools from {len(mcp_server_params)} MCP servers...")
    
    for server_config in mcp_server_params:
        try:
            client = Client(server_config["url"])
            await client.__aenter__()
            clients.append(client)
            
            available_tools = await client.list_tools()
            logger.info(f"Found {len(available_tools)} tools from {server_config['url']}")
            
            for tool_info in available_tools:
                tool_name = tool_info.name
                
                if selected_tools and tool_name not in selected_tools:
                    continue
                
                description = tool_info.description or f"MCP tool: {tool_name}"
                simple_tool = SimpleTool(tool_name, description, client)
                tools.append(simple_tool)
                logger.info(f"Loaded tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Error connecting to {server_config['url']}: {e}")
    
    logger.info(f"Successfully loaded {len(tools)} tools")
    return tools, clients


async def cleanup_mcp_clients(clients):
    """Disconnect from all MCP servers"""
    logger.info("Disconnecting from MCP servers...")
    for client in clients:
        try:
            await client.__aexit__(None, None, None)
        except Exception as e:
            logger.error(f"Error disconnecting client: {e}")


def create_tool_descriptions(tools: List[SimpleTool]) -> str:
    """Create a formatted string of tool descriptions for the prompt"""
    if not tools:
        return "No tools available."
    
    descriptions = ["Available tools:"]
    for tool in tools:
        descriptions.append(f"- {tool.name}: {tool.description}")
    
    return "\n".join(descriptions)


def parse_tool_calls(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from LLM response.
    Expected format: TOOL_CALL: tool_name(arg1="value1", arg2="value2")
    """
    tool_calls = []
    
    # Look for TOOL_CALL patterns
    pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
    matches = re.finditer(pattern, response_text, re.DOTALL)
    
    for match in matches:
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        
        # Parse arguments
        args = {}
        if args_str:
            # Simple parsing - handles key="value" or key='value'
            arg_pattern = r'(\w+)\s*=\s*["\']([^"\']*)["\']'
            for arg_match in re.finditer(arg_pattern, args_str):
                key, value = arg_match.groups()
                args[key] = value
        
        tool_calls.append({
            "name": tool_name,
            "args": args
        })
    
    return tool_calls


async def execute_tools_from_response(response_text: str, tools: List[SimpleTool]) -> str:
    """Execute tools mentioned in the LLM response"""
    tool_calls = parse_tool_calls(response_text)
    
    if not tool_calls:
        return response_text
    
    results = []
    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        
        # Find the tool
        tool = next((t for t in tools if t.name == tool_name), None)
        if tool:
            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
            result = await tool.execute(**tool_args)
            results.append(f"Tool {tool_name} result:\n{result}")
        else:
            results.append(f"Tool {tool_name} not found")
    
    return "\n\n".join(results)


# Agent Node Functions
async def researcher_node(state: AgentState, tools: List[SimpleTool] = None) -> AgentState:
    """Research agent that analyzes files and data"""
    logger.info("Researcher agent starting...")
    
    tool_desc = create_tool_descriptions(tools) if tools else "No tools available."
    
    system_message = SystemMessage(
        content=(
            "You are a competent researcher. Use available tools to analyze files "
            "and data, both remote and local.\n\n"
            f"{tool_desc}\n\n"
            "To use a tool, write: TOOL_CALL: tool_name(arg1=\"value1\", arg2=\"value2\")\n"
            "You can call multiple tools. After tool results are provided, give your final answer."
        )
    )
    
    user_message = HumanMessage(
        content=(
            f"Answer '{state.topic}' based on available data only. "
            f"Look for any relevant files to answer the question."
        )
    )
    
    messages = [system_message, user_message]
    
    # First LLM call - may contain tool calls
    response = await llm.ainvoke(messages)
    response_text = response.content
    
    # Check for tool calls and execute them
    if "TOOL_CALL:" in response_text and tools:
        tool_results = await execute_tools_from_response(response_text, tools)
        
        # Add tool results and get final answer
        messages.append(AIMessage(content=response_text))
        messages.append(HumanMessage(content=f"Tool results:\n{tool_results}\n\nBased on these results, provide your final answer."))
        
        final_response = await llm.ainvoke(messages)
        response_text = final_response.content
    
    state.research_result = response_text
    state.messages.append(AIMessage(content=response_text, name="researcher"))
    state.next_step = "sql_agent"
    
    return state


async def sql_agent_node(state: AgentState, tools: List[SimpleTool] = None) -> AgentState:
    """SQL agent that queries databases"""
    logger.info("SQL agent starting...")
    
    tool_desc = create_tool_descriptions(tools) if tools else "No tools available."
    
    system_message = SystemMessage(
        content=(
            "You are a competent researcher specializing in database queries.\n\n"
            f"{tool_desc}\n\n"
            "To use a tool, write: TOOL_CALL: tool_name(arg1=\"value1\", arg2=\"value2\")\n"
            "Remove semicolons from SQL queries. Use full table names."
        )
    )
    
    user_message = HumanMessage(
        content=(
            f"Answer '{state.topic}' based on available databases only. "
            f"Use available tools to find the right data sources."
        )
    )
    
    messages = [system_message, user_message]
    
    # First LLM call
    response = await llm.ainvoke(messages)
    response_text = response.content
    
    # Check for tool calls and execute them
    if "TOOL_CALL:" in response_text and tools:
        tool_results = await execute_tools_from_response(response_text, tools)
        
        messages.append(AIMessage(content=response_text))
        messages.append(HumanMessage(content=f"Tool results:\n{tool_results}\n\nBased on these results, provide your final answer."))
        
        final_response = await llm.ainvoke(messages)
        response_text = final_response.content
    
    state.sql_result = response_text
    state.messages.append(AIMessage(content=response_text, name="sql_researcher"))
    state.next_step = "summarizer"
    
    return state


async def summarizer_node(state: AgentState) -> AgentState:
    """Summarizer agent that creates concise reports"""
    logger.info("Summarizer agent starting...")
    
    system_message = SystemMessage(
        content="You are a competent summarizer. Create concise, well-thought-out summaries."
    )
    
    combined_content = f"""
Research findings:
{state.research_result}

SQL findings:
{state.sql_result}

Please summarize the above contents into a coherent report.
"""
    
    user_message = HumanMessage(content=combined_content)
    messages = [system_message, user_message]
    
    response = await llm.ainvoke(messages)
    
    state.summary = response.content
    state.messages.append(AIMessage(content=response.content, name="summarizer"))
    state.next_step = "end"
    
    return state


async def planning_node(state: AgentState) -> AgentState:
    """Planning node that orchestrates the workflow"""
    logger.info("Planning workflow...")
    
    system_message = SystemMessage(
        content=(
            "You are a planning agent. Analyze the user's question and determine "
            "the best approach to answer it using research and database queries."
        )
    )
    
    user_message = HumanMessage(
        content=f"Create a plan to answer: {state.topic}"
    )
    
    messages = [system_message, user_message]
    response = await llm.ainvoke(messages)
    
    state.messages.append(AIMessage(content=response.content, name="planner"))
    state.next_step = "researcher"
    
    return state


def route_next(state: AgentState) -> Literal["researcher", "sql_agent", "summarizer", "end"]:
    """Router function to determine next node"""
    if state.next_step == "researcher":
        return "researcher"
    elif state.next_step == "sql_agent":
        return "sql_agent"
    elif state.next_step == "summarizer":
        return "summarizer"
    else:
        return "end"


def create_workflow(tools: List[SimpleTool] = None) -> StateGraph:
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Create node functions with tools bound
    async def researcher_with_tools(state):
        return await researcher_node(state, tools)
    
    async def sql_agent_with_tools(state):
        return await sql_agent_node(state, tools)
    
    # Add nodes
    workflow.add_node("planner", planning_node)
    workflow.add_node("researcher", researcher_with_tools)
    workflow.add_node("sql_agent", sql_agent_with_tools)
    workflow.add_node("summarizer", summarizer_node)
    
    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        route_next,
        {
            "researcher": "researcher",
            "sql_agent": "sql_agent",
            "summarizer": "summarizer",
            "end": END,
        }
    )
    workflow.add_conditional_edges(
        "researcher",
        route_next,
        {
            "researcher": "researcher",
            "sql_agent": "sql_agent",
            "summarizer": "summarizer",
            "end": END,
        }
    )
    workflow.add_conditional_edges(
        "sql_agent",
        route_next,
        {
            "researcher": "researcher",
            "sql_agent": "sql_agent",
            "summarizer": "summarizer",
            "end": END,
        }
    )
    workflow.add_edge("summarizer", END)
    
    return workflow.compile()


async def run_workflow(topic: str) -> AgentState:
    """Run the LangGraph workflow"""
    tools, clients = await load_mcp_tools()
    
    try:
        initial_state = AgentState(
            topic=topic,
            messages=[HumanMessage(content=topic)]
        )
        
        app = create_workflow(tools)
        final_state = await app.ainvoke(initial_state)
        
        return final_state
    
    finally:
        await cleanup_mcp_clients(clients)


async def main(topic: str):
    """Main entry point"""
    try:
        logger.info(f"Starting workflow for topic: {topic}")
        result = await run_workflow(topic)
        
        logger.info("=" * 80)
        logger.info("FINAL SUMMARY:")
        logger.info("=" * 80)
        logger.info(result.summary)
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in workflow: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    TOPIC = "My customer name is John and my flight is A105, what is my status and benefits?"
    asyncio.run(main(TOPIC))
