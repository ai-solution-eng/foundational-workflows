"""
LangGraph MCP Agent - Basic Version

This is a basic implementation of a multi-agent workflow using LangGraph and FastMCP.
It converts the CrewAI multi-agent pattern to LangGraph with Pydantic state management.

Features:
- FastMCP client for MCP server connectivity
- Three agents: Researcher, SQL Agent, and Summarizer
- Simple planning with conditional routing
- Automatic MCP tool loading and conversion to LangChain tools

For a more feature-rich version with error handling, configuration, and advanced features,
see langgraph_mcp_agent_enhanced.py
"""
import os
import httpx
import logging
import asyncio
from typing import Dict, Any, List, Annotated, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
from phoenix.otel import register

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from fastmcp import Client

load_dotenv(find_dotenv())
project_name = "langgraph-pydantic"
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


class ResearchOutput(BaseModel):
    """Output schema for research task"""
    findings: str = Field(description="Detailed findings from research")
    sources_used: List[str] = Field(default_factory=list, description="Tools and files used")


class SQLOutput(BaseModel):
    """Output schema for SQL task"""
    query_results: str = Field(description="Results from database queries")
    tables_queried: List[str] = Field(default_factory=list, description="Tables accessed")


class SummaryOutput(BaseModel):
    """Output schema for summary task"""
    summary: str = Field(description="Concise summary of findings")
    key_points: List[str] = Field(default_factory=list, description="Key takeaways")


# Initialize LLM with guardrails
llm = ChatOpenAI(
    model=os.getenv("NEMO_MAIN_MODEL_ID"),
    base_url=os.getenv("GUARDRAILS_BASE_URL"),
    api_key="dummy",
    default_headers=custom_headers,
    model_kwargs={
        "extra_body": {
            "guardrails": {
                "config_id": guardrail_config_id,
            }
        }
    },
)


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

# tools_selected = [
    "create_directory",
    "edit_file",
    "get_file_info",
    "list_allowed_directories",
    "list_directory",
    "list_directory_with_sizes",
    "move_file",
    "read_media_file",
    "read_multiple_files",
    "read_text_file",
    "search_files",
    "write_file",
]


async def load_mcp_tools():
    """Load tools from MCP servers using FastMCP"""
    tools = []
    clients = []
    
    logger.info(f"Loading tools from {len(mcp_server_params)} MCP servers...")
    
    for server_config in mcp_server_params:
        try:
            # Create fastmcp client for each server
            client = Client(server_config["url"])
            await client.__aenter__()
            clients.append(client)
            
            # List available tools from this server
            available_tools = await client.list_tools()
            logger.info(f"Found {len(available_tools)} tools from {server_config['url']}")
            
            # Convert MCP tools to LangChain tools
            for tool_info in available_tools:
                tool_name = tool_info.name
                
                # Filter tools if selection list is provided
                if tools_selected and tool_name not in tools_selected:
                    continue
                
                # Create async function that calls the MCP tool
                async def tool_function(client=client, tool_name=tool_name, **kwargs):
                    try:
                        result = await client.call_tool(tool_name, kwargs)
                        # Extract the actual data from the MCP result
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
                        logger.error(f"Error calling tool {tool_name}: {e}")
                        return f"Error: {str(e)}"
                
                # Create the StructuredTool
                description = tool_info.description or f"MCP tool: {tool_name}"
                langchain_tool = StructuredTool(
                    name=tool_name,
                    description=description,
                    coroutine=tool_function,
                )
                
                tools.append(langchain_tool)
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


# Agent Node Functions
async def researcher_node(state: AgentState, tools: List[BaseTool] = None) -> AgentState:
    """Research agent that analyzes files and data"""
    logger.info("Researcher agent starting...")
    
    system_message = SystemMessage(
        content=(
            "You are a competent researcher. Use available tools to analyze files "
            "and data, both remote and local. Whenever possible, merge data from "
            "local filesystems and databases."
        )
    )
    
    user_message = HumanMessage(
        content=(
            f"Answer '{state.topic}' based on available data only. "
            f"Look for any relevant files to answer the question. "
            f"Use the available tools to search and read files."
        )
    )
    
    messages = [system_message, user_message]
    
    # Invoke LLM with tools if available
    if tools:
        llm_with_tools = llm.bind_tools(tools)
        response = await llm_with_tools.ainvoke(messages)
        
        # Handle tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            messages.append(response)
            
            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_id = tool_call['id']
                
                # Find and execute the tool
                tool_to_use = next((t for t in tools if t.name == tool_name), None)
                if tool_to_use:
                    try:
                        tool_result = await tool_to_use.ainvoke(tool_args)
                        messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_id
                        ))
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
                        messages.append(ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_id
                        ))
            
            # Get final response after tool execution
            response = await llm_with_tools.ainvoke(messages)
    else:
        response = await llm.ainvoke(messages)
    
    state.research_result = response.content
    state.messages.append(AIMessage(content=response.content, name="researcher"))
    state.next_step = "sql_agent"
    
    return state


async def sql_agent_node(state: AgentState, tools: List[BaseTool] = None) -> AgentState:
    """SQL agent that queries databases"""
    logger.info("SQL agent starting...")
    
    system_message = SystemMessage(
        content=(
            "You are a competent researcher specializing in database queries. "
            "Use available tools to find the right data sources for relevant information."
        )
    )
    
    user_message = HumanMessage(
        content=(
            f"Answer '{state.topic}' based on available databases only. "
            f"Remove semicolons from the end of SQL queries and use the full table name. "
            f"Use available tools to find the right data sources."
        )
    )
    
    messages = [system_message, user_message]
    
    # Invoke LLM with tools if available
    if tools:
        llm_with_tools = llm.bind_tools(tools)
        response = await llm_with_tools.ainvoke(messages)
        
        # Handle tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            messages.append(response)
            
            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_id = tool_call['id']
                
                # Find and execute the tool
                tool_to_use = next((t for t in tools if t.name == tool_name), None)
                if tool_to_use:
                    try:
                        tool_result = await tool_to_use.ainvoke(tool_args)
                        messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_id
                        ))
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
                        messages.append(ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_id
                        ))
            
            # Get final response after tool execution
            response = await llm_with_tools.ainvoke(messages)
    else:
        response = await llm.ainvoke(messages)
    
    state.sql_result = response.content
    state.messages.append(AIMessage(content=response.content, name="sql_researcher"))
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
    
    # Invoke LLM
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


def create_workflow(tools: List[BaseTool] = None) -> StateGraph:
    """Create the LangGraph workflow"""
    
    # Create the graph
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
    
    # Load MCP tools
    tools, clients = await load_mcp_tools()
    
    try:
        # Create initial state
        initial_state = AgentState(
            topic=topic,
            messages=[HumanMessage(content=topic)]
        )
        
        # Create and run workflow with tools
        app = create_workflow(tools)
        
        # Execute workflow
        final_state = await app.ainvoke(initial_state)
        
        return final_state
    
    finally:
        # Cleanup MCP clients
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
        raise


if __name__ == "__main__":
    TOPIC = "My customer name is John and my flight is A105, what is my status and benefits?"
    asyncio.run(main(TOPIC))
