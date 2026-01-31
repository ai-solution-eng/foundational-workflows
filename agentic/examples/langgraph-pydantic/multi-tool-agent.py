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
from langchain_core.tools import BaseTool, StructuredTool, tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
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


# Pydantic Models for State Management
class AgentState(BaseModel):
    """
    State schema for the LangGraph workflow.
    
    This Pydantic model defines the complete state that flows through the workflow graph.
    Each node receives the current state, processes it, and returns an updated state.
    
    Attributes:
        messages (List[BaseMessage]): Conversation history including system, user, AI, and tool messages.
            Uses the add_messages annotation for automatic message list merging.
            Messages accumulate as the workflow progresses through different agents.
        
        topic (str): The original user question or task to be answered.
            This remains constant throughout the workflow and guides all agent actions.
        
        research_result (str): Output from the researcher agent.
            Contains findings from file searches, document reads, and other research tools.
            Empty string by default until populated by researcher_node.
        
        sql_result (str): Output from the SQL agent.
            Contains results from database queries and data analysis.
            Empty string by default until populated by sql_agent_node.
        
        summary (str): Final synthesized answer from the summarizer agent.
            Combines research_result and sql_result into a coherent response.
            This is the final output presented to the user.
        
        next_step (str): Routing control for the workflow.
            Determines which agent node to execute next.
            Values: "researcher", "sql_agent", "summarizer", or "end"
    
    Config:
        arbitrary_types_allowed: Allows LangChain BaseMessage types in the model.
    
    Example:
        >>> state = AgentState(topic="What files are in /data?")
        >>> state.research_result = "Found 3 files: a.txt, b.csv, c.json"
        >>> state.next_step = "summarizer"
    """
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
    extra_body= {
        "guardrails": {
            "config_id": guardrail_config_id,
        }
    }
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
# ]


async def load_mcp_tools(selected_tools=None):
    """
    Load and convert tools from MCP servers to LangChain StructuredTools.
    
    This function connects to all configured MCP servers, discovers available tools,
    filters them based on the selection criteria, and converts them to LangChain-compatible
    tools that can be bound to LLMs for agent execution.
    
    Args:
        selected_tools (List[str], optional): List of tool names to load. If None, uses
            the global tools_selected list. Only tools whose names appear in this list
            will be loaded. Use None or empty list to load all available tools.
    
    Returns:
        Tuple[List[BaseTool], List[Client]]: A tuple containing:
            - tools (List[BaseTool]): List of LangChain StructuredTool objects ready to
              be bound to LLMs. Each tool wraps an MCP tool call.
            - clients (List[Client]): List of active FastMCP Client instances. These must
              be cleaned up later using cleanup_mcp_clients() to properly close connections.
    
    Process Flow:
        1. Initialize empty lists for tools and clients
        2. For each MCP server in mcp_server_params:
           a. Create FastMCP Client with server URL
           b. Connect to server using async context manager entry
           c. List all available tools from the server
           d. For each tool that matches selected_tools filter:
              - Create a LangChain StructuredTool wrapper
              - Use function factory pattern to avoid closure issues
              - Add tool to the tools list
        3. Log success/failure for each server connection
        4. Return all loaded tools and client connections
    
    Tool Function Factory Pattern:
        Uses make_tool_function(client_ref, name) to create properly scoped async functions.
        This avoids Python closure issues where loop variables are captured incorrectly.
        Each tool function:
        - Calls the MCP server's tool via client.call_tool()
        - Extracts content from various MCP result formats
        - Handles errors and returns error messages on failure
    
    Error Handling:
        - Connection errors are logged but don't stop the loading process
        - Other servers continue to load even if one fails
        - Tool execution errors are caught and returned as error strings
    
    Example:
        >>> tools, clients = await load_mcp_tools(["read_file", "search_files"])
        INFO:__main__:Loading tools from 3 MCP servers...
        INFO:__main__:Found 13 tools from https://filesystem-server/mcp
        INFO:__main__:Loaded tool: read_file
        INFO:__main__:Loaded tool: search_files
        INFO:__main__:Successfully loaded 2 tools
        >>> len(tools)
        2
        >>> # Later, cleanup:
        >>> await cleanup_mcp_clients(clients)
    
    Notes:
        - Requires active MCP servers at the URLs in mcp_server_params
        - Connections remain open until cleanup_mcp_clients() is called
        - Tool descriptions come from MCP tool metadata
        - Large tool lists may take several seconds to load
    """
    if selected_tools is None:
        selected_tools = tools_selected
    
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
                if selected_tools and tool_name not in selected_tools:
                    continue
                
                # Create a function factory to avoid closure issues
                def make_tool_function(client_ref, name):
                    """
                    Factory function to create properly scoped async tool functions.
                    
                    This pattern is necessary to avoid Python closure issues where loop
                    variables (client, tool_name) would be captured by reference instead
                    of by value, causing all tools to reference the last loop iteration.
                    
                    Args:
                        client_ref (Client): The FastMCP client instance for this tool
                        name (str): The tool name to call on the MCP server
                    
                    Returns:
                        Callable: An async function that executes the MCP tool
                    """
                    async def tool_function(**kwargs) -> str:
                        """
                        Execute the MCP tool and return results.
                        
                        Args:
                            **kwargs: Tool-specific arguments passed from the LLM
                        
                        Returns:
                            str: Tool execution result or error message
                        """
                        try:
                            result = await client_ref.call_tool(name, kwargs)
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
                            logger.error(f"Error calling tool {name}: {e}")
                            return f"Error: {str(e)}"
                    return tool_function
                
                # Create the LangChain tool using StructuredTool with explicit schema
                description = tool_info.description or f"MCP tool: {tool_name}"
                
                # Create a minimal Pydantic model for args_schema
                from pydantic import create_model
                
                # Create dynamic args schema - accept any kwargs
                ArgsSchema = create_model(
                    f"{tool_name}_args",
                    __config__=None,
                )
                
                try:
                    langchain_tool = StructuredTool(
                        name=tool_name,
                        description=description,
                        coroutine=make_tool_function(client, tool_name),
                        args_schema=ArgsSchema,
                    )
                    tools.append(langchain_tool)
                    logger.info(f"Loaded tool: {tool_name}")
                except Exception as tool_error:
                    logger.warning(f"Failed to create StructuredTool for {tool_name}, trying alternate method: {tool_error}")
                    # Fallback: use the @tool decorator approach
                    func = make_tool_function(client, tool_name)
                    func.__name__ = tool_name
                    func.__doc__ = description
                    
                    # Use tool decorator
                    langchain_tool = tool(func)
                    tools.append(langchain_tool)
                    logger.info(f"Loaded tool (alternate): {tool_name}")
                
        except Exception as e:
            logger.error(f"Error connecting to {server_config['url']}: {e}")
    
    logger.info(f"Successfully loaded {len(tools)} tools")
    return tools, clients


async def cleanup_mcp_clients(clients):
    """
    Disconnect from all MCP servers and clean up client connections.
    
    This function properly closes all FastMCP client connections that were opened
    during tool loading. It should always be called after the workflow completes
    to prevent resource leaks and ensure graceful shutdown.
    
    Args:
        clients (List[Client]): List of FastMCP Client instances to disconnect.
            Typically this is the second element of the tuple returned by load_mcp_tools().
    
    Returns:
        None
    
    Process:
        For each client in the list:
        1. Call the async context manager exit method (__aexit__)
        2. Handle any disconnection errors gracefully
        3. Log errors but continue disconnecting remaining clients
    
    Error Handling:
        Disconnection errors are logged but don't raise exceptions, ensuring
        all clients get a chance to disconnect even if some fail.
    
    Example:
        >>> tools, clients = await load_mcp_tools()
        >>> # ... use tools in workflow ...
        >>> await cleanup_mcp_clients(clients)
        INFO:__main__:Disconnecting from MCP servers...
        
    Notes:
        - Safe to call even if some clients are already disconnected
        - Should be called in a try/finally block to ensure cleanup happens
        - Part of the cleanup process in run_workflow()
    """
    logger.info("Disconnecting from MCP servers...")
    for client in clients:
        try:
            await client.__aexit__(None, None, None)
        except Exception as e:
            logger.error(f"Error disconnecting client: {e}")


# Agent Node Functions
async def researcher_node(state: AgentState, tools: List[BaseTool] = None) -> AgentState:
    """
    Research agent that analyzes files and data using available MCP tools.
    
    This node is responsible for gathering information from file systems, documents,
    and other data sources to answer the user's question. It uses LLM-driven tool
    selection to decide which tools to call and how to use them.
    
    Args:
        state (AgentState): Current workflow state containing:
            - topic: The user's question to research
            - messages: Conversation history
            - Other state fields (not modified by this node)
        
        tools (List[BaseTool], optional): List of LangChain tools (from MCP servers)
            that the agent can use. If None, the agent operates without tools.
            Tools typically include: read_file, search_files, list_directory, etc.
    
    Returns:
        AgentState: Updated state with:
            - research_result: String containing findings from file/data analysis
            - messages: Appended with AI response from researcher
            - next_step: Set to "sql_agent" to continue workflow
    
    Process Flow:
        1. Create system message defining researcher role and capabilities
        2. Create user message with the research task (state.topic)
        3. If tools are available:
           a. Bind tools to LLM
           b. Invoke LLM - may return tool calls
           c. Execute each tool call:
              - Find the tool by name
              - Call tool.ainvoke() with arguments
              - Add ToolMessage with results
           d. Get final response from LLM after tool execution
        4. If no tools, invoke LLM directly for text-only response
        5. Store result in state.research_result
        6. Add researcher's message to conversation history
        7. Set next_step to "sql_agent"
    
    Tool Execution:
        When the LLM decides to use tools, it returns tool_calls in the response.
        Each tool call contains:
        - name: Tool to execute (e.g., "read_text_file")
        - args: Dictionary of arguments (e.g., {"path": "/data/file.txt"})
        - id: Unique identifier for matching tool results
        
        The node executes these tools and adds ToolMessage objects to the conversation,
        then invokes the LLM again to synthesize the tool results into an answer.
    
    Error Handling:
        - Tool execution errors are caught and returned as error messages
        - Errors don't stop the workflow; partial results are still useful
        - All errors are logged for debugging
    
    Example Execution:
        >>> state = AgentState(topic="What files are in /data?")
        >>> tools = [list_directory_tool, read_file_tool]
        >>> updated_state = await researcher_node(state, tools)
        >>> print(updated_state.research_result)
        "Found 3 files in /data: config.json, data.csv, readme.txt"
        >>> print(updated_state.next_step)
        "sql_agent"
    
    Notes:
        - This is a LangGraph node function, called by the workflow graph
        - Must return updated state (not create new state)
        - Tool calls are LLM-driven; agent decides which tools to use
        - May require multiple LLM calls if tools are used
    """
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
    """
    SQL/Database agent that queries databases and analyzes structured data.
    
    This node specializes in database queries and data analysis. It uses MCP tools
    to connect to databases, execute queries, and retrieve structured information
    that complements the file-based research from the researcher_node.
    
    Args:
        state (AgentState): Current workflow state containing:
            - topic: The user's question to answer
            - research_result: Findings from researcher_node (for context)
            - messages: Conversation history
        
        tools (List[BaseTool], optional): List of LangChain tools from MCP servers.
            Database-related tools might include: query_database, get_schema,
            list_tables, execute_sql, etc.
    
    Returns:
        AgentState: Updated state with:
            - sql_result: String containing database query results and analysis
            - messages: Appended with AI response from SQL agent
            - next_step: Set to "summarizer" to proceed to final summarization
    
    Process Flow:
        1. Create system message defining SQL specialist role
        2. Create user message with database query task
        3. If tools are available:
           a. Bind tools to LLM
           b. Invoke LLM - may return tool calls for database operations
           c. Execute each tool call (database queries, schema inspection, etc.)
           d. Add ToolMessage objects with query results
           e. Get final response synthesizing the database findings
        4. If no tools, invoke LLM for text-only response
        5. Store result in state.sql_result
        6. Add SQL agent's message to conversation history
        7. Set next_step to "summarizer"
    
    Special Instructions:
        - Removes semicolons from SQL queries (MCP compatibility requirement)
        - Uses full table names in queries (schema.table format)
        - Focuses on structured data that complements file-based research
        - Can access research_result to avoid duplicate work
    
    Tool Execution Pattern:
        Similar to researcher_node, but typically calls database-specific tools:
        - List available tables/schemas
        - Execute SELECT queries
        - Retrieve table metadata
        - Join data from multiple sources
    
    Error Handling:
        - Database connection errors are caught and logged
        - Invalid SQL syntax errors are returned to LLM for correction
        - Partial results are preserved even if some queries fail
    
    Example Execution:
        >>> state = AgentState(
        ...     topic="What's the revenue for customer John?",
        ...     research_result="Customer John has account ID 12345"
        ... )
        >>> tools = [query_database_tool, get_schema_tool]
        >>> updated_state = await sql_agent_node(state, tools)
        >>> print(updated_state.sql_result)
        "Customer John (ID: 12345) has total revenue of $45,230 from 15 orders"
        >>> print(updated_state.next_step)
        "summarizer"
    
    Notes:
        - This is the second agent in the workflow pipeline
        - Can leverage research findings to make better queries
        - SQL results will be combined with research in the summarizer
        - Should focus on database-specific information, not file content
    """
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
    """
    Summarizer agent that synthesizes findings into a coherent final answer.
    
    This node is the final step in the workflow. It takes the outputs from both
    the researcher and SQL agents and combines them into a well-structured,
    concise summary that directly answers the user's original question.
    
    Args:
        state (AgentState): Current workflow state containing:
            - topic: The original user question
            - research_result: Findings from file/document research
            - sql_result: Findings from database queries
            - messages: Full conversation history
    
    Returns:
        AgentState: Updated state with:
            - summary: Final synthesized answer combining all findings
            - messages: Appended with summarizer's final response
            - next_step: Set to "end" to terminate the workflow
    
    Process Flow:
        1. Create system message defining summarizer role
        2. Construct combined content from research_result and sql_result
        3. Create user message asking for summary
        4. Invoke LLM (no tools needed for summarization)
        5. Store response in state.summary
        6. Add summarizer's message to conversation history
        7. Set next_step to "end" to signal workflow completion
    
    Summarization Strategy:
        The LLM is instructed to:
        - Synthesize information from both agents
        - Focus on answering the original question
        - Present information coherently
        - Avoid unnecessary repetition
        - Highlight key findings
    
    Input Processing:
        Both research_result and sql_result are provided to the LLM in a
        structured format, clearly labeled so the LLM can:
        - Identify which findings came from files vs. databases
        - Resolve conflicts or redundancies
        - Fill gaps where one source lacks information
        - Cross-reference findings for validation
    
    Example Execution:
        >>> state = AgentState(
        ...     topic="What is customer John's flight status?",
        ...     research_result="Found customer file: John, flight A105",
        ...     sql_result="Database shows flight A105 is delayed 30 minutes"
        ... )
        >>> updated_state = await summarizer_node(state)
        >>> print(updated_state.summary)
        "Customer John is booked on flight A105, which is currently delayed
        by 30 minutes. The flight information was confirmed in both the
        customer records and the flight database."
        >>> print(updated_state.next_step)
        "end"
    
    Notes:
        - This is the terminal node in the workflow
        - Does not use tools (pure LLM summarization)
        - The summary is the final output presented to the user
        - Should be clear, concise, and directly answer the question
        - Handles cases where one agent failed or returned no results
    """
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
    """
    Planning agent that analyzes the question and creates an execution strategy.
    
    This is the first node executed in the workflow. It analyzes the user's question
    and determines the best approach to answer it, though in this basic version it
    primarily serves to initialize the workflow and set the next step.
    
    Args:
        state (AgentState): Current workflow state containing:
            - topic: The user's original question
            - messages: Initial conversation (usually just the user's question)
    
    Returns:
        AgentState: Updated state with:
            - messages: Appended with planner's strategic analysis
            - next_step: Set to "researcher" to begin the research phase
    
    Process Flow:
        1. Create system message defining planning role
        2. Create user message asking for a plan
        3. Invoke LLM to generate strategic approach
        4. Add plan to conversation history
        5. Set next_step to "researcher" to start execution
    
    Planning Strategy:
        The planner considers:
        - What information is needed to answer the question
        - Which data sources might contain relevant information
        - Whether file research or database queries are needed
        - The order in which to gather information
    
    Current Implementation:
        In this basic version, planning is minimal - it mainly serves to:
        - Document the initial approach
        - Provide context to downstream agents
        - Initialize the workflow routing
        
        More advanced versions could:
        - Dynamically choose which agents to use
        - Skip agents if not needed
        - Determine parallel vs. sequential execution
        - Set timeouts or iteration limits
    
    Example Execution:
        >>> state = AgentState(topic="What are customer John's benefits?")
        >>> updated_state = await planning_node(state)
        >>> # Planner's message added to conversation
        >>> print(updated_state.messages[-1].content)
        "To answer this question, I'll:
        1. Search customer files for John's account details
        2. Query the benefits database for his account
        3. Synthesize the findings into a summary"
        >>> print(updated_state.next_step)
        "researcher"
    
    Notes:
        - Always sets next_step to "researcher" (fixed routing)
        - Does not use tools (planning is LLM reasoning only)
        - Could be extended to support dynamic routing
        - Plan is stored in messages but not explicitly tracked in state
    """
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
    """
    Router function to determine the next node in the workflow graph.
    
    This function implements the conditional routing logic for the LangGraph workflow.
    It examines the current state and returns the name of the next node to execute,
    enabling dynamic control flow through the agent pipeline.
    
    Args:
        state (AgentState): Current workflow state. The routing decision is based
            on the state.next_step field, which is set by each agent node.
    
    Returns:
        Literal["researcher", "sql_agent", "summarizer", "end"]: The name of the
            next node to execute. Must match one of the node names defined in the
            workflow graph. Possible values:
            - "researcher": Route to researcher_node for file/data research
            - "sql_agent": Route to sql_agent_node for database queries
            - "summarizer": Route to summarizer_node for final synthesis
            - "end": Terminate workflow (maps to END in LangGraph)
    
    Routing Logic:
        The function uses simple string matching on state.next_step:
        - "researcher" → researcher_node (gather file/document data)
        - "sql_agent" → sql_agent_node (query databases)
        - "summarizer" → summarizer_node (create final summary)
        - Any other value → "end" (terminate workflow)
    
    Typical Flow:
        planner → researcher → sql_agent → summarizer → end
        
        Each node sets next_step to determine where to go next:
        1. planning_node sets next_step = "researcher"
        2. researcher_node sets next_step = "sql_agent"
        3. sql_agent_node sets next_step = "summarizer"
        4. summarizer_node sets next_step = "end"
    
    Usage in Graph:
        This function is passed to add_conditional_edges() in create_workflow():
        ```python
        workflow.add_conditional_edges(
            "researcher",  # From node
            route_next,    # This router function
            {              # Mapping of return values to target nodes
                "researcher": "researcher",
                "sql_agent": "sql_agent",
                "summarizer": "summarizer",
                "end": END
            }
        )
        ```
    
    Example:
        >>> state = AgentState(next_step="sql_agent")
        >>> next_node = route_next(state)
        >>> print(next_node)
        "sql_agent"
        
        >>> state.next_step = "end"
        >>> next_node = route_next(state)
        >>> print(next_node)
        "end"
    
    Notes:
        - Must be a regular function (not async) for LangGraph compatibility
        - Return type annotation is enforced by LangGraph for type safety
        - Could be extended to support loops (agent calling itself)
        - Could implement more complex routing logic based on state content
        - Default case ("end") handles any unexpected next_step values
    """
    if state.next_step == "researcher":
        return "researcher"
    elif state.next_step == "sql_agent":
        return "sql_agent"
    elif state.next_step == "summarizer":
        return "summarizer"
    else:
        return "end"


def create_workflow(tools: List[BaseTool] = None) -> StateGraph:
    """
    Create and compile the LangGraph workflow with all agent nodes and routing logic.
    
    This function builds the complete workflow graph that orchestrates the multi-agent
    system. It defines nodes (agents), edges (transitions), and routing logic to create
    a directed graph representing the agent execution flow.
    
    Args:
        tools (List[BaseTool], optional): List of MCP tools converted to LangChain format.
            These tools are bound to the researcher and SQL agent nodes so they can
            interact with external data sources. If None, agents run without tools.
    
    Returns:
        StateGraph: A compiled LangGraph application ready to execute. The compiled
            graph can be invoked with an initial state to run the complete workflow.
    
    Workflow Structure:
        START → planner → researcher → sql_agent → summarizer → END
        
        Nodes:
        - planner: Analyzes question and creates execution strategy
        - researcher: Gathers information from files and documents
        - sql_agent: Queries databases for structured data
        - summarizer: Synthesizes all findings into final answer
        
        Edges:
        - Fixed: START → planner (always starts with planning)
        - Fixed: summarizer → END (always ends after summary)
        - Conditional: planner → [researcher|sql_agent|summarizer|end]
        - Conditional: researcher → [researcher|sql_agent|summarizer|end]
        - Conditional: sql_agent → [researcher|sql_agent|summarizer|end]
    
    Tool Integration:
        Tools are bound to specific nodes using wrapper functions:
        - researcher_with_tools: Wraps researcher_node with tools
        - sql_agent_with_tools: Wraps sql_agent_node with tools
        
        This pattern allows nodes to access tools while maintaining clean signatures.
    
    Routing Logic:
        Conditional edges use the route_next() function to determine transitions.
        Each agent node sets state.next_step, which route_next() reads to decide
        the next destination. This enables:
        - Linear progression (planner → researcher → sql → summarizer)
        - Potential loops (agent could call itself if needed)
        - Early termination (any node can set next_step="end")
    
    Graph Compilation:
        The workflow.compile() call:
        - Validates the graph structure (no orphaned nodes, etc.)
        - Optimizes execution paths
        - Creates an executable application
        - Returns a runnable that accepts initial state
    
    Example Usage:
        >>> tools = await load_mcp_tools()
        >>> app = create_workflow(tools)
        >>> initial_state = AgentState(topic="What files exist?")
        >>> final_state = await app.ainvoke(initial_state)
        >>> print(final_state.summary)
    
    Advanced Features (not implemented in basic version):
        - Parallel execution of researcher and sql_agent
        - Checkpointing for long-running workflows
        - Human-in-the-loop approval steps
        - Dynamic agent selection based on question type
        - Retry logic for failed nodes
    
    Notes:
        - Graph structure is fixed at compile time
        - Tools must be provided before compilation
        - Compiled graph is reusable for multiple invocations
        - Each invocation gets independent state
        - Graph is acyclic in this version (no infinite loops possible)
    """
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Create node functions with tools bound
    async def researcher_with_tools(state):
        """Wrapper that passes tools to researcher_node"""
        return await researcher_node(state, tools)
    
    async def sql_agent_with_tools(state):
        """Wrapper that passes tools to sql_agent_node"""
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
    """
    Execute the complete multi-agent workflow for a given question.
    
    This is the main execution function that orchestrates the entire workflow from
    start to finish. It handles MCP tool loading, workflow execution, and cleanup,
    providing a complete end-to-end solution for answering user questions.
    
    Args:
        topic (str): The user's question or task to be answered. This should be a
            clear, specific question that can be answered using file research and/or
            database queries. Examples:
            - "What files are in the /data directory?"
            - "What is customer John's account status?"
            - "Find all orders from last month"
    
    Returns:
        AgentState: The final workflow state containing:
            - summary: The complete answer to the user's question
            - research_result: Raw findings from file/document research
            - sql_result: Raw findings from database queries
            - messages: Complete conversation history
            - topic: The original question
            
    Process Flow:
        1. Load MCP Tools:
           - Connect to all configured MCP servers
           - Discover and filter available tools
           - Convert to LangChain format
           - Store client connections for later cleanup
        
        2. Initialize State:
           - Create initial AgentState with user's question
           - Add question as first message in conversation
        
        3. Build and Execute Workflow:
           - Compile workflow graph with loaded tools
           - Execute graph starting from initial state
           - Progress through: planner → researcher → sql_agent → summarizer
           - Each node updates state and determines next step
        
        4. Cleanup:
           - Disconnect all MCP client connections
           - Release resources even if workflow fails
           - Ensure no lingering connections
        
        5. Return Results:
           - Return final state with complete answer
    
    Resource Management:
        Uses try/finally to ensure MCP clients are always cleaned up, even if
        the workflow fails or raises an exception. This prevents resource leaks
        and ensures proper connection closure.
    
    Error Handling:
        - MCP connection errors are logged in load_mcp_tools()
        - Agent execution errors are handled by individual nodes
        - Cleanup errors are logged but don't prevent function return
        - Workflow continues even if some tools/servers fail
    
    Example Usage:
        >>> result = await run_workflow("What is customer John's status?")
        INFO:__main__:Loading tools from 3 MCP servers...
        INFO:__main__:Successfully loaded 12 tools
        INFO:__main__:Researcher agent starting...
        INFO:__main__:SQL agent starting...
        INFO:__main__:Summarizer agent starting...
        INFO:__main__:Disconnecting from MCP servers...
        
        >>> print(result.summary)
        "Customer John has an active premium account with flight A105 
        departing at 14:30. His benefits include lounge access and 
        priority boarding."
        
        >>> print(result.research_result)
        "Found customer file: John Doe, Account #12345, Premium tier"
        
        >>> print(result.sql_result)
        "Database query returned: Flight A105, Status: On-time, 
        Departure: 14:30"
    
    Performance Considerations:
        - Tool loading typically takes 1-3 seconds
        - Workflow execution time depends on:
          * Number of tool calls made
          * LLM response time
          * Database query complexity
          * File system access speed
        - Typical total time: 10-30 seconds for complex queries
        - Can be optimized by reducing tool selection or using faster models
    
    Notes:
        - This function is async and must be awaited
        - Creates new MCP connections for each invocation
        - Workflow state is independent per invocation
        - Can be called multiple times concurrently
        - Tools are loaded fresh each time (no caching)
    """
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
