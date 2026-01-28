import os
import httpx
import logging
import asyncio
from crewai_tools import MCPServerAdapter
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv, find_dotenv
from phoenix.otel import register
from typing import Dict, Any

load_dotenv(find_dotenv())
project_name = "crewai"
tracer_provider = register(project_name=project_name, auto_instrument=True)

custom_headers = {"X-Model-Authorization": f"{os.getenv('NEMO_MAIN_MODEL_TOKEN')}"}
guardrail_config_id = os.getenv("NEMO_NAMESPACE") + '/' + os.getenv("GUARDRAIL_CONFIG_NAME")

# AIE certs
CA_CERT_PATH = "/etc/ezua-domain-ca-certs/ezua-domain-ca-cert.crt"
SSL_CERT_FILE = "/etc/ezua-domain-ca-certs/ezua-domain-ca-cert.crt"
os.environ["REQUESTS_CA_BUNDLE"] = SSL_CERT_FILE
os.environ["SSL_CERT_FILE"] = SSL_CERT_FILE

llm = LLM(
    provider="openai",
    model=os.getenv("NEMO_MAIN_MODEL_ID"),
    base_url=os.getenv("GUARDRAILS_BASE_URL"),
    api_key="dummy",
    extra_headers=custom_headers,
    extra_body={
        "guardrails": {
            "config_id": guardrail_config_id,
        }
    },
)

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

tools_selected = [
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


def run_crew(topic: str):
    with MCPServerAdapter(mcp_server_params, connect_timeout=120) as mcp_tools:
        # Define tool-calling agents
        researcher = Agent(
            role="Researcher",
            goal=(
                "Use available tools analize files and data, both remote and local."
                " Whenever possible, merge data from local filesystems and databases."
            ),
            backstory="A competent researcher.",
            tools=mcp_tools,
            verbose=True,
            llm=llm,
        )

        summarizer = Agent(
            role="Summarizer",
            goal="Summarize content into concise reports.",
            backstory="A competent summarizer.",
            verbose=True,
            llm=llm,
        )

        research = Task(
            description=(
                "Answer '{topic}' based on available data only. "
                "Look for any relevant files to answer the question."
            ),
            expected_output="A detailed, correct, and tool-based solution to '{topic}'.",
            agent=researcher,
        )

        sql = Task(
            description=(
                "Answer '{topic}' based on available databases only. Remove semicolons "
                "from the end of SQL queries and use the full table name. Use available "
                "tools to find the right data sources for the relevant information."
            ),
            expected_output="A detailed, correct, and tool-based solution to '{topic}'.",
            agent=researcher,
        )

        summary = Task(
            description="Summarize the contents.",
            expected_output="A well thought out summary.",
            agent=summarizer,
        )

        # Create a Crew to manage execution
        crew = Crew(
            agents=[
                researcher,
                summarizer,
            ],
            tasks=[research, sql, summary],
            planning=True,
            planning_llm=llm,
            # manager_agent=researcher,
        )

        # Run asynchronously
        result = crew.kickoff(inputs={"topic": topic})

        return result


async def main(topic):

    try:
        result = await asyncio.to_thread(run_crew, topic=topic)

    except Exception as e:
        logger.info(e)


if __name__ == "__main__":
    TOPIC = f"""My customer name is John and my flight is A105, what is my status and benefits?"""
    asyncio.run(main(TOPIC))
