# Import necessary modules and classes for agent, server, and environment setup
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI chat model with specified parameters
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Set up server parameters for Firecrawl MCP tool using environment variables
server_params = StdioServerParameters(
    command="npx",
    env={
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY"),
    },
    args=["firecrawl-mcp"]
)

# Main asynchronous function to run the agent loop
async def main() -> None:
    # Start a client session with the Firecrawl MCP tool
    async with stdio_client(server_params) as (read, write):
        # Establish another session using the read/write streams
        async with ClientSession(read, write) as session:
            # Initialize the session (handshake/setup)
            await session.initialize()
            # Load available MCP tools for the agent
            tools = await load_mcp_tools(session)
            # Create a ReAct agent with the model and loaded tools
            agent = create_react_agent(
                model=model,
                tools=tools
            )
            # Initialize the conversation with a system prompt
            messages = [
                {
                    "role": "system",
                   "content": "You are a helpful assistant that can scrape websites, crawl pages, and extract data using Firecrawl tools. Think step by step and use the appropriate tools to help the user."
                }
            ]
            # Print available tools for user reference
            print("Available Tools -", *[tool.name for tool in tools])
            print("-" * 60)

            # Main interaction loop: get user input, process, and respond
            while True:
                user_input = input("\nYou: ")
                if user_input == "quit":
                    print("Goodbye")
                    break

                # Append user input to the conversation history (truncate if too long)
                messages.append({"role": "user", "content": user_input[:175000]})

                try:
                    # Invoke the agent asynchronously with the current conversation
                    agent_response = await agent.ainvoke({"messages": messages})

                    # Extract and print the agent's latest response
                    ai_message = agent_response["messages"][-1].content
                    print("\nAgent:", ai_message)
                except Exception as e:
                    # Print any errors encountered during agent invocation
                    print("Error:", e)

# Entry point: run the main async function if the script is executed directly
if __name__ == "__main__":
    asyncio.run(main())