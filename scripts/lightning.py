from langchain_mcp_adapters.client import MultiServerMCPClient


    # MCP Client setup - using test3.py pattern
duck = MultiServerMCPClient({
        "research1": {
            "transport": "stdio",
            "command": "duckduckgo-mcp-server",
            "args": ["serve"],
        }
    })

mili = MultiServerMCPClient({
        "research2": {
            "transport": "stdio",
            "command": "uvx",
            "args": ["-n", "meilisearch-mcp"],
        }
    })