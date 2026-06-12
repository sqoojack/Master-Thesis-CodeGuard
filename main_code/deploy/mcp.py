import asyncio
import sys
import argparse
import json
import os

# Resolve path issues so it can find defense module and its dependencies
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
defense_dir = os.path.join(project_root, "defense")
if defense_dir not in sys.path:
    sys.path.append(defense_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from mcp.server.stdio import stdio_server
from mcp.server import Server
import mcp.types as types

# Import the defense framework functions from main.py
from main import (
    build_arg_parser,
    apply_params_file,
    load_model_and_tokenizer,
    build_guardrails,
    run_pipeline
)

# Initialize the defense framework components globally
print("Initializing HiPert defense framework components...", file=sys.stderr)
parser = build_arg_parser()

# Provide fallback arguments for initialization
sys_args = ["--attack_type", "default_attack", "--mode", "all"]
args = parser.parse_args(sys_args)

try:
    args = apply_params_file(args)
except Exception as e:
    print(f"Config file lookup omitted or failed: {e}. Using defaults.", file=sys.stderr)

# Load underlying models and guardrails
model, tokenizer, device = load_model_and_tokenizer(args)
guardrails = build_guardrails(args, model, tokenizer, device)
print("HiPert defense framework initialization complete.", file=sys.stderr)

# Initialize MCP server instances
server = Server("hipert-sanitize-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available sanitization tools for the LLM agent."""
    return [
        types.Tool(
            name="HiPert-Sanitize",
            description="Sanitize external or untrusted code blocks using the 3-layer guardrail system before standard code ingestion.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The raw untrusted code block to filter"},
                    "language": {"type": "string", "description": "The target programming language (e.g., c, python, java, solidity)"}
                },
                "required": ["code", "language"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    """Execute code sanitization pipeline upon tool request."""
    if name != "HiPert-Sanitize":
        raise ValueError(f"Unsupported tool requested: {name}")
    
    if not arguments or "code" not in arguments or "language" not in arguments:
        raise ValueError("Invalid parameters. Both 'code' and 'language' are strictly required.")

    raw_code = arguments["code"]
    language = arguments["language"]

    print(f"Processing inbound tool request for language: {language}", file=sys.stderr)
    pipeline_result = run_pipeline(raw_code, language, guardrails, args)
    sanitized_output = pipeline_result.get("final_code", raw_code)

    return [
        types.TextContent(
            type="text",
            text=sanitized_output
        )
    ]

async def main():
    """Run the MCP server over standard input/output streams."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())