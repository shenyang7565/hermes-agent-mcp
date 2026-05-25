#!/usr/bin/env python3
"""
DeepSeek-MCP-Server — MCP stdio Server using MiniMax API (OpenAI-compatible)
注册到 Hermes: hermes mcp add deepseek /usr/bin/python3 --args /path/to/deepseek_mcp_server.py
"""
import os
import sys
import json
import asyncio
from typing import Any, Dict, List, Optional

# MiniMax OpenAI-compatible config
API_KEY = os.environ.get("MINIMAX_CN_API_KEY", "")
BASE_URL = os.environ.get("MINIMAX_CN_BASE_URL", "https://api.minimaxi.com/v1")
MODEL = os.environ.get("MINIMAX_MCP_MODEL", "MiniMax-M2.7")


def _chat_sync(messages: List[Dict], model: str = None, **kwargs) -> Dict:
    """Synchronous chat call via requests (avoids Windows urllib idna bug)."""
    import requests
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": model or MODEL,
        "messages": messages,
        **{k: v for k, v in kwargs.items() if v is not None}
    }
    resp = requests.post(
        url,
        json=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


async def _chat(messages: List[Dict], model: str = None, **kwargs) -> Dict:
    """Async wrapper — runs sync call in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _chat_sync(messages, model, **kwargs))


# ─── MCP Protocol ──────────────────────────────────────────────────────────────

async def handle_request(request: Dict) -> Optional[Dict]:
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    # JSON-RPC error helper
    def err(code: int, msg: str, data: Any = None):
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": code, "message": msg, "data": data}}

    def resp(result: Any):
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    # ── Notifications we handle ──────────────────────────────────────────────
    if method == "initialize":
        return resp({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {"name": "deepseek-mcp", "version": "1.0.0"},
        })

    if method == "notifications/initialized":
        return None  # ack, no response

    if method == "tools/list":
        return resp({
            "tools": [
                {
                    "name": "deepseek_chat",
                    "description": "Send a message to DeepSeek/MiniMax chat model and get a response. "
                                  "Use for coding, analysis, writing, reasoning, and general Q&A.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The user's message or question"
                            },
                            "system_prompt": {
                                "type": "string",
                                "description": "Optional system prompt to set context/behavior"
                            },
                            "model": {
                                "type": "string",
                                "description": "Model to use (default: MiniMax-M2.7)"
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Sampling temperature 0-2 (default: 0.7)"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Max response tokens (default: 4096)"
                            },
                        },
                        "required": ["message"]
                    }
                },
                {
                    "name": "deepseek_code",
                    "description": "Write or edit code with DeepSeek/MiniMax. "
                                  "Provides stronger coding focus than deepseek_chat.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "Coding task description — e.g. 'write a fast fibonacci function in Python'"
                            },
                            "language": {
                                "type": "string",
                                "description": "Target programming language (python, javascript, rust, go, etc.)"
                            },
                            "model": {
                                "type": "string",
                                "description": "Model override"
                            },
                        },
                        "required": ["task"]
                    }
                }
            ]
        })

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        try:
            if tool_name == "deepseek_chat":
                messages = []
                if tool_args.get("system_prompt"):
                    messages.append({"role": "system", "content": tool_args["system_prompt"]})
                messages.append({"role": "user", "content": tool_args["message"]})
                result = await _chat(
                    messages,
                    model=tool_args.get("model"),
                    temperature=tool_args.get("temperature", 0.7),
                    max_tokens=tool_args.get("max_tokens", 4096),
                )
                content = result["choices"][0]["message"]["content"]
                return resp({
                    "content": [{"type": "text", "text": content}],
                    "isError": False
                })

            elif tool_name == "deepseek_code":
                # Build a coding-focused prompt
                lang = tool_args.get("language", "")
                task = tool_args["task"]
                system = (
                    "You are an expert programmer. Write clean, efficient, well-commented code. "
                    f"Language: {lang}" if lang else "Write clean, efficient, well-commented code."
                )
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": task}
                ]
                result = await _chat(
                    messages,
                    model=tool_args.get("model"),
                    temperature=0.3,
                    max_tokens=8192,
                )
                content = result["choices"][0]["message"]["content"]
                return resp({
                    "content": [{"type": "text", "text": content}],
                    "isError": False
                })

            else:
                return err(-32601, f"Unknown tool: {tool_name}")

        except Exception as e:
            return resp({
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True
            })

    if method in ("ping", "shutdown"):
        return resp({"pong" if method == "ping" else None})

    return err(-32601, f"Method not found: {method}")


# ─── Main Loop ────────────────────────────────────────────────────────────────

async def main():
    # Validate config
    if not API_KEY:
        print("ERROR: MINIMAX_CN_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    write_lock = asyncio.Lock()

    async def send_response(data: Dict):
        line = json.dumps(data) + "\n"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, sys.stdout.write, line)
        await loop.run_in_executor(None, sys.stdout.flush)

    while True:
        try:
            line_bytes = await reader.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8").strip()
            if not line:
                continue
            request = json.loads(line)
            response = await handle_request(request)
            if response is not None:
                await send_response(response)
        except json.JSONDecodeError as e:
            err_resp = {"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32700, "message": f"Parse error: {e}"}}
            await send_response(err_resp)
        except Exception as e:
            err_resp = {"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32603, "message": f"Internal error: {e}"}}
            await send_response(err_resp)


if __name__ == "__main__":
    asyncio.run(main())