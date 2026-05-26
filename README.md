# hermes-agent-mcp

Hermes Agent MCP Supervision 增强 —— 健康检查循环、HTTP API、确定性路由。

> 本仓库是 [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) 的定制分支，存放 MCP Supervision 相关改动的独立验证环境。

## 改动内容

### 1. MCP 健康检查循环 (`tools/mcp_tool.py`)

`MCPServerTask` 新增完整健康检查机制：

| 新增字段/方法 | 说明 |
|---|---|
| `is_healthy(grace_seconds=30.0)` | 返回 True 当 server 已连接且最近有响应。判断 stdio 异常退出 + 超时 |
| `_health_check_loop()` | 后台协程，定期触发健康检查 |
| `_health_check_task` | asyncio Task 引用，支持 cancel |
| `_last_health_check_ts` | 上次成功检查时间戳 |
| `_stdio_exited_unexpectedly` | stdio 进程异常退出标志 |
| `_sampling` | sampling 配置字段 |

```python
# 调用示例
server: MCPServerTask = ...
if server.is_healthy(grace_seconds=30.0):
    print("server healthy")
```

**健康检查策略：**

- **HTTP 服务器**：定期调 `GET /+health`，超时 5s
- **Stdio 服务器**：调用 `session.list_tools()`，超时 `tool_timeout`
- **Grace period**：30s 内无检查记录视为 healthy（允许冷启动）
- **Reconnect**：stdio 异常退出后自动重启，错误计数 3 次内 exponential backoff

### 2. `/mcp/list` HTTP 端点 (`gateway/platforms/api_server.py`)

```
GET /mcp/list
```

返回所有已注册 MCP server 状态，JSON 格式，无需 LLM 调用，零 token 消耗：

```json
{
  "servers": [
    {
      "name": "deepseek",
      "transport": "stdio",
      "healthy": true,
      "last_check_ts": 1748246400.123,
      "tools": ["deepseek_chat", "deepseek_code", "..."],
      "error": null
    }
  ]
}
```

注册路由：`self._app.router.add_get("/mcp/list", self._handle_mcp_list)`（第 1746 行）

### 3. `/mcp list` Layer 1 消息平台路由 (`gateway/run.py`)

消息平台（Gateway）层新增命令路由，入口在 `run.py:2497`，handler 在 `run.py:4018`：

```python
async def _handle_mcp_list_command(self, event: MessageEvent) -> str:
```

与 HTTP 端点返回相同结构，用于 Telegram/Discord 等消息平台中 `/mcp list` 命令响应。

### 4. DeerFlow idna bug 修复 (`deerflow_smart_api.py`)

```diff
- from urllib.request import urlopen
+ import requests
  response = requests.get(url, timeout=10)
```

两处 `urllib.request` → `requests`，解决 `idna` 编码问题。

### 5. CodeWhale 禁用 (`config.yaml`)

```yaml
mcp_servers:
  codewhale:
    enabled: false
```

## 文件对照

| 文件 | 改动行 | 说明 |
|---|---|---|
| `tools/mcp_tool.py` | 新增 ~200 行 | 健康检查 + sampling |
| `gateway/platforms/api_server.py` | 507, 1746 | `/mcp/list` HTTP 路由 |
| `gateway/run.py` | 2497, 4018 | `/mcp list` 消息平台路由 |
| `deerflow_smart_api.py` | 2 处 | urllib → requests |
| `config.yaml` | 1 处 | codewhale enabled: false |

## 验证方法

```bash
# 1. 启动 hermes gateway
cd /home/xiaomi/hermes-agent-main
pip install -e .
hermes gateway

# 2. HTTP 验证
curl http://127.0.0.1:8642/mcp/list

# 3. 消息平台验证（Gateway 已启动时）
# 发送 /mcp list 到任一已配置平台（Telegram/Discord/...）

# 4. Python 验证
python3 -c "
import sys; sys.path.insert(0, '/home/xiaomi/hermes-agent-main')
from tools.mcp_tool import MCPServerTask
checks = ['is_healthy', '_health_check_loop', '_health_check_task', '_last_health_check_ts', '_sampling', '_stdio_exited_unexpectedly']
print({c: hasattr(MCPServerTask, c) for c in checks})
"
# {'is_healthy': True, '_health_check_loop': True, ...}
```

## 原仓库

- 主仓库：[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
- 本定制分支：[shenyang7565/hermes-agent-mcp](https://github.com/shenyang7565/hermes-agent-mcp)

## 上游同步

本仓库改动已在上游 `NousResearch/hermes-agent` 验证通过。如需同步到上游，提交 PR 到 `main` 分支。