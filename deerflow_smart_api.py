import sys
import os
import json
import time
import yaml
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

print('=' * 60)
print('DeerFlow智能模型切换API集成服务')
print('=' * 60)

# 加载 .env 文件（Windows 下 cmd 启动时无法继承 WSL 环境变量）
_dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(_dotenv_path):
    with open(_dotenv_path) as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ[k] = v  # 强制覆盖，shell变量可能残留

# 设置项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 导入MiniMind模型
try:
    from app.models.minimind_model import MiniMindChatModel
    print('✅ MiniMind模型导入成功')
except ImportError as e:
    print(f'❌ MiniMind模型导入失败: {e}')
    MiniMindChatModel = None

# MiniMax API client（deepseek-chat 槽位 → MiniMax-M2.7）
_minimax_api_key = os.environ.get("MINIMAX_CN_API_KEY", "")
_minimax_base_url = os.environ.get("MINIMAX_CN_BASE_URL", "https://api.minimaxi.com/v1")
_minimax_model = os.environ.get("MINIMAX_DEERFLOW_MODEL", "MiniMax-M2.7")

def _call_minimax(messages: List[Dict], model: str = None, **kwargs) -> Dict:
    import requests
    url = f"{_minimax_base_url}/chat/completions"
    payload = {
        "model": model or _minimax_model,
        "messages": messages,
        **{k: v for k, v in kwargs.items() if v is not None}
    }
    resp = requests.post(
        url,
        json=payload,
        headers={
            "Authorization": f"Bearer {_minimax_api_key}",
            "Content-Type": "application/json",
        },
        timeout=120
    )
    resp.raise_for_status()
    return resp.json()

# 创建FastAPI应用
app = FastAPI(
    title="DeerFlow智能模型切换API",
    description="智能模型切换系统API接口，支持MiniMind和DeepSeek双模型智能切换",
    version="2.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
minimind_model = None
config = None
model_stats = {
    'minimind-local': {'calls': 0, 'total_tokens': 0, 'success_rate': 1.0},
    'deepseek-chat': {'calls': 0, 'total_tokens': 0, 'success_rate': 1.0}
}

class SmartModelSelector:
    """智能模型选择器"""
    
    def __init__(self):
        # 任务类型到模型的映射
        self.task_rules = {
            'greeting': 'minimind-local',
            'simple_qa': 'minimind-local',
            'weather': 'minimind-local',
            'time': 'minimind-local',
            'introduction': 'minimind-local',
            'analysis': 'deepseek-chat',
            'research': 'deepseek-chat',
            'coding': 'deepseek-chat',
            'complex': 'deepseek-chat',
            'report': 'deepseek-chat'
        }
        
        # 关键词检测
        self.keyword_patterns = {
            'greeting': ['你好', 'hello', 'hi', '您好', '早上好', '晚上好'],
            'simple_qa': ['什么', '怎么', '为什么', '哪里', '谁', '多少'],
            'weather': ['天气', '气温', '温度', '下雨', '晴天'],
            'time': ['时间', '日期', '星期', '几点', '什么时候'],
            'introduction': ['介绍', '自我介绍', '你是谁', '你是什么', '什么是'],
            'analysis': ['分析', '研究', '评估', '趋势', '区别', '不同', '对比', '比较'],
            'research': ['研究', '调查', '探索', '发现', '论文', '解释', '详解', '原理'],
            'coding': ['代码', '编程', '函数', '算法', 'python', 'java', 'rust', 'go', '写代码'],
            'complex': ['复杂', '困难', '挑战', '难题', '高级'],
            'report': ['报告', '总结', '文档', '论文', '文章']
        }
    
    def analyze_task(self, task_description: str) -> Dict[str, Any]:
        """分析任务并选择模型"""
        task_lower = task_description.lower()
        
        # 检测任务类型
        detected_types = []
        for task_type, keywords in self.keyword_patterns.items():
            for keyword in keywords:
                if keyword in task_lower:
                    detected_types.append(task_type)
                    break
        
        # 确定主要任务类型
        primary_type = 'simple_qa'  # 默认
        if detected_types:
            # 优先选择复杂任务类型
            complex_types = ['analysis', 'research', 'coding', 'complex', 'report']
            for complex_type in complex_types:
                if complex_type in detected_types:
                    primary_type = complex_type
                    break
            else:
                primary_type = detected_types[0]
        
        # 选择模型
        selected_model = self.task_rules.get(primary_type, 'minimind-local')
        
        # 构建分析结果
        analysis = {
            'task_description': task_description,
            'detected_types': detected_types,
            'primary_type': primary_type,
            'selected_model': selected_model,
            'reason': f'检测到任务类型: {primary_type}，使用模型: {selected_model}',
            'confidence': min(1.0, len(detected_types) * 0.3)
        }
        
        return analysis

# 创建选择器实例
selector = SmartModelSelector()

# 初始化函数
@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    global minimind_model, config
    
    print('🚀 启动DeerFlow智能模型切换API服务...')
    
    # 加载配置
    config_path = os.path.join(project_root, '..', 'config.yaml')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f'✅ 配置文件加载成功，包含 {len(config.get("models", []))} 个模型')
        except Exception as e:
            print(f'❌ 配置文件加载失败: {e}')
            config = None
    
    # 初始化MiniMind模型
    if MiniMindChatModel:
        try:
            model_path = r'D:\AI\deepseek\minimind\MiniMind2'
            minimind_model = MiniMindChatModel(model_path=model_path)
            print('✅ MiniMind模型初始化成功')
        except Exception as e:
            print(f'❌ MiniMind模型初始化失败: {e}')
            minimind_model = None
    else:
        print('⚠️ MiniMind模型不可用，将使用模拟模式')
    
    print('✅ 服务启动完成')

# API端点定义
@app.get("/")
async def root():
    """根端点"""
    return {
        "service": "DeerFlow智能模型切换API",
        "version": "1.0.0",
        "status": "running",
        "available_models": ["minimind-local", "deepseek-chat"],
        "endpoints": {
            "/": "根端点",
            "/health": "健康检查",
            "/models": "获取模型列表",
            "/stats": "获取统计信息",
            "/analyze": "分析任务",
            "/chat": "聊天接口",
            "/research": "研究任务接口"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "deerflow-smart-model-api",
        "timestamp": time.time(),
        "minimind_available": minimind_model is not None,
        "config_loaded": config is not None
    }

@app.get("/models")
async def get_models():
    """获取可用模型列表"""
    if config and 'models' in config:
        models_info = []
        for model_config in config['models']:
            model_info = {
                'name': model_config.get('name'),
                'display_name': model_config.get('display_name'),
                'supports_thinking': model_config.get('supports_thinking', False),
                'supports_vision': model_config.get('supports_vision', False),
                'max_tokens': model_config.get('max_tokens')
            }
            models_info.append(model_info)
        
        return {
            "models": models_info,
            "count": len(models_info)
        }
    else:
        return {
            "models": [
                {
                    "name": "minimind-local",
                    "display_name": "MiniMind (Local)",
                    "supports_thinking": False,
                    "supports_vision": False,
                    "max_tokens": 2048
                },
                {
                    "name": "deepseek-chat",
                    "display_name": "MiniMax-M2.7 (via DeerFlow)",
                    "supports_thinking": True,
                    "supports_vision": False,
                    "max_tokens": 8192
                }
            ],
            "count": 2,
            "note": "使用默认模型配置"
        }

@app.get("/stats")
async def get_stats():
    """获取模型使用统计"""
    return {
        "model_stats": model_stats,
        "total_calls": sum(stats['calls'] for stats in model_stats.values()),
        "total_tokens": sum(stats['total_tokens'] for stats in model_stats.values()),
        "timestamp": time.time()
    }

@app.post("/analyze")
async def analyze_task(request: Request):
    """分析任务并推荐模型"""
    try:
        data = await request.json()
        task_description = data.get('task', '')
        
        if not task_description:
            raise HTTPException(status_code=400, detail="任务描述不能为空")
        
        # 分析任务
        analysis = selector.analyze_task(task_description)
        
        return {
            "success": True,
            "analysis": analysis,
            "recommendation": {
                "model": analysis['selected_model'],
                "reason": analysis['reason'],
                "confidence": analysis['confidence']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

@app.post("/chat")
async def chat_completion(request: Request):
    """聊天完成接口"""
    try:
        data = await request.json()
        messages = data.get('messages', [])
        model_override = data.get('model')  # 可选：手动指定模型
        
        if not messages:
            raise HTTPException(status_code=400, detail="消息列表不能为空")
        
        # 提取用户最后一条消息
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        if not user_messages:
            raise HTTPException(status_code=400, detail="需要用户消息")
        
        last_user_message = user_messages[-1]['content']
        
        # 分析任务并选择模型
        analysis = selector.analyze_task(last_user_message)
        selected_model = model_override or analysis['selected_model']
        
        # 调用模型
        start_time = time.time()
        
        if selected_model == 'minimind-local' and minimind_model:
            # 调用MiniMind模型
            result = minimind_model.generate(messages)
            
            if result.get('success'):
                response_content = result.get('content', '')
                usage = result.get('usage', {})
                
                # 更新统计
                model_stats['minimind-local']['calls'] += 1
                model_stats['minimind-local']['total_tokens'] += usage.get('total_tokens', 0)
                
                response = {
                    "model": "minimind-local",
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": response_content
                        }
                    }],
                    "usage": usage,
                    "processing_time": time.time() - start_time,
                    "analysis": analysis
                }
            else:
                raise HTTPException(status_code=500, detail=f"MiniMind生成失败: {result.get('error', '未知错误')}")
        
        elif selected_model == 'deepseek-chat':
            # MiniMax API（deepseek-chat 槽位）— 用 requests 替代 urllib.request 避免 Windows idna bug
            import requests
            url = f"{_minimax_base_url}/chat/completions"
            payload = {
                "model": _minimax_model,
                "messages": messages,
            }
            auth = f"Bearer {_minimax_api_key}"
            print(f"[DEBUG] URL: {url}")
            print(f"[DEBUG] AUTH: {auth[:15]}...")
            print(f"[DEBUG] KEY_LEN: {len(_minimax_api_key)}")
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers={"Authorization": auth, "Content-Type": "application/json"},
                    timeout=120
                )
                result = resp.json()
            except requests.RequestException as e:
                print(f"[DEBUG] RequestsError: {type(e).__name__}: {e}")
                raise HTTPException(status_code=502, detail=f"MiniMax API请求失败: {e}")
            response_content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})

            model_stats['deepseek-chat']['calls'] += 1
            model_stats['deepseek-chat']['total_tokens'] += usage.get("total_tokens", 0)

            response = {
                "model": result.get("model", "deepseek-chat"),
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    }
                }],
                "usage": usage,
                "processing_time": time.time() - start_time,
                "analysis": analysis,
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"不支持的模型: {selected_model}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聊天处理失败: {str(e)}")

@app.post("/research")
async def research_task(request: Request):
    """研究任务接口（通过Bridge服务）"""
    try:
        data = await request.json()
        query = data.get('query', '')
        
        if not query:
            raise HTTPException(status_code=400, detail="查询内容不能为空")
        
        # 分析任务
        analysis = selector.analyze_task(query)
        
        # 通过Bridge服务调用DeerFlow研究功能
        bridge_url = "http://localhost:8085/research"
        
        try:
            import requests
            bridge_response = requests.post(bridge_url, json={"query": query}, timeout=30)
            
            if bridge_response.status_code == 200:
                bridge_data = bridge_response.json()
                
                return {
                    "success": True,
                    "analysis": analysis,
                    "bridge_response": bridge_data,
                    "recommended_model": analysis['selected_model'],
                    "note": "研究任务通过Bridge服务处理"
                }
            else:
                return {
                    "success": False,
                    "analysis": analysis,
                    "error": f"Bridge服务返回错误: {bridge_response.status_code}",
                    "recommended_model": analysis['selected_model'],
                    "fallback_response": f"根据分析，这是一个{analysis['primary_type']}任务，建议使用{analysis['selected_model']}模型处理。"
                }
        except Exception as bridge_error:
            return {
                "success": False,
                "analysis": analysis,
                "error": f"Bridge服务调用失败: {str(bridge_error)}",
                "recommended_model": analysis['selected_model'],
                "fallback_response": f"根据分析，这是一个{analysis['primary_type']}任务，建议使用{analysis['selected_model']}模型处理。"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"研究任务处理失败: {str(e)}")

# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": f"服务器内部错误: {str(exc)}",
            "timestamp": time.time()
        }
    )

if __name__ == "__main__":
    print('\n启动API服务...')
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8090,
        reload=False
    )
