import json
import time
import requests
from ..core.memory import get_history, save_history
from ..core.config import DEEPSEEK_API_KEY, MODEL_NAME, DEEPSEEK_URL
from .tools import get_weather, calculate, search_documents, agent_tools,list_all_documents


# ---------- 通用请求函数 (带重试) ----------
def safe_request_llm(data, max_retries=3, wait_time=10):
    for attempt in range(max_retries):
        try:
            resp = requests.post(DEEPSEEK_URL, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}, json=data,
                                 timeout=60)
            response_data = resp.json()
            if "error" in response_data:
                print(f" 触发异常：{response_data['error']['message']}")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    return None
            return response_data
        except Exception as e:
            print(f" 网络请求异常：{e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                return None
    return None


# ---------- Agent 核心逻辑 ----------
def ask_agent_stream(question: str,session_id: str = "default"):
    # 1. 获取历史记忆
    history = get_history(session_id)

    # 2. 组装本次请求的 messages
    messages = history.copy()  # 浅拷贝历史记录
    messages.append({"role": "user", "content": question})
    data = {"model": MODEL_NAME, "messages": messages, "tools": agent_tools, "temperature": 0.1}

    # 3. 判定意图
    response_data = safe_request_llm(data)
    if not response_data:
        yield "data: [SYS] 系统繁忙，请稍后再试。\n\n"
        return

    message = response_data["choices"][0]["message"]

    # 4. 如果调用工具
    if "tool_calls" in message:
        yield "data: [SYS] 正在调用工具...\n\n"
        messages.append(message)

        for tool_call in message["tool_calls"]:
            func_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            yield f"data: [SYS] [执行 {func_name} : {args}]\n\n"

            if func_name == "get_weather":
                result = get_weather(args["city"])
            elif func_name == "calculate":
                result = calculate(args["expression"])
            elif func_name == "search_documents":
                result = search_documents(args["query"])
            elif func_name == "list_all_documents":
                result = list_all_documents()
            else:
                result = "未知工具"

            messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})

        yield "data: [SYS] \n\n 正在生成最终回答...\n\n"
        data["messages"] = messages

    #最终请求不能再带 tools 字段
    data.pop("tools", None)

    # 5. 流式生成最终回答
    full_answer = ""  # 用来记录最终完整的回答，存入记忆

    data["stream"] = True
    max_stream_retries = 3
    for attempt in range(max_stream_retries):
        try:
            resp = requests.post(DEEPSEEK_URL, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}, json=data,
                                 stream=True)
            if resp.status_code != 200:
                yield "data: [SYS] 生成失败，正在重试...\n\n"
                time.sleep(20)
                continue

            for line in resp.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if not line_str.startswith("data: "): continue
                    json_str = line_str[6:]
                    if json_str.strip() == "[DONE]": break
                    try:
                        chunk = json.loads(json_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            full_answer += delta
                            safe = delta.replace("\n", "\ndata: ")
                            yield f"data: {safe}\n\n"
                    except:
                        continue
            # 6. 流式输出完成后，更新记忆
            # 注意：只存 user 和 assistant 的纯文本，不存复杂的 tool_calls，省 Token
            if full_answer:
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": full_answer})
                save_history(session_id, history)

            break  # 成功执行完流，退出重试
        except Exception as e:
            if attempt == max_stream_retries - 1:
                yield "data: [SYS] 网络中断，生成失败。\n\n"
