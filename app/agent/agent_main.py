import re
import json
import time
import requests
from ..core.config import DEEPSEEK_API_KEY, MODEL_NAME, DEEPSEEK_URL,END_MODEL,END_URL,EMBEDDING_API_KEY
from ..core.memory import get_history, save_history, get_summary, save_summary
from .tools import get_weather, calculate, search_documents, agent_tools,list_all_documents


# ---------- 通用请求函数 (带重试) ----------
def safe_request_llm(data, max_retries=3, wait_time=10):
    for attempt in range(max_retries):
        try:
            resp = requests.post(END_URL, headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}"}, json=data,
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
    # =========================================================
    # 阶段 1：历史记忆压缩（只有超过 50条历史才触发）
    # =========================================================
    history = get_history(session_id)
    old_summary = get_summary(session_id)

    # 如果历史记录超过 50 条（25轮对话），触发压缩
    if len(history) > 50:
        print("触发历史摘要压缩...")
        # 取出最老的 8 条记录准备压缩
        to_compress = history[:8]

        # 1. 构造摘要提示词
        summary_prompt = f"请将以下对话历史提炼成一段简短的摘要，保留关键信息：\n\n{to_compress}\n\n摘要："

        # 2. 调用大模型（非流式）进行摘要
        summary_data = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": summary_prompt}],
            "temperature": 0.1
        }
        resp = requests.post(DEEPSEEK_URL, headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}, json=summary_data)

        if resp.status_code == 200:
            # 拿到新生成的摘要，和旧摘要拼在一起
            new_summary = resp.json()["choices"][0]["message"]["content"]
            final_summary = f"{old_summary}\n{new_summary}".strip()

            # 把摘要存进 Redis
            save_summary(session_id, final_summary)

            # 3. 代码截断：把前 8 条删掉，只保留最近 2 条
            history = history[8:]
            save_history(session_id, history)

    # =========================================================
    # 阶段 2：组装 messages（给大模型 注入核心人设）
    # =========================================================
    # 4.组装 messages
    messages = history.copy()  # 浅拷贝历史记录

    # 优化：将摘要和系统人设合并成一个 system 提示词
    system_content = "你是一个全能的智能助手，请根据用户问题自主决定是否调用工具。如果调用了 search_documents，请严格基于参考资料回答；" \
                     "若资料中无答案，直接说“根据当前文档无法回答”，严禁编造。"
    # 如果有历史摘要，追加到 system_content 后面
    current_summary = get_summary(session_id)
    if current_summary:
        system_content += f"\n\n【用户之前的对话历史摘要】：{current_summary}"

    # 只插入一次 system
    messages.insert(0, {"role": "system", "content": system_content})

    # 最后追加用户当前问题
    messages.append({"role": "user", "content": question})

    # =========================================================
    #  阶段 3：意图识别（问大模型 ：要不要调工具？）
    # =========================================================
    data = {"model": END_MODEL, "messages": messages, "tools": agent_tools, "temperature": 0.1}

    # 6. 判定意图
    response_data = safe_request_llm(data)
    if not response_data:
        yield "data: [SYS] 系统繁忙，请稍后再试。\n\n"
        return

    message = response_data["choices"][0]["message"]

    # 声明一个变量存来源
    extracted_sources_list = []
    # 7. 如果调用工具
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
                # 提取参考来源
                sources_match = re.search(r"\[SOURCES_START\](.*?)\[SOURCES_END\]", result, re.DOTALL)
                if sources_match:
                    # 获取纯文本来源
                    source_text = sources_match.group(1).strip()#group(1)只提取括号 () 里面的内容
                    #按行切分，去重，然后再加进列表
                    for line in source_text.split("\n"):
                        if line.strip() and line.strip() not in extracted_sources_list:
                            extracted_sources_list.append(line.strip())
                    # 把参考资料返回给大模型时，把参考来源删掉
                    result = result.replace(sources_match.group(0), "").strip()#group(0)：提取整个正则表达式匹配到的全部内容
            elif func_name == "list_all_documents":
                result = list_all_documents()
            else:
                result = "未知工具"

            messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": result})

        yield "data: [SYS] \n\n 正在生成最终回答...\n\n"
        data["messages"] = messages

    #最终请求不能再带 tools 字段
    data.pop("tools", None)

    # =========================================================
    # 阶段 4：流式生成最终回答
    # =========================================================
    full_answer = ""  # 用来记录最终完整的回答，存入记忆

    data["stream"] = True

    max_stream_retries = 3
    for attempt in range(max_stream_retries):
        try:
            resp = requests.post(END_URL, headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}"}, json=data,
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
            # 8. 流式输出完成后，输出来源，更新记忆
            #输出文档的参考来源,把列表拼接成字符串
            if extracted_sources_list:
                # 用双换行符连接，保证前端显示或替换 <br> 时是分行的
                extracted_sources = "\n\n".join(extracted_sources_list)
                source_text = "\n\n参考来源：\n\n" + extracted_sources
                safe_sources = source_text.replace("\n", "\ndata: ")
                yield f"data: {safe_sources}\n\n"

            # 注意：只存 user 和 assistant 的纯文本，不存复杂的 tool_calls，省 Token
            if full_answer:
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": full_answer})
                save_history(session_id, history)

            break  # 成功执行完流，退出重试
        except Exception as e:
            if attempt == max_stream_retries - 1:
                yield "data: [SYS] 网络中断，生成失败。\n\n"
