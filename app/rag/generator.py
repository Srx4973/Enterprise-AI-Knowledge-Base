import time
import json
import requests
from .retriever import build_context_and_prompt
from ..core.config import DEEPSEEK_API_KEY,MODEL_NAME,DEEPSEEK_URL
from ..core.memory import get_history, save_history


# -----------流式问答函数---------------
def ask_question_stream(question: str,session_id: str = "default"):
    # 1. 异常兜底
    try:
        prompt, source_info = build_context_and_prompt(question)
    except Exception as e:
        print(f"检索阶段异常: {e}")
        yield "data: 检索服务暂时不可用，请稍后再试。\n\n"
        return

    #  空结果兜底
    if not prompt:
        yield "data: 根据当前文档无法回答。\n\n"
        return

    # 2. 组装历史记忆
    history = get_history(session_id)
    messages = history.copy()
    # 把检索到的上下文和当前问题拼在一起，作为最新的一句话发给大模型
    messages.append({"role": "user", "content": prompt})

    # 3. 调用大模型 API
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 500,
        "stream": True
    }

    #流式重试循环（连上服务器被拒绝或连不上服务器）
    max_retries = 3
    resp = None  # 在循环外先定义 resp，确保跳出循环后能拿到

    for attempt in range(max_retries):
        try:
            # 发起网络请求，设置 timeout 防止死等
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=data, stream=True, timeout=60)

            # 如果不是 200 状态码，说明被限流或报错
            if resp.status_code != 200:
                if attempt < max_retries - 1:
                    print(f"流式请求失败 (状态码 {resp.status_code})，等待 20 秒重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(20)
                    continue  # 继续下一次循环
                else:
                    yield "data: 系统服务繁忙，请稍后再试。\n\n"
                    return
            #极其关键：如果请求成功建立，立刻跳出重试循环！
            break

        #抛异常情况（连不上服务器）
        except Exception as e:
            print(f"请求大模型异常: {e}")
            if attempt < max_retries - 1:
                print(f"网络异常，等待 20 秒重试 ({attempt + 1}/{max_retries})...")
                time.sleep(20)
            else:
                yield "data: 大模型服务网络异常，请稍后再试。\n\n"
                return

    # 4. 流式读取（新增一个变量记录完整回答）
    full_answer = ""

    for line in resp.iter_lines():
        if line:
            line_str = line.decode("utf-8")

            # 【优化点 1】用 startswith 代替 replace，防止误删正文里的 data:
            if not line_str.startswith("data: "):
                continue

            json_str = line_str[6:]  # 截取掉 "data: " 这 6 个字符
            if json_str.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(json_str)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    full_answer += delta  # 【优化点 2】累加回答，用于最后判断
                    safe = delta.replace("\n", "\ndata: ")#在每一行的开头，都插入了 data: 这个标记。
                    yield f"data: {safe}\n\n"   #这样无论模型输出多少个换行，前端按 \n\n 切分时，切出来的每一行都有合法的 data: 前缀，就不会被丢弃了。
            except json.JSONDecodeError:
                print(f"解析 JSON 失败: {json_str}")
                continue

    # 5. 【核心修改】判断大模型是否输出了“无法回答”
    # 如果包含这句话，直接 return，不输出参考来源
    if "根据当前文档无法回答" in full_answer:
        return

    # 6.保存记忆
    if full_answer:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": full_answer})
        save_history(session_id, history)

    # 7. 只有真正回答了，才拼接参考来源
    source_text = "\n\n参考来源：\n\n" + "\n\n".join(source_info)
    safe_sources = source_text.replace("\n", "\ndata: ")
    yield f"data: {safe_sources}\n\n"