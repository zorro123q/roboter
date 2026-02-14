import os
import sys
import glob  # <--- 之前可能漏了这个
import threading
import requests
import docx  # 确保安装了 python-docx
import uvicorn
import webview
import json
import math
import time
import re
import io
import contextlib
import traceback  # 用于打印详细报错
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ================= 配置区域 =================

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
    RESOURCE_DIR = sys._MEIPASS
    # 调试阶段建议注释掉日志重定向，这样能在控制台看到报错
    # sys.stdout = open(os.path.join(BASE_DIR, "app_log.txt"), "w", encoding="utf-8")
    # sys.stderr = sys.stdout
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCE_DIR = BASE_DIR

# ⚠️ 强制指定为 D 盘固定目录
DOC_FOLDER = r"D:\MyAI_Assistant\dist\docs_input"
DB_FILE = os.path.join(BASE_DIR, "knowledge_db.json")
HISTORY_FILE = os.path.join(BASE_DIR, "chat_history_agent.json")
HTML_FILE = os.path.join(RESOURCE_DIR, "index.html")

# 模型配置
CURRENT_MODEL = "qwen2.5:1.5b"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_API_URL = "http://127.0.0.1:11434"
PORT = 12345
MAX_LOOPS = 5


# ================= 辅助函数：安全读取 Word =================

def read_docx(file_path):
    """读取 Word 文档，带错误处理"""
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        return "\n".join(full_text)
    except Exception as e:
        print(f"❌ 读取文件失败 {file_path}: {e}")
        return ""


# ================= 工具：Python 代码执行器 =================

class PythonRunner:
    def __init__(self):
        self.globals = {"math": math, "os": os, "json": json, "time": time}

    def run(self, code):
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(code, self.globals)
            output = buffer.getvalue().strip()
            return output if output else "代码执行成功，但没有输出。"
        except Exception as e:
            return f"❌ 代码执行报错: {str(e)}"


python_runner = PythonRunner()


# ================= 🧠 历史记录管理器 =================

class HistoryManager:
    def __init__(self, filepath, retention_days=3):
        self.filepath = filepath
        self.retention_seconds = retention_days * 24 * 3600
        self.history = []
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                current_time = time.time()
                self.history = [msg for msg in data if current_time - msg.get('timestamp', 0) < self.retention_seconds]
            except:
                self.history = []

    def save(self):
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except:
            pass

    def add(self, role, content):
        self.history.append({"role": role, "content": content, "timestamp": time.time()})
        self.save()

    def get_recent(self, limit=10):
        return [{"role": m["role"], "content": m["content"]} for m in self.history[-limit:]]

    def clear(self):
        self.history = []
        if os.path.exists(self.filepath):
            os.remove(self.filepath)


history_mgr = HistoryManager(HISTORY_FILE)


# ================= 向量数据库 (RAG) =================

class SimpleVectorDB:
    def __init__(self):
        self.documents = []
        self.load()

    def add(self, text, vec, source):
        self.documents.append({'text': text, 'vec': vec, 'source': source})

    def search(self, query_vec, top_k=3):
        if not self.documents: return []
        scores = []
        q_norm = math.sqrt(sum(x * x for x in query_vec)) + 1e-9
        for doc in self.documents:
            d_vec = doc['vec']
            dot_product = sum(a * b for a, b in zip(query_vec, d_vec))
            d_norm = math.sqrt(sum(x * x for x in d_vec)) + 1e-9
            score = dot_product / (q_norm * d_norm)
            scores.append((score, doc['text']))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scores[:top_k]]

    def save(self):
        try:
            with open(DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False)
        except:
            pass

    def load(self):
        if os.path.exists(DB_FILE):
            try:
                with open(DB_FILE, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            except:
                pass

    def clear(self):
        self.documents = []


db = SimpleVectorDB()


# ================= 辅助函数：调用模型 =================

def call_ollama(messages):
    try:
        print(f"正在调用模型: {CURRENT_MODEL}...")
        res = requests.post(
            f"{OLLAMA_API_URL}/api/chat",
            json={"model": CURRENT_MODEL, "messages": messages, "stream": False},
            timeout=120
        )
        if res.status_code == 200:
            return res.json()['message']['content']
        else:
            error_msg = f"Ollama 报错 (状态码 {res.status_code}): {res.text}"
            print(error_msg)
            return error_msg
    except Exception as e:
        return f"请求异常: {str(e)}"


def get_embedding(text):
    try:
        res = requests.post(f"{OLLAMA_API_URL}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text})
        if res.status_code == 200: return res.json()["embedding"]
    except:
        pass
    return [0.0] * 768


# ================= Web 服务 =================

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


class ChatRequest(BaseModel): question: str


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(HTML_FILE, "r", encoding="utf-8") as f: return f.read()


@app.get("/get_history")
async def get_history_api(): return history_mgr.history


@app.post("/clear_history")
async def clear_history_api():
    history_mgr.clear()
    return {"status": "success"}


# === 核心修复：同步接口 ===
@app.get("/sync_docs_get")
async def sync_docs():
    # 🔥 增加全局异常捕获，防止 500 错误导致前端无法解析
    try:
        global db
        print(f"📂 正在从 {DOC_FOLDER} 读取文档...")

        if not os.path.exists(DOC_FOLDER):
            os.makedirs(DOC_FOLDER, exist_ok=True)

        # 这里必须要用到 glob，如果之前没导入就会报错
        docx_files = glob.glob(os.path.join(DOC_FOLDER, "*.docx"))

        file_status_list = []

        if not docx_files:
            return {
                "status": "warning",
                "message": f"文件夹为空: {DOC_FOLDER}",
                "files": []
            }

        db.clear()

        for f in docx_files:
            filename = os.path.basename(f)
            try:
                # 调用 read_docx
                txt = read_docx(f)

                if txt:
                    chunks = [txt[i:i + 500] for i in range(0, len(txt), 500)]
                    for chunk in chunks:
                        vec = get_embedding(chunk)
                        db.add(text=chunk, vec=vec, source=filename)

                    file_status_list.append({"name": filename, "status": "success", "chunks": len(chunks)})
                    print(f"✅ 读取成功: {filename}")
                else:
                    file_status_list.append({"name": filename, "status": "empty", "chunks": 0})
            except Exception as e:
                file_status_list.append({"name": filename, "status": "error", "chunks": 0})
                print(f"❌ 读取失败: {filename} -> {e}")

        db.save()

        return {
            "status": "success",
            "message": f"处理完成！共扫描 {len(docx_files)} 个文件。",
            "files": file_status_list
        }
    except Exception as e:
        # 🔥 打印详细报错堆栈到控制台
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"后端严重错误: {str(e)}",
            "files": []
        }


SYSTEM_PROMPT = """你是一个智能助手。
1. 如果用户的问题可以通过已有知识回答，直接回答。
2. ⚠️ 如果需要【计算】、【处理字符串】或【获取系统信息】，请务必编写 Python 代码。
3. 编写代码的格式：请将代码包裹在 ```python 和 ``` 之间。
4. 我会替你执行代码，并将“执行结果”告诉你。
5. 看到执行结果后，请根据结果回答用户。
"""


@app.post("/chat")
async def chat(request: ChatRequest):
    q_vec = get_embedding(request.question)
    docs = db.search(q_vec, top_k=3)
    rag_context = "\n---\n".join(docs) if docs else "无"

    current_messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n【参考文档】:\n{rag_context}"}
    ]
    current_messages.extend(history_mgr.get_recent(6))
    current_messages.append({"role": "user", "content": request.question})

    final_answer = ""
    steps_log = []

    loop_count = 0
    while loop_count < MAX_LOOPS:
        loop_count += 1
        response_text = call_ollama(current_messages)
        code_blocks = re.findall(r"```python(.*?)```", response_text, re.DOTALL)

        if not code_blocks:
            final_answer = response_text
            break

        code_to_run = code_blocks[0].strip()
        steps_log.append(f"🧠 思考: 需要运行代码...\n💻 代码: {code_to_run}")
        exec_result = python_runner.run(code_to_run)
        steps_log.append(f"⚙️ 系统执行结果: {exec_result}")

        current_messages.append({"role": "assistant", "content": response_text})
        current_messages.append(
            {"role": "system", "content": f"【系统反馈】代码执行结果:\n{exec_result}\n\n请根据这个结果回答用户。"})

    history_mgr.add("user", request.question)
    history_mgr.add("assistant", final_answer)

    return {
        "answer": final_answer,
        "steps": steps_log
    }


def start_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="critical")
    server = uvicorn.Server(config)
    server.run()


# ================= 🆕 新增：启动时检查数据库状态 =================
@app.get("/get_db_status")
async def get_db_status():
    # 统计当前内存里的数据库包含哪些文件
    unique_files = {}
    for doc in db.documents:
        src = doc.get('source', '未知文件')
        if src not in unique_files:
            unique_files[src] = 0
        unique_files[src] += 1

    # 格式化列表
    file_list = []
    for name, count in unique_files.items():
        file_list.append({"name": name, "status": "cached", "chunks": count})

    return {
        "total_chunks": len(db.documents),
        "files": file_list
    }


if __name__ == "__main__":
    # 确保文档目录存在
    if not os.path.exists(DOC_FOLDER):
        os.makedirs(DOC_FOLDER, exist_ok=True)

    print(f"🚀 Agent 后端服务已启动，监听端口: {PORT}")
    print(f"📂 文档目录: {DOC_FOLDER}")

    # --- 修改点开始 ---
    # 注释掉 webview 相关的代码
    # t = threading.Thread(target=start_server, daemon=True)
    # t.start()
    # window = webview.create_window(...)
    # webview.start(debug=True)

    # 直接启动 uvicorn 服务
    uvicorn.run(app, host="127.0.0.1", port=PORT)