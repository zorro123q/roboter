# encoding:utf-8

import requests
import plugins
from plugins import *
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger


# 注册插件
@plugins.register(
    name="MyRAGBot",
    desire_priority=999,  # 优先级最高，拦截所有消息
    hidden=False,
    desc="连接本地 RAG Agent 后端",
    version="1.0",
    author="User"
)
class MyRAGBot(Plugin):
    def __init__(self):
        super().__init__()
        # 监听处理上下文事件
        self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
        # 你的 main.py 服务地址
        self.api_url = "http://127.0.0.1:12345/chat"
        self.sync_url = "http://127.0.0.1:12345/sync_docs_get"
        logger.info("[MyRAGBot] 插件已初始化")

    def on_handle_context(self, e_context: EventContext):
        context = e_context['context']

        # 1. 只处理文本消息
        if context.type != ContextType.TEXT:
            return

        content = context.content.strip()
        logger.debug(f"[MyRAGBot] 收到消息: {content}")

        # 2. 特殊指令：同步文档
        if content == "#同步文档":
            self.handle_sync(e_context)
            return

        # 3. 转发给 Agent 后端
        try:
            # 构造请求数据
            payload = {"question": content}

            # 发送请求 (设置超时，防止微信这边卡死)
            response = requests.post(self.api_url, json=payload, timeout=60)

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "后端未返回答案")
                steps = data.get("steps", [])

                # 格式化输出：如果 Agent 有思考步骤，可以选择是否展示
                # 这里我们简单拼接一下，让微信里能看到它执行了代码
                final_reply = answer
                if steps:
                    # 提取最后一步执行结果简单展示，避免刷屏
                    final_reply += "\n\n(已调用 Python 执行计算)"

                reply = Reply(ReplyType.TEXT, final_reply)
                e_context['reply'] = reply
                e_context.action = EventAction.BREAK_PASS  # 拦截消息，不再给 GPT 处理

            else:
                # 后端报错
                error_msg = f"Agent 后端报错: {response.status_code}"
                reply = Reply(ReplyType.ERROR, error_msg)
                e_context['reply'] = reply
                e_context.action = EventAction.BREAK_PASS

        except Exception as e:
            logger.error(f"[MyRAGBot] 连接异常: {e}")
            # 如果连接失败，可以选择 pass 让默认的 GPT 接管，或者报错
            # 这里我们选择报错提示用户
            reply = Reply(ReplyType.TEXT, f"⚠️ 无法连接到 Agent 后端。\n请检查 main.py 是否已启动。\n错误: {e}")
            e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS

    def handle_sync(self, e_context):
        """处理同步文档指令"""
        try:
            requests.get(self.sync_url, timeout=30)
            reply = Reply(ReplyType.TEXT, "📚 文档同步/索引构建完成！")
            e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS
        except Exception as e:
            reply = Reply(ReplyType.ERROR, f"同步失败: {e}")
            e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS