"""LLM 访问层：统一封装 Ollama，带可用性探测、JSON 解析与降级。

项目在无 LLM 时也必须跑通（规则引擎 + 模板测试），所以这里所有方法都可能返回 None，
调用方必须处理 None，而不是让异常冒到图里。
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from typing import Any

from ..config import SETTINGS


def _extract_json(text: str) -> Any | None:
    """从模型输出里抠出第一个完整 JSON（容忍 ```json 围栏与前后废话）。"""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class LLMClient:
    def __init__(self, enabled: bool = True, model: str = SETTINGS.llm_model) -> None:
        self.model = model
        self.calls = 0
        self.failures = 0
        self.last_error: str | None = None
        self.enabled = enabled and self._probe() if enabled else False
        self._client = None

    # ---------- 探测 ----------
    def _probe(self) -> bool:
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f"{SETTINGS.ollama_base_url.rstrip('/')}/api/tags", timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            names = [m.get("name", "") for m in data.get("models", [])]
            if self.model in names:
                return True
            base = self.model.split(":")[0]
            same = [n for n in names if n.split(":")[0] == base]
            if same:
                # 配置里写 :latest 但本地实际是 :7b 之类，自动对齐到已安装的 tag
                self.model = same[0]
                return True
            self.last_error = f"ollama 未安装模型 {self.model}（现有：{', '.join(names[:5])}）"
            return False
        except Exception as exc:
            self.last_error = f"ollama 不可用：{exc}"
            return False

    def _ensure_client(self):
        if self._client is None:
            from langchain_ollama import ChatOllama

            self._client = ChatOllama(
                model=self.model,
                base_url=SETTINGS.ollama_base_url,
                temperature=SETTINGS.llm_temperature,
                num_ctx=8192,
                client_kwargs={"trust_env": False, "timeout": SETTINGS.llm_timeout_s},
            )
        return self._client

    # ---------- 调用 ----------
    def chat(self, system: str, user: str, retries: int = 1) -> str | None:
        if not self.enabled:
            return None
        for attempt in range(retries + 1):
            try:
                t0 = time.time()
                msg = self._ensure_client().invoke([("system", system), ("human", user)])
                self.calls += 1
                content = getattr(msg, "content", "")
                if isinstance(content, list):  # 某些版本返回分块
                    content = "".join(str(c) for c in content)
                if content.strip():
                    return content
            except Exception as exc:
                self.failures += 1
                self.last_error = str(exc)[:300]
                time.sleep(0.4 * (attempt + 1))
        return None

    def chat_json(self, system: str, user: str, retries: int = 1) -> Any | None:
        out = self.chat(system + "\n只输出 JSON，不要输出任何解释文字。", user, retries=retries)
        return _extract_json(out) if out else None

    @property
    def status(self) -> str:
        if not self.enabled:
            return f"LLM 关闭（{self.last_error or '未启用'}）"
        return f"LLM={self.model}，调用 {self.calls} 次"
