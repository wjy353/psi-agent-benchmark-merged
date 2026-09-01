import json
import os
import socket
import subprocess
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Optional, TextIO

import requests
from pydantic import BaseModel, Field

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
)
from tau2.environment.tool import Tool


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WORKSPACE = DEFAULT_REPO_ROOT / "examples" / "tau2-workspace"
DEFAULT_CACHE_ROOT = DEFAULT_REPO_ROOT / ".cache"


class PsiAgentState(BaseModel):
    """State kept by the tau2 adapter while psi-agent owns model history."""

    session_url: str
    session_id: str
    bootstrapped: bool = False
    messages: list[APICompatibleMessage] = Field(default_factory=list)


class PsiAgentTau2Adapter(HalfDuplexAgent[PsiAgentState]):
    """Bridge tau2 half-duplex simulations to a psi-agent Session process."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        task=None,
        **_: object,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm or "psi-agent"
        self.llm_args = dict(llm_args or {})
        self.task = task
        self.workspace = Path(
            self.llm_args.get("psi_workspace")
            or os.environ.get("PSI_TAU2_WORKSPACE")
            or DEFAULT_WORKSPACE
        )
        self.repo_root = Path(
            self.llm_args.get("psi_repo_root")
            or os.environ.get("PSI_AGENT_REPO_ROOT")
            or DEFAULT_REPO_ROOT
        )
        self.cache_root = Path(
            self.llm_args.get("cache_root")
            or os.environ.get("PSI_TAU2_CACHE_ROOT")
            or DEFAULT_CACHE_ROOT
        )
        self.ai_socket = str(
            self.llm_args.get("psi_ai_socket")
            or os.environ.get("PSI_TAU2_AI_SOCKET")
            or "http://127.0.0.1:18080"
        )
        self.uv_executable = str(
            self.llm_args.get("psi_uv") or os.environ.get("PSI_TAU2_UV") or "uv"
        )
        self.session_url = self.llm_args.get("psi_session_url") or os.environ.get(
            "PSI_TAU2_SESSION_URL"
        )
        self.request_timeout = float(
            self.llm_args.get("psi_request_timeout")
            or os.environ.get("PSI_TAU2_REQUEST_TIMEOUT", "300")
        )
        self.start_timeout = float(
            self.llm_args.get("psi_start_timeout")
            or os.environ.get("PSI_TAU2_START_TIMEOUT", "30")
        )
        self._session_process: subprocess.Popen | None = None
        self._session_log_file: TextIO | None = None

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> PsiAgentState:
        session_id = self._session_id()
        session_url = self.session_url or self._start_session(session_id)
        return PsiAgentState(
            session_url=session_url,
            session_id=session_id,
            messages=list(message_history or []),
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: PsiAgentState
    ) -> tuple[AssistantMessage, PsiAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif message is not None:
            state.messages.append(message)

        payload = self._format_turn(message, state)
        raw_response = self._post_to_session(state.session_url, payload)
        assistant = self._parse_response(raw_response)
        assistant.raw_data = {
            "psi_agent_raw_response": raw_response,
            "psi_agent_session_url": state.session_url,
            "psi_agent_session_id": state.session_id,
        }
        state.messages.append(assistant)
        state.bootstrapped = True
        return assistant, state

    def stop(
        self,
        message: Optional[ValidAgentInputMessage] = None,
        state: Optional[PsiAgentState] = None,
    ) -> None:
        if self._session_process is None:
            return
        self._session_process.terminate()
        try:
            self._session_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._session_process.kill()
            self._session_process.wait(timeout=10)
        finally:
            self._session_process = None
            if self._session_log_file is not None:
                self._session_log_file.close()
                self._session_log_file = None

    def is_stop(self, message: AssistantMessage) -> bool:
        content = (message.content or "").strip()
        return content == "###STOP###"

    def _session_id(self) -> str:
        task_id = getattr(self.task, "id", "task") or "task"
        safe_task = "".join(c if c.isalnum() or c in "_-" else "_" for c in task_id)
        return f"tau2-{safe_task}-{uuid.uuid4().hex[:8]}"

    def _start_session(self, session_id: str) -> str:
        port = self._free_port()
        session_url = f"http://127.0.0.1:{port}"
        appdata = self.cache_root / "psi-appdata"
        command = [
            self.uv_executable,
            "run",
            "--project",
            str(self.repo_root),
            "psi-agent",
            "session",
            "--workspace",
            str(self.workspace),
            "--agent",
            str(self.workspace),
            "--appdata",
            str(appdata),
            "--session-id",
            session_id,
            "--ai-socket",
            self.ai_socket,
            "--channel-socket",
            session_url,
        ]
        env = self._cache_env()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        log_dir = self.cache_root / "tau2-psi-agent" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._session_log_file = (log_dir / f"{session_id}.log").open(
            "w", encoding="utf-8"
        )
        self._session_process = subprocess.Popen(
            command,
            cwd=str(self.repo_root),
            env=env,
            stdout=self._session_log_file,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + self.start_timeout
        while time.monotonic() < deadline:
            if self._session_process.poll() is not None:
                raise RuntimeError("psi-agent session exited before it was ready")
            if self._session_ready(session_url):
                return session_url
            time.sleep(0.25)
        self.stop()
        raise TimeoutError(
            f"psi-agent session did not start within {self.start_timeout}s"
        )

    def _cache_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("PSI_APPDATA", str(self.cache_root / "psi-appdata"))
        env.setdefault("UV_CACHE_DIR", str(self.cache_root / "uv"))
        env.setdefault("PIP_CACHE_DIR", str(self.cache_root / "pip"))
        env.setdefault("XDG_CACHE_HOME", str(self.cache_root / "xdg"))
        env.setdefault("PYTHONPYCACHEPREFIX", str(self.cache_root / "pycache"))
        env.setdefault("HF_HOME", str(self.cache_root / "huggingface"))
        env.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.1.0")
        env.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PSI_AGENT", "0.1.0")
        return env

    def _session_ready(self, session_url: str) -> bool:
        parsed = urllib.parse.urlparse(session_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port
        if port is None:
            return False
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            return False

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _format_turn(
        self, message: Optional[ValidAgentInputMessage], state: PsiAgentState
    ) -> str:
        parts = []
        if not state.bootstrapped:
            parts.append(self._bootstrap_context())
        parts.append("<tau2_turn>")
        parts.append(self._message_to_text(message))
        parts.append("</tau2_turn>")
        return "\n\n".join(parts)

    def _bootstrap_context(self) -> str:
        task_id = getattr(self.task, "id", None)
        task_text = ""
        if self.task is not None:
            try:
                task_text = self.task.model_dump_json(indent=2)
            except Exception:
                task_text = str(self.task)
        tools = [tool.openai_schema for tool in self.tools]
        return "\n".join(
            [
                "<tau2_context>",
                f"task_id: {task_id}",
                "<domain_policy>",
                self.domain_policy,
                "</domain_policy>",
                "<available_tools_json>",
                json.dumps(tools, ensure_ascii=False, indent=2),
                "</available_tools_json>",
                "<task_json>",
                task_text,
                "</task_json>",
                "</tau2_context>",
            ]
        )

    @staticmethod
    def _message_to_text(message: Optional[ValidAgentInputMessage]) -> str:
        if message is None:
            return "No incoming message."
        if isinstance(message, MultiToolMessage):
            return "\n".join(
                PsiAgentTau2Adapter._message_to_text(m) for m in message.tool_messages
            )
        if isinstance(message, ToolMessage):
            return "\n".join(
                [
                    "<tau2_tool_result>",
                    f"tool_call_id: {message.id}",
                    f"requestor: {message.requestor}",
                    str(message.content or ""),
                    "</tau2_tool_result>",
                ]
            )
        role = getattr(message, "role", "user")
        content = getattr(message, "content", "") or ""
        return f"{role}: {content}"

    def _post_to_session(self, session_url: str, content: str) -> str:
        response = requests.post(
            f"{session_url}/chat/completions",
            json={"messages": [{"role": "user", "content": content}], "stream": True},
            timeout=self.request_timeout,
            stream=True,
        )
        response.raise_for_status()
        chunks = []
        # SSE events can carry multi-line data (JSON containing literal newlines
        # arrives split across several "data:" lines). Accumulate until a blank
        # line (event boundary), then parse the joined payload. Tolerate
        # malformed events instead of crashing the whole simulation.
        data_lines: list[str] = []

        def _flush(data_lines: list[str]) -> None:
            if not data_lines:
                return
            data = "\n".join(data_lines)
            if data == "[DONE]":
                return
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                return
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    chunks.append(delta["content"])

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                _flush(data_lines)
                data_lines = []
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        _flush(data_lines)
        return "".join(chunks).strip()

    def _parse_response(self, raw_response: str) -> AssistantMessage:
        data = self._extract_json(raw_response)
        if data is None:
            return AssistantMessage(role="assistant", content=raw_response or " ")
        if isinstance(data.get("message"), str) and data["message"].strip():
            return AssistantMessage(role="assistant", content=data["message"].strip())
        raw_calls = data.get("tool_calls")
        if isinstance(raw_calls, list) and raw_calls:
            calls = []
            for raw_call in raw_calls:
                parsed = self._parse_tool_call(raw_call)
                if parsed is not None:
                    calls.append(parsed)
            if calls:
                return AssistantMessage(role="assistant", content=None, tool_calls=calls)
        fallback = json.dumps(data, ensure_ascii=False)
        return AssistantMessage(role="assistant", content=fallback)

    @staticmethod
    def _extract_json(raw_response: str) -> dict | None:
        text = raw_response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _parse_tool_call(raw_call: object) -> ToolCall | None:
        if not isinstance(raw_call, dict):
            return None
        name = raw_call.get("name")
        arguments = raw_call.get("arguments")
        if not name and isinstance(raw_call.get("function"), dict):
            function = raw_call["function"]
            name = function.get("name")
            arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None
        return ToolCall(
            id=str(raw_call.get("id") or f"call_{uuid.uuid4().hex[:8]}"),
            name=name,
            arguments=arguments,
            requestor="assistant",
        )


def create_psi_agent(tools, domain_policy, **kwargs):
    """Factory registered as --agent psi_agent."""

    return PsiAgentTau2Adapter(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        task=kwargs.get("task"),
    )
