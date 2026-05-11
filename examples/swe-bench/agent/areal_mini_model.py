from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel

from ._mini_path import ensure_mini_swe_agent_on_path

ensure_mini_swe_agent_on_path()

from minisweagent.models.utils.actions_toolcall import (  # noqa: E402
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)


class ArealMiniModelConfig(BaseModel):
    model_name: str = "areal"
    model_kwargs: dict[str, Any] = {}
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    format_error_template: str = "{{ error }}"


class ArealMiniModel:
    """mini-swe-agent Model adapter backed by AReaL's OpenAI-compatible client.

    The mini agent is synchronous. Run it in a worker thread, like the Terminal-Bench
    example does for blocking environment work. This adapter submits async AReaL
    client work back to the workflow event loop so loop-bound HTTP clients stay valid.
    """

    def __init__(
        self,
        client,
        *,
        event_loop: asyncio.AbstractEventLoop,
        config_class: type = ArealMiniModelConfig,
        **kwargs,
    ):
        self.client = client
        self.event_loop = event_loop
        self.config = config_class(**kwargs)

    def query(self, messages: list[dict], **kwargs) -> dict:
        request_kwargs = self.config.model_kwargs | kwargs
        future = asyncio.run_coroutine_threadsafe(
            self.client.chat.completions.create(
                model=self.config.model_name,
                messages=self._prepare_messages(messages),
                tools=[BASH_TOOL],
                **request_kwargs,
            ),
            self.event_loop,
        )
        response = future.result()
        raw_message = response.choices[0].message
        message = self._model_dump_json(raw_message)
        message["extra"] = {
            "actions": parse_toolcall_actions(
                raw_message.tool_calls or [],
                format_error_template=self.config.format_error_template,
            ),
            "response": self._model_dump_json(response),
            "cost": 0.0,
            "timestamp": time.time(),
        }
        return message

    def _model_dump_json(self, obj: Any) -> dict:
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        return [{k: v for k, v in message.items() if k != "extra"} for message in messages]

    def format_message(self, **kwargs) -> dict:
        return kwargs

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        return format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump() | kwargs

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }
