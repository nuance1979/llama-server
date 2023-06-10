# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Yi Su. All rights reserved.
#
"""Serve llama models with llama.cpp ."""
import logging
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional

import click
import uvicorn
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from pyllamacpp.model import Model
from sse_starlette import EventSourceResponse


PROMPT_PATH = Path(__file__).parent / "prompts" / "chat-with-bob.txt"
PROMPT = PROMPT_PATH.read_text(encoding="utf-8")
REVERSE_PROMPT = "\n User:"
REPLY_PREFIX = "\n Bob:"


class Message(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int
    temperature: float
    stream: bool


class Choice(BaseModel):
    message: Optional[Message] = None
    delta: Optional[Message] = None
    finish_reason: Optional[str] = None


class Completion(BaseModel):
    choices: List[Choice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "organization-owner"


class ModelList(BaseModel):
    data: List[ModelInfo]
    object: str = "list"


logger = None
model_id = None
model = None

app = FastAPI()


def _chat(user_utt: str, temperature: float) -> Generator[str, None, None]:
    return model.generate(
        user_utt, n_predict=256, repeat_penalty=1.0, n_threads=8, temp=temperature
    )


def chat_stream(
    user_utt: str, temperature: float
) -> Generator[Dict[str, Any], None, None]:
    for text in _chat(user_utt, temperature):
        logger.debug("text: %s", text)
        payload = Completion(
            choices=[Choice(delta=Message(role="assistant", content=text))]
        )
        yield {"event": "event", "data": payload.json()}
    payload = Completion(choices=[Choice(finish_reason="stop")])
    yield {"event": "event", "data": payload.json()}


def chat_nonstream(user_utt: str, temperature: float) -> Completion:
    assistant_utt = "".join(_chat(user_utt, temperature))
    logger.info("assistant: %s", assistant_utt)
    return Completion(
        choices=[Choice(message=Message(role="assistant", content=assistant_utt))]
    )


@app.post("/v1/chat/completions")
def chat(conv: Conversation):
    user_utt = conv.messages[-1].content
    temperature = conv.temperature
    logger.info("user: %s temperature: %f", user_utt, temperature)
    if not conv.stream:
        return chat_nonstream(user_utt, temperature)
    else:
        return EventSourceResponse(
            chat_stream(user_utt, temperature), ping_message_factory=None
        )


@app.get("/v1/models")
def models():
    return ModelList(data=[ModelInfo(id=model_id)])


class ModelPath(BaseModel):
    name: str
    path: str


class KnownModels(BaseModel):
    model_home: str
    models: Dict[str, ModelPath]


@click.command(context_settings={"show_default": True})
@click.option(
    "--models-yml",
    type=click.Path(exists=True),
    required=True,
    help="Path to the `models.yml` file.",
)
@click.option("--host", type=click.STRING, default="127.0.0.1", help="Server host.")
@click.option("--port", type=click.INT, default=8000, help="Server port.")
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Reload server automatically (for development).",
)
@click.option("--model-id", type=click.STRING, default="llama-7b", help="Model id.")
@click.option("--model-path", type=click.Path(exists=True), help="Model path.")
@click.option(
    "--log-level",
    type=click.Choice(["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
    help="Log level.",
)
def main(
    models_yml: Path,
    host: str,
    port: int,
    reload: bool,
    model_id: Optional[str] = None,
    model_path: Optional[Path] = None,
    log_level: Optional[str] = None,
):
    with open(models_yml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    KNOWN_MODELS = KnownModels.parse_obj(data)
    if model_id is None:
        model_id = os.environ.get("LLAMA_MODEL_ID", "llama-7b")
        assert model_id in KNOWN_MODELS.models, f"Unknown model id: {model_id}"
    if model_path is None:
        model_path = Path(KNOWN_MODELS.models.get(model_id).path)
        if not model_path.is_absolute():
            model_path = Path(KNOWN_MODELS.model_home) / model_path
    globals()["model_id"] = model_id
    globals()["model"] = Model(
        model_path=str(model_path),
        n_ctx=512,
        prompt_context=PROMPT,
        prompt_prefix=REVERSE_PROMPT,
        prompt_suffix=REPLY_PREFIX,
    )
    globals()["logger"] = logging.getLogger(name=__name__)
    globals()["logger"].setLevel(log_level)

    uvicorn.run("llama_server.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
