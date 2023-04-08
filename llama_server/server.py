# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Yi Su. All rights reserved.
#
"""Serve llama models with llama.cpp ."""
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from multiprocessing import Process
from multiprocessing import Queue
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
PROMPT = PROMPT_PATH.read_text().strip()
PROMPT_SIZE = len(PROMPT)
REVERSE_PROMPT = "User:"
REPLY_PREFIX = "Bob: "


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


class Completion(BaseModel):
    choices: List[Choice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "organization-owner"


class ModelList(BaseModel):
    data: List[ModelInfo]
    object: str = "list"


class Buffer:
    def __init__(self, prompt_size: int, reverse_prompt: str) -> None:
        self._q = deque(maxlen=10)
        self._c = 0
        self._prompt_size = prompt_size
        self._reverse_prompt = reverse_prompt
        self._is_first = True
        self._c_first = 0

    def __len__(self) -> int:
        return self._c

    def prompt_consumed(self) -> bool:
        return self._c_first >= self._prompt_size

    def clear(self) -> None:
        self._q.clear()
        self._c = 0

    def append(self, data: str) -> None:
        if self._is_first:
            self._c_first += len(data)
            if self._c_first < self._prompt_size:
                return
            else:
                self._is_first = False
                diff = self._c_first - self._prompt_size
                if diff > 0:
                    self.append(data[-diff:])
                self._c_first = self._prompt_size
        else:
            self._c += len(data)
            self._q.append(data)

    def popleft(self) -> str:
        if self._c < len(self._reverse_prompt):
            return ""
        data = self._q.popleft()
        self._c -= len(data)
        if self._c < len(self._reverse_prompt):
            diff = self._c - len(self._reverse_prompt)
            self._q.appendleft(data[diff:])
            self._c = len(self._reverse_prompt)
            data = data[:diff]
        return data

    def turnends(self) -> bool:
        return "".join(self._q).endswith(self._reverse_prompt)


logging.basicConfig(
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", level=logging.INFO
)
logger = logging.getLogger(name=__name__)

model_id = None
model_path = None
output_q = Queue()
input_q = Queue()
buffer = Buffer(PROMPT_SIZE, REVERSE_PROMPT)


def generate(model_path: str, input_q: Queue, output_q: Queue) -> None:
    def output_callback(text: str) -> None:
        output_q.put(text)

    def input_callback() -> str:
        return input_q.get()

    model = Model(ggml_model=model_path, n_ctx=512)
    model.generate(
        PROMPT,
        new_text_callback=output_callback,
        grab_text_callback=input_callback,
        n_predict=256,
        n_batch=1024,
        n_keep=48,
        repeat_penalty=1.0,
        n_threads=8,
        interactive=True,
        antiprompt=[REVERSE_PROMPT],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    p = Process(target=generate, args=(model_path, input_q, output_q))
    p.start()
    # Skip system prompt
    while not buffer.prompt_consumed():
        buffer.append(output_q.get())
    logger.info("ready to serve...")
    yield
    input_q.close()
    output_q.close()
    if p.is_alive():
        p.terminate()
        time.sleep(5)
        p.kill()


app = FastAPI(lifespan=lifespan)


def chat_stream(user_utt: str) -> Generator[Dict[str, Any], None, None]:
    for text in _chat(user_utt):
        payload = Completion(
            choices=[Choice(delta=Message(role="assistant", content=text))]
        )
        yield {"event": "event", "data": payload.json()}
    yield {"event": "event", "data": "[DONE]"}


def _chat(user_utt: str) -> Generator[str, None, None]:
    input_q.put(user_utt)
    counter = 0
    while not buffer.turnends():
        text = output_q.get()
        counter += len(text)
        if counter <= len(REPLY_PREFIX):
            continue
        buffer.append(text)
        yield buffer.popleft()
    while True:
        text = buffer.popleft()
        if not text:
            break
        yield text
    buffer.clear()


def chat_nonstream(user_utt: str) -> Completion:
    assistant_utt = "".join(_chat(user_utt))
    logger.info("assistant: %s", assistant_utt)
    return Completion(
        choices=[Choice(message=Message(role="assistant", content=assistant_utt))]
    )


@app.post("/v1/chat/completions")
def chat(conv: Conversation):
    user_utt = conv.messages[-1].content
    logger.info("user: %s", user_utt)
    if not conv.stream:
        return chat_nonstream(user_utt)
    else:
        return EventSourceResponse(chat_stream(user_utt))


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
def main(
    models_yml: Path,
    host: str,
    port: int,
    reload: bool,
    model_id: Optional[str] = None,
    model_path: Optional[Path] = None,
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
    globals()["model_path"] = str(model_path)

    uvicorn.run("llama_server.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
