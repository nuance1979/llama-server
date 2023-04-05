# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Yi Su. All rights reserved.
#
"""Serve llama models with llama.cpp ."""
import logging
import os
import selectors
from collections import deque
from pathlib import Path
from subprocess import PIPE
from subprocess import Popen
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

LLAMA_CPP_HOME = Path(os.environ.get("LLAMA_CPP_HOME", "../llama.cpp"))
LLAMA_CPP_BIN = LLAMA_CPP_HOME / "main"
LLAMA_CPP_MODELS = LLAMA_CPP_HOME / "models"
PROMPTS = LLAMA_CPP_HOME / "prompts" / "chat-with-bob.txt"
MODEL_FILE_NAME = "ggml-model-q4_0.bin"
KNOWN_MODELS = {
    "llama-7b": {
        "name": "LLAMA-7B",
        "path": str(LLAMA_CPP_MODELS / "7B" / MODEL_FILE_NAME),
    },
    "llama-13b": {
        "name": "LLAMA-13B",
        "path": str(LLAMA_CPP_MODELS / "13B" / MODEL_FILE_NAME),
    },
    "llama-33b": {
        "name": "LLAMA-33B",
        "path": str(LLAMA_CPP_MODELS / "33B" / MODEL_FILE_NAME),
    },
    "llama-65b": {
        "name": "LLAMA-65B",
        "path": str(LLAMA_CPP_MODELS / "65B" / MODEL_FILE_NAME),
    },
}


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


logging.basicConfig(
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", level=logging.INFO
)
logger = logging.getLogger(name=__name__)

app = FastAPI()


model_id = os.environ.get("LLAMA_MODEL_ID", "llama-7b")
assert model_id in KNOWN_MODELS, f"Unknown model id: {model_id}"
cmds = [
    str(LLAMA_CPP_BIN),
    "-m",
    KNOWN_MODELS[model_id]["path"],
    "-c",
    "512",
    "-b",
    "1024",
    "-n",
    "256",
    "--keep",
    "48",
    "--repeat_penalty",
    "1.0",
    "-i",
    "-r",
    "User:",
    "-f",
    str(PROMPTS),
]
pipe = Popen(cmds, stdin=PIPE, stdout=PIPE)
buffer = deque([], maxlen=20)


# Adapted from example code returned by ChatGPT
def chat_stream(user_utt: str) -> Generator[Dict[str, Any], None, None]:
    selector = selectors.DefaultSelector()
    selector.register(pipe.stdout, selectors.EVENT_READ)
    pipe.stdin.write(f"{user_utt}\n".encode())
    pipe.stdin.flush()

    ok = False
    done = False
    while not done:
        for key, _ in selector.select(0.1):
            char = key.fileobj.read(1).decode()
            buffer.append(char)
            if len(buffer) < 5:
                continue
            if ok:
                # Check reverse prompt
                if "".join(buffer).startswith("User:"):
                    done = True
                    break
                payload = Completion(
                    choices=[
                        Choice(
                            delta=Message(role="assistant", content=buffer.popleft())
                        )
                    ]
                )
                yield {"event": "event", "data": payload.json()}
            elif "".join(buffer).endswith("User:Bob: "):  # Skip system prompt
                buffer.clear()
                ok = True
    yield {"event": "event", "data": "[DONE]"}


def chat_nonstream(user_utt: str) -> Completion:
    pipe.stdin.write(f"{user_utt}\n".encode())
    pipe.stdin.flush()
    # HACK: assumes the assistant response is the last line returned
    #   from llama.cpp binary and that line has prefix `User:Bob: `.
    line = pipe.stdout.readline().decode()
    while line:
        logger.debug("line: %s", line.strip())
        if line.startswith("User:Bob:"):
            break
        line = pipe.stdout.readline().decode()
    assistant_utt = line.split(": ")[-1]
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
