# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Yi Su. All rights reserved.
#
"""Test llama server."""
import logging
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from llama_server.server import app
from llama_server.server import Conversation
from llama_server.server import Message


class MockModel:
    tokens = ["44th", " presi", "dent", " of", " USA"]

    def generate(self, *args, **kwargs):
        for tok in MockModel.tokens:
            yield tok


class Testapp(TestCase):
    """Test the server."""

    def setUp(self):
        self._client = TestClient(app)

    @patch("llama_server.server.model_id", "myModelId")
    def testGetModels(self):
        response = self._client.get("/v1/models")
        self.assertEqual(200, response.status_code)
        json = response.json()
        self.assertEqual(1, len(json["data"]))
        self.assertEqual("myModelId", json["data"][0]["id"])

    @patch("llama_server.server.logger", logging)
    @patch("llama_server.server.model_id", "myModelId")
    @patch("llama_server.server.model", MockModel())
    def testPostChat(self):
        conv = Conversation(
            model="myModelId",
            messages=[Message(role="user", content="who is barack obama?")],
            max_tokens=256,
            temperature=0.8,
            stream=False,
        )
        response = self._client.post("/v1/chat/completions", data=conv.json())
        json = response.json()
        self.assertEqual(1, len(json["choices"]))
        self.assertEqual("assistant", json["choices"][0]["message"]["role"])
        self.assertEqual(
            "".join(MockModel.tokens), json["choices"][0]["message"]["content"]
        )

    @patch("llama_server.server.logger", logging)
    @patch("llama_server.server.model_id", "myModelId")
    @patch("llama_server.server.model", MockModel())
    def testPostChatStreaming(self):
        conv = Conversation(
            model="myModelId",
            messages=[Message(role="user", content="who is barack obama?")],
            max_tokens=256,
            temperature=0.8,
            stream=True,
        )
        response = self._client.post("/v1/chat/completions", data=conv.json())
        from json import loads

        datalines = [line for line in response.iter_lines() if line.startswith("data")]
        for line, tok in zip(datalines, MockModel.tokens):
            json = loads(line[6:])
            self.assertEqual(1, len(json["choices"]))
            self.assertEqual("assistant", json["choices"][0]["delta"]["role"])
            self.assertEqual(tok, json["choices"][0]["delta"]["content"])
