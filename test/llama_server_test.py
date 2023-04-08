# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Yi Su. All rights reserved.
#
"""Test llama server."""
from unittest import TestCase

from llama_server.server import Buffer


class TestBuffer(TestCase):
    """Test Buffer class."""

    def setUp(self):
        self._buffer = Buffer(10, "user:")

    def testInit(self):
        self.assertTrue(isinstance(self._buffer, Buffer))

    def testAppendPopleft(self):
        self.assertEqual(0, len(self._buffer))
        self._buffer.append("0123456789")
        self.assertEqual(0, len(self._buffer))  # consume prompt
        self._buffer.append("test")
        self.assertEqual(4, len(self._buffer))
        self._buffer.append("abc")
        self.assertEqual(7, len(self._buffer))
        self.assertEqual("te", self._buffer.popleft())  # because len("user:") == 5
        self.assertEqual("", self._buffer.popleft())
        self._buffer.append("this is test")
        self.assertEqual("st", self._buffer.popleft())
        self.assertEqual("abc", self._buffer.popleft())

    def testPromptConsumed(self):
        self._buffer.append("abcdefgh")
        self.assertFalse(self._buffer.prompt_consumed())
        self._buffer.append("123")
        self.assertTrue(self._buffer.prompt_consumed())

    def testTurnends(self):
        self.assertFalse(self._buffer.turnends())
        self._buffer.append("0123456789")
        self._buffer.append("user")
        self.assertFalse(self._buffer.turnends())
        self._buffer.append(":")
        self.assertTrue(self._buffer.turnends())

    def testClear(self):
        self._buffer.append("0123456789")
        self.assertTrue(self._buffer.prompt_consumed())
        self._buffer.append("abc")
        self._buffer.append("xyz")
        self.assertEqual(6, len(self._buffer))
        self._buffer.clear()
        self.assertEqual(0, len(self._buffer))
        self.assertTrue(self._buffer.prompt_consumed())
