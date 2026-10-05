"""Shared helpers for the chatbot hardening tests. The LLM provider is ALWAYS faked."""
import uuid
from types import SimpleNamespace

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


class FakeStream:
    """Iterable of provider-style chunks that records whether it was closed."""

    def __init__(self, pieces, fail_after=None):
        self.pieces = list(pieces)
        self.fail_after = fail_after
        self.closed = False

    def __iter__(self):
        for index, piece in enumerate(self.pieces):
            if self.fail_after is not None and index >= self.fail_after:
                raise RuntimeError("provider exploded: secret-detail-123")
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])
        if self.fail_after is not None and self.fail_after >= len(self.pieces):
            raise RuntimeError("provider exploded: secret-detail-123")

    def close(self):
        self.closed = True


class FakeGroqClient:
    """Records every call; supports both normal and stream=True requests."""

    def __init__(self):
        self.calls = []
        self.reply = "FAKE"
        self.stream_pieces = ["Hel", "lo ", "there"]
        self.stream_fail_after = None
        self.error = None
        self.last_stream = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            self.last_stream = FakeStream(self.stream_pieces, self.stream_fail_after)
            return self.last_stream
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


def new_user_headers():
    email = f"hx_{uuid.uuid4().hex[:10]}@example.com"
    password = "StrongPass123!"
    r = client.post(
        "/auth/register",
        json={"email": email, "full_name": "X", "password": password, "confirm_password": password},
    )
    assert r.status_code == 200, r.text
    r = client.post("/auth/login", data={"username": email, "password": password})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def token():
    return "zq" + uuid.uuid4().hex[:10]


def parse_sse(text):
    """[(event, data_dict)] from a Server-Sent Events body."""
    import json

    events = []
    for block in text.strip().split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if name:
            events.append((name, data))
    return events
