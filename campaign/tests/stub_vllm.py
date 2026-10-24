"""A stub vLLM OpenAI-compatible server, good enough to prove the streaming and
metrics paths before renting anything.

It speaks the three endpoints VLLMSubstrate touches: /v1/models,
/v1/chat/completions with SSE, and /metrics. Tokens are emitted with a real
inter-token delay so TTFT and e2e are genuinely different numbers - which is the
whole point, since the bug being fixed is TTFT that silently equals e2e.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "vllm_metrics_v1.txt"


class Handler(BaseHTTPRequestHandler):
    served_model = "small"
    prefill_s = 0.05
    token_s = 0.004
    n_tokens = 20
    fail_next = False

    def log_message(self, *a):      # keep pytest output clean
        pass

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            body = json.dumps({"object": "list",
                               "data": [{"id": self.served_model, "object": "model"}]})
            self._send(200, body, "application/json")
        elif self.path.rstrip("/").endswith("/metrics"):
            self._send(200, FIXTURE.read_text(encoding="utf-8"), "text/plain")
        else:
            self._send(404, "not found")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if type(self).fail_next:
            type(self).fail_next = False
            self._send(500, json.dumps({"error": "simulated engine failure"}),
                       "application/json")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        want_lp = bool(req.get("logprobs"))
        time.sleep(self.prefill_s)                      # prefill: TTFT lives here
        n_tok = min(self.n_tokens, int(req.get("max_tokens", self.n_tokens)))
        for i in range(n_tok):
            if i:
                time.sleep(self.token_s)
            delta = {"content": ("Answer: B " if i == 0 else "tok%d " % i)}
            choice = {"index": 0, "delta": delta, "finish_reason": None}
            if want_lp:
                choice["logprobs"] = {"content": [{"token": "t", "logprob": -0.12}]}
            self._chunk({"id": "stub", "object": "chat.completion.chunk",
                         "choices": [choice]})
        self._chunk({"id": "stub", "choices": [{"index": 0, "delta": {},
                                                "finish_reason": "stop"}]})
        self._chunk({"id": "stub", "choices": [],
                     "usage": {"prompt_tokens": 123, "completion_tokens": n_tok,
                               "total_tokens": 123 + n_tok,
                               "prompt_tokens_details": {"cached_tokens": 40}}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _chunk(self, obj):
        self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
        self.wfile.flush()

    def _send(self, code, body, ctype="text/plain"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class StubServer:
    def __init__(self, served_model: str = "small") -> None:
        self.handler = type("H", (Handler,), {"served_model": served_model})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self.handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        return "http://127.0.0.1:%d/v1" % self.port

    def __enter__(self) -> "StubServer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
