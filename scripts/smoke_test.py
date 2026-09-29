"""Smoke test for Omni Slated Mem Core — exercises the platform contract.

  1. Boots the service (uvicorn subprocess) and waits for /health.
  2. Concurrent adds: 64 worker threads (mirrors the platform's 64 Add
     workers), each POSTing /v1/memories/add for a rotating set of users.
  3. Durability: every add returns 200 only after searchable — verified by
     searching for planted facts immediately after the adds complete.
  4. Contract shapes: add/search response schemas, top_k respected,
     most-relevant-first ordering.
  5. user_id isolation: users never see each other's memories.
  6. Relevance gating: clearly irrelevant queries return [].

Usage:  python scripts/smoke_test.py [--port 18000]
Exit code 0 on success, 1 on any failure.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import time
import urllib.request

ADD = "/v1/memories/add"
SEARCH = "/v1/memories/search"

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name + (f": {detail}" if detail else ""))


def post(base: str, path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = {}
        return e.code, body


def get(base: str, path: str) -> tuple[int, dict]:
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return r.status, json.loads(r.read().decode())


def wait_healthy(base: str, timeout: int = 180) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            code, body = get(base, "/health")
            if code == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


# Planted facts: (user, session, role, content)
FACTS = [
    ("alice", "s1", "user", "My dog's name is Biscuit and he loves peanut butter."),
    ("alice", "s1", "assistant", "Noted: Biscuit the dog, peanut butter enthusiast."),
    ("alice", "s2", "user", "I was born in 1988 in Lisbon, Portugal."),
    ("alice", "s2", "user", "My dentist appointment is on the third Thursday of next month."),
    ("bob", "s1", "user", "The launch code for project Zephyr is ZEPH-4417."),
    ("bob", "s1", "user", "I hate cilantro; it tastes like soap to me."),
    ("carol", "s9", "user", "Meeting notes: Q3 revenue was 4.2 million, up 12 percent."),
]


def add_worker(base: str, wid: int, n_rounds: int = 3) -> None:
    for rnd in range(n_rounds):
        for u, s, role, content in FACTS:
            payload = {
                "request_id": f"w{wid}-r{rnd}",
                "user_id": u,
                "session_id": s,
                "messages": [{"role": role, "content": content,
                              "timestamp": 1720000000000 + wid}],
            }
            code, body = post(base, ADD, payload)
            assert code == 200, f"add failed: {code} {body}"
            assert body.get("success") is True
            assert body["request_id"] == payload["request_id"]
            assert body["user_id"] == u and body["session_id"] == s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18000)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(args.port)],
        cwd=".",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        check("health 200 unauthenticated", wait_healthy(base), "service did not start")
        if FAILURES:
            return 1

        # --- 1. concurrent adds (64 workers) ---
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=64) as ex:
            futs = [ex.submit(add_worker, base, w) for w in range(64)]
            for f in cf.as_completed(futs):
                f.result()  # raises on assertion failure
        print(f"PASS 64 concurrent add workers completed in {time.time()-t0:.1f}s")

        # --- 2. durability: searchable immediately after 200 ---
        code, body = post(base, SEARCH, {
            "query": "What is the name of my dog?", "user_id": "alice", "top_k": 3})
        check("search 200", code == 200, f"got {code}")
        data = body.get("data", [])
        check("response has data list", isinstance(data, list))
        check("durability: planted fact searchable",
              any("Biscuit" in d.get("content", "") for d in data),
              f"top contents: {[d.get('content','')[:40] for d in data]}")

        # --- 3. schema of results ---
        if data:
            d0 = data[0]
            check("result has id/content/score/created_at",
                  all(k in d0 for k in ("id", "content", "score", "created_at")),
                  str(sorted(d0.keys())))

        # --- 4. top_k respected + ordering ---
        code, body = post(base, SEARCH, {
            "query": "dog peanut butter Lisbon dentist", "user_id": "alice", "top_k": 2})
        data = body.get("data", [])
        check("top_k respected", len(data) <= 2, f"got {len(data)}")
        scores = [d.get("score", 0) for d in data]
        check("most-relevant-first ordering",
              all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)),
              str(scores))

        # --- 5. user_id isolation ---
        code, body = post(base, SEARCH, {
            "query": "launch code Zephyr", "user_id": "alice", "top_k": 5})
        leaked = any("ZEPH" in d.get("content", "") for d in body.get("data", []))
        check("user isolation: alice cannot see bob's facts", not leaked)
        code, body = post(base, SEARCH, {
            "query": "launch code Zephyr", "user_id": "bob", "top_k": 5})
        found = any("ZEPH-4417" in d.get("content", "") for d in body.get("data", []))
        check("owner sees own facts", found)

        # --- 6. unknown user -> [] ---
        code, body = post(base, SEARCH, {
            "query": "anything at all", "user_id": "nobody-here", "top_k": 5})
        check("unknown user returns []", code == 200 and body.get("data") == [],
              str(body)[:100])

        # --- 7. relevance gating: irrelevant query -> [] ---
        code, body = post(base, SEARCH, {
            "query": "quantum chromodynamics lattice gauge beta function two-loop",
            "user_id": "alice", "top_k": 5})
        check("irrelevant query returns []", code == 200 and body.get("data") == [],
              f"got {len(body.get('data', []))} results")

        # --- 8. add response echo ---
        code, body = post(base, ADD, {
            "request_id": "echo-1", "user_id": "dave", "session_id": "s1",
            "messages": [{"role": "user", "content": "hello"}]})
        check("add echoes ids",
              code == 200 and body.get("success") is True
              and body.get("request_id") == "echo-1"
              and body.get("user_id") == "dave"
              and body.get("session_id") == "s1", str(body)[:120])

        print()
        if FAILURES:
            print(f"{len(FAILURES)} FAILURES:")
            for f in FAILURES:
                print("  -", f)
            return 1
        print("ALL SMOKE TESTS PASSED")
        return 0
    finally:
        proc.terminate()
        proc.wait(timeout=15)


if __name__ == "__main__":
    sys.exit(main())
