"""The worker loop (python -m qccompute.serve), with the toy program in a subprocess."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from qcdata import ProgramInput, Structure

from qccompute.testing.toy import ToySession

ROOT = Path(__file__).resolve().parents[1]
ENV = {
    **os.environ,
    "PYTHONPATH": f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
}
ARGON3 = Structure(
    symbols=["Ar", "Ar", "Ar"],
    geometry=np.array([[0.0, 0.0, 0.0], [7.2, 0.0, 0.0], [3.0, 6.1, 0.4]]),
)
INP = ProgramInput.model_validate(
    {
        "program": "toy",
        "calctype": "gradient",
        "structure": ARGON3,
        "model": {"method": "lj"},
    }
)


def worker(spec="qccompute.testing.toy:program", *args):
    return subprocess.Popen(
        [sys.executable, "-m", "qccompute.serve", spec, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=ENV,
    )


class Client:
    def __init__(self, p):
        self.p = p
        self.hello = json.loads(p.stdout.readline())
        self.ids = 0

    def ask(self, **msg):
        self.ids += 1
        self.p.stdin.write(json.dumps({"id": self.ids, **msg}) + "\n")
        self.p.stdin.flush()
        reply = json.loads(self.p.stdout.readline())
        assert reply["id"] == self.ids
        return reply


@pytest.fixture
def client():
    p = worker()
    c = Client(p)
    yield c
    if p.poll() is None:
        p.stdin.close()
        p.wait(timeout=30)


def test_sessions_cross_the_wire_bit_for_bit(client):
    hello = client.hello
    assert hello["protocol"] == 1 and hello["program"] == "toy"
    assert "lj" in hello["capabilities"]["methods"]
    sid = client.ask(op="open", input=INP.model_dump(mode="json"))["result"]["session"]
    x = np.asarray(ARGON3.geometry) + 0.01
    first = client.ask(op="compute", session=sid)["result"]
    second = client.ask(op="compute", session=sid, positions=x.tolist())["result"]
    assert first["record"]["warm_start"] is False and second["record"]["warm_start"]
    with ToySession(INP) as s:
        here = [s.compute(), s.compute(x)]
    for wire, local in zip((first, second), here, strict=True):
        assert wire["results"]["energy"] == local.results.energy  # bit for bit
        assert np.array_equal(wire["results"]["gradient"], local.results.gradient)


def test_lineages_seeds_commits_and_resets(client):
    sid = client.ask(op="open", input=INP.model_dump(mode="json"))["result"]["session"]
    x = np.asarray(ARGON3.geometry)
    client.ask(op="compute", session=sid, lineage=["ts"])
    seeded = client.ask(
        op="compute",
        session=sid,
        positions=(x + 0.1).tolist(),
        lineage="f",
        seed=[["ts"]],
    )["result"]
    assert seeded["record"]["from_lineage"] == ["ts"]
    many = client.ask(
        op="compute_many",
        session=sid,
        requests=[
            {"positions": (x + 0.2).tolist(), "lineage": "f", "commit": False},
            {"positions": np.full((3, 3), 1.0).tolist()[:2]},
        ],
    )["result"]
    assert many[0]["success"] and many[0]["record"]["from_lineage"] == "f"
    assert not many[1]["success"] and many[1]["error"]["kind"] == "input"
    assert sorted(
        map(json.dumps, client.ask(op="lineages", session=sid)["result"])
    ) == [
        '"f"',
        '["ts"]',
    ]
    client.ask(op="compute", session=sid)  # the default lineage
    assert client.ask(op="reset", session=sid, lineage=None)["ok"]
    assert None not in client.ask(op="lineages", session=sid)["result"]
    assert client.ask(op="reset", session=sid)["ok"]  # absent: every lineage
    assert client.ask(op="lineages", session=sid)["result"] == []


def test_failures_and_plans(client):
    sid = client.ask(
        op="open", input=INP.model_dump(mode="json"), calctypes=["gradient"]
    )
    sid = sid["result"]["session"]
    hessian = client.ask(op="compute", session=sid, calctype="hessian")
    assert hessian["ok"] and not hessian["result"]["success"]  # a failed frame
    assert hessian["result"]["error"]["code"] == "not_applicable"
    nan = client.ask(op="compute", session=sid, positions=[[float("nan")] * 3] * 3)
    assert not nan["ok"] and "NaN" in nan["error"]["message"]  # refused on the wire
    assert client.ask(op="plan", session=sid)["result"]["resolved"]["settings"]
    at_hessian = client.ask(op="plan", session=sid, calctype="hessian")["result"]
    assert at_hessian["resolved"]["settings"]
    one = client.ask(op="compute", input=INP.model_dump(mode="json"))["result"]
    assert one["success"] and one["input_data"]["program"] == "toy"
    bad = INP.model_dump(mode="json") | {"keywords": {"sigmaa": 1.0}}
    refused = client.ask(op="open", input=bad)
    assert not refused["ok"] and refused["error"]["kind"] == "input"
    assert (
        client.ask(op="compute", session="nope")["error"]["code"] == "unknown_session"
    )
    assert not client.ask(op="fly")["ok"]
    client.p.stdin.write("{not json\n")
    client.p.stdin.flush()
    assert json.loads(client.p.stdout.readline())["ok"] is False
    assert client.ask(op="close", session=sid)["ok"]
    client.p.stdin.close()
    assert client.p.wait(timeout=30) == 0
    assert client.p.stdout.read() == ""


def test_stdout_carries_protocol_lines_only():
    c = Client(worker("tests.serve_programs:noisy"))
    noisy = INP.model_copy(update={"program": "noisy"})
    assert c.ask(op="compute", input=noisy.model_dump(mode="json"))["result"]["success"]
    c.p.stdin.close()
    assert c.p.wait(timeout=30) == 0
    assert c.p.stdout.read() == ""
    stderr = c.p.stderr.read()
    assert "python print on stdout" in stderr and "fd 1 write" in stderr


def test_device_lost_is_answered_then_the_worker_exits():
    inp = INP.model_copy(update={"program": "device-lost"}).model_dump(mode="json")
    lines = [{"id": 1, "op": "compute", "input": inp}, {"id": 2, "op": "fly"}]
    r = subprocess.run(
        [sys.executable, "-m", "qccompute.serve", "tests.serve_programs:device_lost"],
        input="".join(json.dumps(m) + "\n" for m in lines),
        capture_output=True,
        text=True,
        env=ENV,
    )
    replies = [json.loads(line) for line in r.stdout.splitlines()]
    assert r.returncode == 3 and len(replies) == 2, r.stderr[-2000:]
    assert replies[1]["result"]["error"]["code"] == "device_lost"


def test_socket_sessions_outlive_connections(tmp_path):
    path = tmp_path / "w.sock"
    p = worker("qccompute.testing.toy:program", "--socket", str(path))
    try:
        for _ in range(200):
            if path.exists():
                break
            time.sleep(0.05)

        def connect(messages):
            with socket.socket(socket.AF_UNIX) as conn:
                conn.connect(str(path))
                f = conn.makefile("rw", encoding="utf-8")
                replies = [json.loads(f.readline())]
                for m in messages:
                    f.write(json.dumps(m) + "\n")
                    f.flush()
                    replies.append(json.loads(f.readline()))
                return replies

        first = connect(
            [
                {"id": 1, "op": "open", "input": INP.model_dump(mode="json")},
                {"id": 2, "op": "compute", "session": "s1"},
            ]
        )
        assert first[0]["program"] == "toy" and first[2]["result"]["success"]
        again = connect([{"id": 1, "op": "compute", "session": "s1"}])
        assert again[1]["result"]["record"]["warm_start"] is True
    finally:
        p.terminate()
        p.wait(timeout=30)
