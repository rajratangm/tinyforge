"""Step 3 check: GGUF (llama.cpp) vs the merged HF model on the held-out SQL set, greedy decoding.

Reports end-to-end generation speed and output agreement. Run from the project root with the project venv:
    PYTHONPATH=tools/llama.cpp/src/llama.cpp-b11380/gguf-py .venv/Scripts/python.exe tools/gguf_parity.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from tinyforge.ft_task import sql_score  # noqa: E402

ROOT = Path(".")
RUN = ROOT / "runs" / "sql"
SERVER = ROOT / "tools" / "llama.cpp" / "bin" / "llama-server.exe"
N, MAX_NEW, PORT = 173, 96, 8099
VARIANTS = ["f16", "q8_0", "q4_k_m"]


def load_val() -> tuple[list[str], list[str]]:
    recs = [json.loads(line)["messages"] for line in
            (ROOT / "data" / "sql" / "val.jsonl").read_text(encoding="utf-8").splitlines()][:N]
    users = [next(m["content"] for m in r if m["role"] == "user") for r in recs]
    return users, [r[-1]["content"] for r in recs]


def run_hf(users: list[str]) -> tuple[list[str], float]:
    tok = AutoTokenizer.from_pretrained(RUN / "merged")
    model = AutoModelForCausalLM.from_pretrained(RUN / "merged", torch_dtype=torch.float16).to("cuda").eval()
    outs, n_tok, t_gen = [], 0, 0.0
    for u in users:
        text = tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        t_gen += time.perf_counter() - t0
        new = out[0][enc.input_ids.size(1):]
        n_tok += len(new)
        outs.append(tok.decode(new, skip_special_tokens=True).strip())
    del model
    torch.cuda.empty_cache()
    return outs, n_tok / t_gen


def run_gguf(variant: str, users: list[str]) -> tuple[list[str], float]:
    proc = subprocess.Popen([str(SERVER), "-m", str(RUN / "gguf" / f"model-{variant}.gguf"), "-ngl", "99",
                             "-c", "2048", "--port", str(PORT), "--log-disable"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server did not become healthy")
        outs, n_tok, t_gen = [], 0, 0.0
        for u in users:
            body = json.dumps({"messages": [{"role": "user", "content": u}], "temperature": 0,
                               "max_tokens": MAX_NEW, "cache_prompt": False}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                         {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120))
            outs.append(r["choices"][0]["message"]["content"].strip())
            n_tok += r["usage"]["completion_tokens"]
            t_gen += r["timings"]["predicted_ms"] / 1000
        return outs, n_tok / t_gen
    finally:
        proc.terminate()
        proc.wait()


def rates(users: list[str], outs: list[str], refs: list[str]) -> dict:
    rows = [sql_score(u, o, r) for u, o, r in zip(users, outs, refs, strict=True)]
    return {k: round(sum(x[k] for x in rows) / len(rows), 4) for k in ("strict_em", "lenient_em", "valid")}


def main() -> None:
    users, refs = load_val()
    hf_out, hf_tps = run_hf(users)
    res = {"n": len(users), "hf_fp16": {"tok_s": round(hf_tps, 1), **rates(users, hf_out, refs)}}
    diffs = {}
    for v in VARIANTS:
        out, tps = run_gguf(v, users)
        same = [a == b for a, b in zip(out, hf_out, strict=True)]
        res[v] = {"tok_s": round(tps, 1), "identical_to_hf": round(sum(same) / len(same), 4),
                  **rates(users, out, refs)}
        diffs[v] = [{"question": u.split("\n\n")[0], "hf": h, "gguf": o}
                    for u, h, o, s in zip(users, hf_out, out, same, strict=True) if not s][:5]
    res["example_diffs"] = diffs
    (RUN / "gguf" / "parity.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "example_diffs"}, indent=2))


if __name__ == "__main__":
    main()
