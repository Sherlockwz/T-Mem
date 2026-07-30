"""Credential-free self-test for the cost-ledger plumbing.

Verifies, WITHOUT any Venus API call:
  1. tiktoken encoder loads and counts.
  2. cost_ledger.record writes well-formed rows (tiktoken source).
  3. venus_provider._record_cost_safe extracts API usage from a fake `ret`
     and prefers it over tiktoken.
  4. conv contextvar set/get works and is picked up when conv_id omitted.
  5. Disabled mode (no T_MEM_COST_LOG) is a no-op.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

tmp = Path(tempfile.mkdtemp(prefix="tmem_cost_selftest_"))
ledger = tmp / "llm_calls.jsonl"
os.environ["T_MEM_COST_LOG"] = str(ledger)
os.environ["T_MEM_COST_STAGE"] = "stage_test"

from T_mem.utils import cost_ledger as CL  # noqa: E402

# 1) encoder + token count
enc = CL._get_encoder()
n = CL.count_tokens("Hello world, this is a token counting test.")
print(f"[1] encoder={CL._ENC_NAME}  sample_token_count={n}")
assert n > 0, "token count should be > 0"

# 2) direct record (tiktoken source) with conv via contextvar
CL.set_conv(3)
CL.record(prompt="a" * 400, completion="b" * 200, model="gpt-4.1-mini",
          call_site="stage2.item")  # conv_id omitted -> from ctx
# 3) record with API usage -> should prefer api
CL.record(prompt="short prompt", completion="short out", model="gpt-4.1-mini",
          call_site="stage1.boundary", conv_id=0,
          api_usage={"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168})

# 4) venus_provider recording path with a fake Venus `ret`
from T_mem.llm.venus_provider import _record_cost_safe, _extract_usage  # noqa: E402
fake_ret = {"data": {"response": "ok", "usage": {"prompt_tokens": 777,
                                                 "completion_tokens": 111,
                                                 "total_tokens": 888}}}
assert _extract_usage(fake_ret) == {"prompt_tokens": 777, "completion_tokens": 111, "total_tokens": 888}
_record_cost_safe("prompt text here", "resp text", "gpt-4.1-mini", fake_ret,
                  {"call_site": "stage4.scene_trigger", "conv_id": 7}, 0.42)

rows = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"[2-4] wrote {len(rows)} rows")
assert len(rows) == 3, f"expected 3 rows, got {len(rows)}"

r0, r1, r2 = rows
# row0: tiktoken source, conv from ctx = "3"
assert r0["token_source"] == "tiktoken" and r0["conv_id"] == "3" and r0["call_site"] == "stage2.item", r0
assert r0["prompt_tokens"] == r0["prompt_tokens_tok"] and r0["completion_tokens"] == r0["completion_tokens_tok"]
# row1: api source preferred
assert r1["token_source"] == "api" and r1["prompt_tokens"] == 123 and r1["completion_tokens"] == 45 and r1["total_tokens"] == 168, r1
assert r1["conv_id"] == "0" and r1["stage"] == "stage_test"
# row2: from venus path, api usage
assert r2["token_source"] == "api" and r2["prompt_tokens"] == 777 and r2["total_tokens"] == 888 and r2["conv_id"] == "7", r2
assert r2["call_site"] == "stage4.scene_trigger" and r2["latency_s"] == 0.42
print("[2-4] OK: sources/conv/callsite/latency all correct")

# 5) disabled mode = no-op
os.environ.pop("T_MEM_COST_LOG")
assert CL.enabled() is False
before = ledger.read_text(encoding="utf-8")
CL.record(prompt="x", completion="y", model="m", call_site="should_not_write")
after = ledger.read_text(encoding="utf-8")
assert before == after, "disabled mode must not write"
print("[5] OK: disabled mode is a no-op")

print(f"\nSELFTEST PASSED. ledger dir: {tmp}")
print("--- ledger rows ---")
for r in rows:
    print(json.dumps({k: r[k] for k in ("stage", "call_site", "conv_id", "model",
                                        "prompt_tokens", "completion_tokens", "total_tokens",
                                        "token_source")}, ensure_ascii=False))
print(f"LEDGER_PATH={ledger}")
