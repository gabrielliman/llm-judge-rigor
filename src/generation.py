"""Judge generation with vLLM: prompts, guided JSON decoding, retry protocol, raw records.

All requests of a model are sent to one loaded ``vllm.LLM`` instance, so vLLM batches
across prompts and items. Decoding is greedy with JSON-schema guided decoding
(``SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))``, vLLM 0.32).
Built-in thinking is disabled through the chat template (``enable_thinking=False``) and
no reasoning parser is used.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "prompts"


def load_prompt(pid: str) -> str:
    """System prompt ``prompts/{pid}.txt`` (verbatim)."""
    return (PROMPT_DIR / f"{pid}.txt").read_text(encoding="utf-8")


def load_user_template() -> str:
    """User message template with ``{TEXTO}`` and ``{OPINIAO}`` placeholders."""
    return (PROMPT_DIR / "user_template.txt").read_text(encoding="utf-8")


def load_schema(name: str) -> dict:
    """JSON schema ``prompts/schemas/{name}.json``."""
    return json.loads((PROMPT_DIR / "schemas" / f"{name}.json").read_text(encoding="utf-8"))


def prompt_sha256(pid: str, schema_name: str) -> str:
    """Hash of system prompt + user template + schema, stored with every output."""
    h = hashlib.sha256()
    for part in (load_prompt(pid), load_user_template(), json.dumps(load_schema(schema_name), sort_keys=True)):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def build_messages(pid: str, texto: str, opiniao: str) -> list[dict]:
    """System + user messages for one item.

    Placeholders are filled with ``str.replace`` (not ``str.format``) so that braces in
    transcripts cannot break the template.
    """
    user = load_user_template().replace("{TEXTO}", texto).replace("{OPINIAO}", opiniao)
    return [{"role": "system", "content": load_prompt(pid)}, {"role": "user", "content": user}]


def parse_output(text: str, schema: dict):
    """Parsed JSON object, or ``None`` unless it has exactly the schema's keys and a
    boolean ``opiniao_inferida``."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != set(schema["properties"]):
        return None
    if not isinstance(obj.get("opiniao_inferida"), bool):
        return None
    return obj


def make_llm(model_cfg: dict, gen_cfg: dict, seed: int, tensor_parallel: int = 1,
             no_whitespace: bool = False):
    """Load one model at its pinned revision.

    ``model_cfg["engine"]`` may add engine arguments that only affect memory (e.g.
    ``max_num_seqs``). ``no_whitespace`` makes the JSON grammar forbid free whitespace;
    in vLLM 0.32 this is an engine-level option of the xgrammar backend.
    """
    from vllm import LLM

    extra = dict(model_cfg.get("engine") or {})
    if no_whitespace:
        from vllm.config import StructuredOutputsConfig
        extra["structured_outputs_config"] = StructuredOutputsConfig(backend="xgrammar",
                                                                     disable_any_whitespace=True)
    return LLM(model=model_cfg["id"], revision=model_cfg["revision"],
               tokenizer_revision=model_cfg["revision"], dtype=gen_cfg["dtype"],
               max_model_len=gen_cfg["max_model_len"],
               gpu_memory_utilization=gen_cfg["gpu_memory_utilization"],
               limit_mm_per_prompt=gen_cfg["limit_mm_per_prompt"],
               tensor_parallel_size=tensor_parallel, seed=seed, **extra)


def sampling_params(schema: dict, max_tokens: int, seed: int, logprobs: int | None = None):
    """Greedy decoding constrained to ``schema``; optionally top-``logprobs`` per token."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    return SamplingParams(temperature=0.0, seed=seed, max_tokens=max_tokens, logprobs=logprobs,
                          structured_outputs=StructuredOutputsParams(json=schema))


def verdict_logprobs(c) -> dict | None:
    """Log-probabilities of ``true``/``false`` at the verdict token.

    vLLM returns raw log-probabilities, i.e. before the grammar mask, so they reflect the
    model's own preference. The verdict token is the last generated token that follows
    ``"opiniao_inferida":`` and starts a prefix of ``true``/``false``; top-k alternatives
    that are prefixes of each word are summed. Returns ``p_halluc`` = P(false) /
    (P(true) + P(false)), or ``None`` if the token is not found.
    """
    if not c.logprobs:
        return None
    pieces, text_so_far = [], ""
    for k, tid in enumerate(c.token_ids):
        tok = c.logprobs[k][tid].decoded_token if tid in c.logprobs[k] else ""
        pieces.append((k, text_so_far, tok))
        text_so_far += tok or ""
    for k, before, tok in reversed(pieces):
        st = (tok or "").strip().lower()
        if st and ("true".startswith(st) or "false".startswith(st)) and \
                before.rstrip().endswith(":") and '"opiniao_inferida"' in before[-40:]:
            lt = lf = None
            for alt in c.logprobs[k].values():
                a = (alt.decoded_token or "").strip().lower()
                if not a:
                    continue
                if "true".startswith(a):
                    lt = alt.logprob if lt is None else float(np.logaddexp(lt, alt.logprob))
                elif "false".startswith(a):
                    lf = alt.logprob if lf is None else float(np.logaddexp(lf, alt.logprob))
            if lt is None and lf is None:
                return None
            lt = -np.inf if lt is None else lt
            lf = -np.inf if lf is None else lf
            m = max(lt, lf)
            ph = float(np.exp(lf - m) / (np.exp(lt - m) + np.exp(lf - m)))
            return {"lp_true": float(lt), "lp_false": float(lf), "p_halluc": ph, "pos": k}
    return None


def run_model(llm, model_cfg: dict, prompt_schemas: dict, requests: list[tuple[str, dict]],
              gen_cfg: dict, seed: int, out_dir: Path, logprobs: int | None = None,
              chunk_size: int | None = None) -> dict:
    """Generate ``requests`` = [(prompt_id, item), ...] with an already-loaded model.

    ``item`` is a dict with ``item_uid``, ``hearing_id``, ``texto`` and ``opiniao``.
    Retry protocol: outputs that hit ``max_tokens`` or do not parse are regenerated once
    with ``max_tokens_retry``; if they fail again the verdict is missing (``flag = None``).
    Writes ``{out_dir}/{model}__{prompt}.jsonl`` (one record per request; first attempts
    that failed go to ``*.attempt1_failures.jsonl``) and returns per-prompt diagnostics
    and timing. ``chunk_size`` splits the first pass into several ``llm.chat`` calls to
    bound host memory when per-token log-probabilities are requested.
    """
    pids = list(dict.fromkeys(p for p, _ in requests))
    schemas = {p: load_schema(prompt_schemas[p]) for p in pids}
    shas = {p: prompt_sha256(p, prompt_schemas[p]) for p in pids}
    convs = [build_messages(p, it["texto"], it["opiniao"]) for p, it in requests]

    def params(max_tokens):
        return {p: sampling_params(schemas[p], max_tokens, seed, logprobs) for p in pids}

    def chat(conversations, sps):
        return llm.chat(conversations, sampling_params=sps, use_tqdm=True,
                        chat_template_kwargs={"enable_thinking": gen_cfg["enable_thinking"]})

    def record(p, it, c, attempt):
        parsed = parse_output(c.text, schemas[p])
        ok = parsed is not None and c.finish_reason != "length"
        verdict = parsed["opiniao_inferida"] if ok else None
        rec = {"item_uid": it["item_uid"], "hearing_id": it["hearing_id"],
               "model_id": model_cfg["id"], "model_revision": model_cfg["revision"],
               "prompt_id": p, "prompt_sha256": shas[p], "raw_text": c.text,
               "parsed": parsed if ok else None, "opiniao_inferida": verdict,
               "flag": None if verdict is None else int(not verdict),   # flag 1 = hallucination
               "finish_reason": c.finish_reason, "n_output_tokens": len(c.token_ids),
               "attempt": attempt, "timestamp": datetime.now(timezone.utc).isoformat()}
        if logprobs:
            rec["verdict_logprobs"] = verdict_logprobs(c)
        return rec

    t0 = time.time()
    sp1 = params(gen_cfg["max_tokens"])
    step = chunk_size or len(requests)
    recs = []
    for a in range(0, len(requests), step):
        batch = requests[a:a + step]
        outs = chat(convs[a:a + step], [sp1[p] for p, _ in batch])
        recs += [record(p, it, o.outputs[0], 1) for (p, it), o in zip(batch, outs)]
    retry = [i for i, r in enumerate(recs) if r["flag"] is None]
    failed_first = {i: recs[i] for i in retry}
    if retry:
        sp2 = params(gen_cfg["max_tokens_retry"])
        outs = chat([convs[i] for i in retry], [sp2[requests[i][0]] for i in retry])
        for i, o in zip(retry, outs):
            recs[i] = record(requests[i][0], requests[i][1], o.outputs[0], 2)
    seconds = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    markers = gen_cfg["thinking_markers"]
    diagnostics = []
    for p in pids:
        idx = [i for i, (q, _) in enumerate(requests) if q == p]
        path = out_dir / f"{model_cfg['short']}__{p}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(recs[i], ensure_ascii=False) + "\n" for i in idx)
        first = [failed_first[i] for i in idx if i in failed_first]
        if first:
            with open(path.with_suffix(".attempt1_failures.jsonl"), "w", encoding="utf-8") as fh:
                fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in first)
        rs = [recs[i] for i in idx]
        diagnostics.append({
            "prompt_id": p, "n_items": len(rs), "n_retried": len(first),
            "n_missing": sum(r["flag"] is None for r in rs),
            "n_thinking_marker": sum(any(m in r["raw_text"] for m in markers) for r in rs),
            "keys_out_of_order": sum(r["parsed"] is not None
                                     and list(r["parsed"]) != list(schemas[p]["properties"]) for r in rs),
        })
    return {"model": model_cfg["short"], "n_requests": len(requests), "n_retried": len(retry),
            "seconds": seconds, "requests_per_min": 60 * len(requests) / seconds,
            "per_prompt": diagnostics}
