"""Perplexity of every solution under each model: exp(-mean log p(t_i | t_<i)) over
the raw file in bfloat16, no prompt or chat template, with the first code token
(and a BOS, where the tokenizer adds one) left unscored.

    EMSE_MODEL_PROFILE=gemma-4-26b python -m pipeline.perplexity   # GPU
"""
from __future__ import annotations

import argparse
import math

from pipeline import config as C
from pipeline import data
from pipeline.runner import read_jsonl, write_jsonl

CHECKPOINT_EVERY = 200
# Longest sequence per forward pass, capped by the model's limit; longer inputs slide by STRIDE.
MAX_WINDOW = 8192
STRIDE = 512
# Positions per float32 log-softmax block; bounds memory, changes no number.
LOGIT_CHUNK = 512


def _token_logprobs(model, ids):
    """log p(t_i | t_<i) for positions 1..n-1, in float32."""
    import torch
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, :-1, :]
    targets = ids[0, 1:]
    out = torch.empty(targets.shape[0], dtype=torch.float32, device=logits.device)
    for s in range(0, logits.shape[0], LOGIT_CHUNK):
        lp = torch.log_softmax(logits[s:s + LOGIT_CHUNK].float(), dim=-1)
        out[s:s + LOGIT_CHUNK] = lp.gather(-1, targets[s:s + LOGIT_CHUNK].unsqueeze(-1)).squeeze(-1)
    return out


def _perplexity(model, tok_ids: list[int], window: int, skip: int) -> tuple[float, int, int]:
    """(ppl, n_scored, n_windows), strided over long inputs. Position 0 and `skip`
    more leading positions stay unscored but serve as context."""
    import torch
    device = next(model.parameters()).device
    n = len(tok_ids)
    total, count, windows, prev_end = 0.0, 0, 0, 1 + skip
    for begin in range(0, n, STRIDE):
        end = min(begin + window, n)
        lp = _token_logprobs(model, torch.tensor([tok_ids[begin:end]], device=device))
        tail = lp[-min(end - prev_end, lp.numel()):]
        total += float(tail.sum().item())
        count += tail.numel()
        windows += 1
        prev_end = end
        if end == n:
            break
    return math.exp(-total / count), count, windows


def score() -> None:
    """Score every dataset solution not yet in the active model's file."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = C.PROFILE.model_name
    out_path = C.DATA / "perplexity" / f"{C.ACTIVE_PROFILE}.jsonl"
    done = {r["solution_id"]: r for r in read_jsonl(out_path)}
    todo = [d for lang in C.LANGS for d in data.load_dataset(lang)
            if d["solution_id"] not in done]

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
    window = min(getattr(model.config, "max_position_embeddings", None) or MAX_WINDOW, MAX_WINDOW)
    rows = list(done.values())
    for i, d in enumerate(todo, 1):
        ids = tok(d["task"]).input_ids
        bos = int(ids[0] == tok.bos_token_id)
        ppl, n_scored, n_win = _perplexity(model, ids, window, skip=bos)
        rows.append({"solution_id": d["solution_id"], "lang": d["lang"], "problem": d["name"],
                     "ch": d["score"], "sloc": d["sloc"], "ppl": ppl, "log_ppl": math.log(ppl),
                     "n_tokens": len(ids), "n_scored": n_scored, "n_windows": n_win})
        if i % CHECKPOINT_EVERY == 0 or i == len(todo):
            write_jsonl(out_path, rows)
    print(f"{C.ACTIVE_PROFILE}: {len(todo)} scored, {len(rows)} in {out_path.name}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Perplexity under the active model "
                                        "(EMSE_MODEL_PROFILE); needs a GPU.").parse_args()
    score()
