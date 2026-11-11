"""Build the exact-match oracle every later part is checked against.

1. 50 fixed prompts (short, long, code, lists, several languages), chat-formatted, greedy, 64 new
   tokens, HuggingFace in fp32 with attn_implementation="eager" -> tests/golden.json.
2. The same sequences teacher-forced through HuggingFace in fp16 at batch 1 and at batch 8 (left
   padded). The largest logit difference between the two runs is how far fp16 moves on its own;
   it becomes the fp16 tolerance -> tests/fp16_tolerance.json. Run step 2 on the T4 you benchmark
   on (python baselines/make_golden.py --fp16-only --device cuda).

    python baselines/make_golden.py                 # both steps, on CUDA if available
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden.json"
TOLERANCE = ROOT / "tests" / "fp16_tolerance.json"
MAX_NEW_TOKENS = 64
# Below this top-2 logit gap, fp32 itself cannot decide: HF's incremental generate() and HF's own
# full forward pass can pick different tokens (both differ by ~1e-5..1e-4 in fp32 logits).
FP32_TIE = 1e-4

_LONG_TEXT = (
    "The history of computing hardware covers the developments from early simple devices to aid "
    "calculation to modern day computers. The first aids to computation were purely mechanical devices "
    "which required the operator to set up the initial values of an elementary arithmetic operation, then "
    "manipulate the device to obtain the result. Later, computers represented numbers in a continuous form, "
    "for instance distance along a scale, rotation of a shaft, or a voltage. Numbers could also be "
    "represented in the form of digits, automatically manipulated by a mechanism. Although this approach "
    "generally required more complex mechanisms, it greatly increased the precision of results. The "
    "development of transistor technology and then the integrated circuit chip led to a series of "
    "breakthroughs, starting with transistor computers and then integrated circuit computers, causing "
    "digital computers to largely replace analog computers. Metal-oxide-semiconductor large-scale "
    "integration then enabled semiconductor memory and the microprocessor, leading to another key "
    "breakthrough, the miniaturized personal computer, in the 1970s. The cost of computers gradually "
    "became so low that personal computers by the 1990s, and then mobile computers, became ubiquitous."
)

PROMPTS: list[str] = [
    # short questions with long answers
    "Explain how a transformer language model generates text, step by step.",
    "What is the difference between a process and a thread?",
    "Why is the sky blue? Give a detailed scientific explanation.",
    "Describe the water cycle in detail.",
    "How does a hash map work internally?",
    "Explain gradient descent to a high school student.",
    "What are the main causes of inflation?",
    "Describe how vaccines train the immune system.",
    "Explain the CAP theorem with examples.",
    "How does public-key cryptography work?",
    "What happens when you type a URL into a browser and press enter?",
    "Explain the concept of opportunity cost with three examples.",
    # lists
    "List ten tips for writing clean Python code.",
    "Give me a numbered list of 8 healthy breakfast ideas with one sentence each.",
    "List the planets of the solar system and one interesting fact about each.",
    "What are seven common mistakes new managers make? Use bullet points.",
    "List the steps to make a cup of pour-over coffee.",
    "Give a bulleted list of the pros and cons of remote work.",
    "Name eight sorting algorithms and their time complexity.",
    "List six ways to reduce memory usage in a deep learning training job.",
    # code
    "Write a Python function that checks whether a string is a palindrome, with comments.",
    "Write a C function that reverses a singly linked list.",
    "Implement binary search in Python and explain each line.",
    "Write a SQL query that finds the top 5 customers by total order value, and explain it.",
    "Write a JavaScript debounce function and show how to use it.",
    "Write a PyTorch module for a two-layer MLP with ReLU and dropout.",
    "Write a bash script that counts lines in every .py file in a directory tree.",
    "Write a Rust function that returns the nth Fibonacci number iteratively.",
    "Implement an LRU cache class in Python.",
    "Show how to read a CSV file with pandas and compute the mean of each numeric column.",
    # long prompts
    f"Summarize the following text in five bullet points:\n\n{_LONG_TEXT}",
    f"Read this passage and write three quiz questions about it:\n\n{_LONG_TEXT}",
    f"Rewrite the following passage for a ten-year-old reader:\n\n{_LONG_TEXT}",
    f"What are the key technological transitions described here? Explain each.\n\n{_LONG_TEXT}",
    "Here is a conversation log. Alice: the server keeps crashing at night. Bob: did you check the "
    "memory graphs? Alice: memory climbs slowly all day and then the process is killed. Bob: sounds like "
    "a leak, maybe request objects are never freed. Alice: we cache every request in a dict for "
    "debugging. Bob: that's it. Question: what is the bug, and how should Alice fix it? Explain.",
    # creative / reasoning
    "Write a short story about a robot who learns to paint.",
    "Write a poem about the ocean at night.",
    "A train leaves at 3 pm going 60 km/h and another at 4 pm going 90 km/h on the same track. "
    "When does the second catch up? Show your reasoning.",
    "Compare and contrast cats and dogs as pets.",
    "Draft a polite email asking a professor for a recommendation letter.",
    "Plan a three-day itinerary for a first trip to Tokyo.",
    "Explain the rules of chess to a beginner.",
    "What would happen if the moon disappeared? Discuss several effects.",
    # other languages (multi-byte characters stress incremental detokenization)
    "用中文解释什么是机器学习，并举两个例子。",
    "请写一首关于春天的短诗，并解释它的意思。",
    "मशीन लर्निंग क्या है? सरल शब्दों में समझाइए।",
    "Explique en français comment fonctionne la photosynthèse.",
    "Erkläre auf Deutsch, wie ein Elektroauto funktioniert.",
    "Write a cheerful birthday message full of emoji 🎉🎂 for a friend who loves hiking.",
    "¿Cuáles son los beneficios de aprender un segundo idioma? Responde en español.",
]
assert len(PROMPTS) == 50


def chat_ids(tokenizer, prompt: str) -> list[int]:
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                        tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids)


def environment(device: str) -> dict:
    env = {"torch": torch.__version__, "python": platform.python_version(), "device": device}
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    if device.startswith("cuda"):
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda
    return env


@torch.inference_mode()
def make_golden(device: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 attn_implementation="eager").to(device).eval()
    cases = []
    t0 = time.time()
    for i, prompt in enumerate(PROMPTS):
        ids = chat_ids(tok, prompt)
        x = torch.tensor([ids], device=device)
        out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, temperature=None, top_p=None, top_k=None,
                             repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
        out_ids = out[0, len(ids):].tolist()
        cases.append({"prompt": prompt, "prompt_ids": ids, "output_ids": out_ids})
        print(f"[{i + 1:2d}/50] {len(ids):4d} prompt tokens -> {len(out_ids)} output tokens "
              f"({time.time() - t0:.0f}s) {ascii(tok.decode(out_ids)[:60])}")
    return {"model": MODEL, "dtype": "float32", "attn_implementation": "eager",
            "max_new_tokens": MAX_NEW_TOKENS, "decoding": "greedy", "env": environment(device), "cases": cases}


@torch.inference_mode()
def teacher_forced_logits(model, seqs: list[list[int]], n_out: list[int], device: str) -> list[torch.Tensor]:
    """Left-pad seqs into one batch; return, per sequence, fp32 logits at the positions that
    predicted its output tokens ([n_out, vocab])."""
    width = max(len(s) for s in seqs)
    pad = 0
    ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs], device=device)
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs], device=device)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    # Sequences are right-aligned, so the output positions are the last n of each row: keep only
    # the last max(n) positions' logits instead of [batch, width, 151936].
    keep = max(n_out)
    logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=keep).logits
    return [logits[b, keep - n:keep].float().cpu() for b, n in enumerate(n_out)]


@torch.inference_mode()
def annotate_fp32_ties(golden: dict, device: str) -> dict:
    """Teacher-force every golden sequence through HF fp32 (full forward, batch 1) and record the
    top-2 logit gap at each output position. A gap below FP32_TIE is an fp32 near-tie: HF's two
    code paths may disagree there, so no implementation can be held to one exact token."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 attn_implementation="eager").to(device).eval()
    for c in golden["cases"]:
        n = len(c["output_ids"])
        lg = teacher_forced_logits(model, [c["prompt_ids"] + c["output_ids"][:-1]], [n], device)[0]
        top2 = lg.topk(2, dim=-1)
        gaps = (top2.values[:, 0] - top2.values[:, 1]).tolist()
        c["top2_gap_fp32"] = [round(g, 7) for g in gaps]
        c["fp32_near_ties"] = [j for j, g in enumerate(gaps) if g < FP32_TIE]
        c["full_forward_disagrees_at"] = [j for j in range(n) if int(top2.indices[j, 0]) != c["output_ids"][j]]
    golden["fp32_tie_threshold"] = FP32_TIE
    return golden


@torch.inference_mode()
def measure_fp16_tolerance(golden: dict, device: str, batch: int = 8) -> dict:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="sdpa").to(device).eval()
    cases = golden["cases"]
    # Each prompt plus all but its last output token: the logits at the output positions are
    # the ones greedy decoding chose from.
    seqs = [c["prompt_ids"] + c["output_ids"][:-1] for c in cases]
    n_out = [len(c["output_ids"]) for c in cases]
    single = [teacher_forced_logits(model, [s], [n], device)[0] for s, n in zip(seqs, n_out)]
    batched: list[torch.Tensor] = []
    for i in range(0, len(seqs), batch):
        batched += teacher_forced_logits(model, seqs[i:i + batch], n_out[i:i + batch], device)
    per_prompt = [float((a - b).abs().max()) for a, b in zip(single, batched)]
    gaps = []
    for lg in single:
        top2 = lg.topk(2, dim=-1).values
        gaps.append(float((top2[:, 0] - top2[:, 1]).min()))
    return {"tolerance": max(per_prompt), "per_prompt_max_abs_diff": per_prompt,
            "min_top2_gap_per_prompt_fp16": gaps, "batch_sizes": [1, batch], "dtype": "float16",
            "method": "teacher-forced logits on golden sequences, HF fp16 batch 1 vs batch 8 (left padded)",
            "env": environment(device)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16-only", action="store_true", help="reuse golden.json; only measure the fp16 tolerance")
    ap.add_argument("--skip-fp16", action="store_true")
    ap.add_argument("--annotate-only", action="store_true", help="reuse golden.json; only record fp32 near-ties")
    args = ap.parse_args()
    if args.fp16_only or args.annotate_only:
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    else:
        golden = make_golden(args.device)
    if not args.fp16_only:
        golden = annotate_fp32_ties(golden, args.device)
        GOLDEN.write_text(json.dumps(golden, ensure_ascii=False, indent=1), encoding="utf-8")
        ties = {i: c["fp32_near_ties"] for i, c in enumerate(golden["cases"]) if c["fp32_near_ties"]}
        print(f"wrote {GOLDEN}; fp32 near-ties (gap < {FP32_TIE}) at {ties or 'none'}")
        if args.annotate_only:
            return
    if not args.skip_fp16:
        tol = measure_fp16_tolerance(golden, args.device)
        TOLERANCE.write_text(json.dumps(tol, indent=1), encoding="utf-8")
        print(f"wrote {TOLERANCE}: tolerance {tol['tolerance']:.4f} on {args.device}")


if __name__ == "__main__":
    main()
