import bisect
import os, re, copy, time, json, math, random, contextlib, types
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
model = tokenizer = device = None

FIRST = "Maya Arjun Elena Tomas Priya Jonas Lucia Kenji Amara Viktor Sana Mateo".split()
LAST = "Okafor Lindqvist Raman Duarte Nakamura Haddad Petrov Moreau Banerjee Costa Weiss Iyer".split()
MIDDLE_KEEP = 16
PROJ = "Falcon Heron Osprey Kestrel Merlin Condor Harrier Swift Plover Tern Egret Lapwing".split()
QTXT = {"id": "What is the official identifier of Project {P}?",
        "mgr": "Who is the manager of Project {P}?",
        "bud": "What is the approved budget of Project {P}?"}
RECENT_KEEP = 16
START_KEEP = 16
TOTAL_KEEP = START_KEEP + MIDDLE_KEEP + RECENT_KEEP
T_BD = ["{P} budget: {X}.",
        "The approved budget for Project {P} is {X}.",
        "After several rounds of negotiation with the finance committee, the approved budget for Project {P} was set at {X} for the full year."]
T_ID = ["{P} code: {X}.",
        "The official identifier for Project {P} is {X}.",
        "After the quarterly compliance review concluded, the registry assigned Project {P} the official identifier {X}, effective immediately."]
T_MG = ["{P} manager: {X}.",
        "Project {P} is managed by {X}.",
        "Following the leadership reorganization, {X} was appointed as the manager responsible for Project {P} and its deliverables."]
WORDS = ("quarterly inventory vendor schedule compliance audit warehouse shipment training facility "
         "maintenance forecast regional supplier contract review staffing logistics procurement safety "
         "renewal inspection transport network upgrade policy meeting summary workflow capacity delivery "
         "handover archive calendar region operations portfolio").split()
expected = "RG-920-Z"
facts = {
    "beginning": "ORCHID-7319",
    "middle": "COBALT-4826",
    "end": "MAPLE-9054",
}
length = 233
question = "What is the assigned project identifier?"
target = "RG-920-Z"
test_question = "What is the assigned project identifier?"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
best = None
_h2o_cache = {}
REPO = "/content/recallguard-kv-repo"
RESULTS_DIR = "/content/recallguard_recovered_results"
REPO = "/content/recallguard-kv-repo"
RESULTS_DIR = "/content/recallguard_recovered_results"
new = ("`exact`: expected value found by a boundary-aware regex in the FIRST LINE of the generated answer "
       "(max 40 new tokens). `strict`: exact AND no known decoy value for that question appears in that line.")
old = "`strict`: correct value present and no other candidate value. `exact`: correct value present."
p = f"{REPO}/README.md"
OUT = f"{REPO}/code/recallguard_core.py"
CURRENT_Q_IDS = None
q_ids = CURRENT_Q_IDS  # alias for snap_policy

def load_model(name="Qwen/Qwen2.5-0.5B-Instruct"):
    global model, tokenizer, device
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("cuda")
    device = next(model.parameters()).device
    return model, tokenizer, device

@contextlib.contextmanager
def eager_attn(model):
    old = model.config._attn_implementation
    try:
        try: model.set_attn_implementation("eager")
        except Exception: model.config._attn_implementation = "eager"
        yield
    finally:
        try: model.set_attn_implementation(old)
        except Exception: model.config._attn_implementation = old

def _compress_by_indices(cache, idx):
    out = copy.deepcopy(cache)
    t = torch.tensor(idx, dtype=torch.long)
    for l in out.layers:
        if getattr(l, "keys", None) is None:
            continue
        ti = t.to(l.keys.device)
        l.keys = l.keys.index_select(2, ti).contiguous()
        l.values = l.values.index_select(2, ti).contiguous()
    return out

def _continue(model, tokenizer, cache, orig_len, q_ids, mode, n_new=12):
    dev = q_ids.device
    def step(ids, pos0):
        r = cache.layers[0].keys.shape[2]
        kw = dict(input_ids=ids, past_key_values=cache, use_cache=True,
                  attention_mask=torch.ones(1, r + ids.shape[1], dtype=torch.long, device=dev))
        if mode == "absolute":
            kw["position_ids"] = torch.arange(pos0, pos0 + ids.shape[1], device=dev).unsqueeze(0)
        return model(**kw)
    q = q_ids.shape[1]
    out = step(q_ids, orig_len)
    cache = out.past_key_values
    first = out.logits[0, -1].float().cpu()
    logits, toks = first, []
    for s in range(n_new):
        t = logits.argmax().item()
        if t == tokenizer.eos_token_id:
            break
        toks.append(t)
        out = step(torch.tensor([[t]], device=dev), orig_len + q + s)
        cache = out.past_key_values
        logits = out.logits[0, -1].float().cpu()
    return first, tokenizer.decode(toks, skip_special_tokens=True)

@torch.no_grad()
def eval_case(model, tokenizer, device, ctx, question, ident, policy_fn, mode, n_new=16):
    toks, L = id_token_span(tokenizer, ctx, ident)
    ids = tokenizer(ctx, return_tensors="pt").input_ids.to(device)
    q_ids = tokenizer("\nQuestion: " + question + "\nAnswer:", add_special_tokens=False,
                      return_tensors="pt").input_ids.to(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
    _sync(); t = time.perf_counter()
    cache = model(input_ids=ids, use_cache=True).past_key_values
    _sync(); prefill_s = time.perf_counter() - t
    t = time.perf_counter()
    idx = list(range(L)) if policy_fn is None else policy_fn(cache)
    _sync(); selector_s = time.perf_counter() - t
    t = time.perf_counter()
    work = _compress_by_indices(cache, idx)
    _sync(); compress_s = time.perf_counter() - t
    kv_mib_pre = kv_bytes(work) / 2**20          # NEW: measured BEFORE question/generation
    n_pos_pre = work.layers[0].keys.shape[2]     # NEW: retained positions before generation
    t = time.perf_counter()
    _, ans = _continue(model, tokenizer, work, L, q_ids, mode, n_new=n_new)
    _sync(); continue_s = time.perf_counter() - t
    kept = set(idx)
    peak = (torch.cuda.max_memory_allocated() - base_mem) / 2**20 if torch.cuda.is_available() else float("nan")
    return dict(
        L=L, depth=(toks[0] + toks[-1]) / 2 / L, id_kept=sum(i in kept for i in toks) / len(toks),
        exact=ident in ans, answer=ans,
        kv_mib_pre=kv_mib_pre, n_pos_pre=n_pos_pre,
        kv_mib=kv_bytes(work) / 2**20,           # original field: measured AFTER generation
        prefill_s=prefill_s, selector_s=selector_s, compress_s=compress_s, continue_s=continue_s,
        peak_delta_mib=peak, idx=idx, id_toks=toks,
    )

def select_snap_span_indices(cache, model, q_ids, start_keep=8, num_mid_spans=3,
                             span_size=8, recent_keep=16, pool=5):
    L = cache.layers[0].keys.shape[2]
    budget = start_keep + num_mid_spans * span_size + recent_keep
    if L <= budget:
        return list(range(L))
    lo, hi = start_keep, L - recent_keep
    q, dev = q_ids.shape[1], q_ids.device
    work = copy.deepcopy(cache)
    with eager_attn(model):
        out = model(input_ids=q_ids, past_key_values=work, use_cache=True,
                    attention_mask=torch.ones(1, L + q, dtype=torch.long, device=dev),
                    position_ids=torch.arange(L, L + q, device=dev).unsqueeze(0),
                    output_attentions=True)
    atts = out.attentions
    if atts is None or atts[0] is None:
        raise RuntimeError("No attentions returned: eager attention switch failed.")
    n = len(atts)
    score = sum(a[0, :, :, lo:hi].float().sum(dim=(0, 1)) for a in atts[n // 3:])
    score = F.avg_pool1d(score.view(1, 1, -1), kernel_size=pool, stride=1,
                         padding=pool // 2).view(-1)
    span_scores = F.avg_pool1d(score.view(1, 1, -1), kernel_size=span_size, stride=1).view(-1)
    offs = _select_k_non_overlapping_spans_dp(span_scores.cpu().tolist(), num_mid_spans, span_size)
    idx = (list(range(start_keep))
           + [lo + o + t for o in offs for t in range(span_size)]
           + list(range(L - recent_keep, L)))
    assert len(idx) == budget == len(set(idx)) and idx == sorted(idx)
    return idx

def _as_list(x): return torch.as_tensor(x).flatten().tolist()

def _like(orig, lst):
    return torch.tensor(lst, dtype=orig.dtype, device=orig.device) if torch.is_tensor(orig) else lst

def _select_k_non_overlapping_spans_dp(span_scores, k_spans, span_size):
    n = len(span_scores)
    if k_spans == 0:
        return []
    if span_size < 1 or k_spans < 0:
        raise ValueError("bad args")
    if n < (k_spans - 1) * span_size + 1:
        raise ValueError(f"infeasible: {n} span starts cannot fit {k_spans} spans of {span_size}")
    NEG = float("-inf")
    best = [[NEG] * (n + 1) for _ in range(k_spans + 1)]
    for i in range(n + 1):
        best[0][i] = 0.0
    for j in range(1, k_spans + 1):
        for i in range(n - 1, -1, -1):
            take = span_scores[i] + best[j - 1][min(i + span_size, n)]
            best[j][i] = max(best[j][i + 1], take)
    out, i, j = [], 0, k_spans
    while j > 0:
        take = span_scores[i] + best[j - 1][min(i + span_size, n)]
        if take >= best[j][i + 1]:
            out.append(i); j -= 1; i = min(i + span_size, n)
        else:
            i += 1
    return out

def _sent(rng):
    return (f"Report {rng.randint(100,999)}: unit {rng.choice(WORDS)} logged value "
            f"{rng.randint(10,9999)} at site {rng.choice(WORDS)}-{rng.randint(1,99)}.")

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def boundary_fn(B, q_ids, ctx):
    ids = tokenizer(ctx, add_special_tokens=False).input_ids
    toks = [tokenizer.decode([t]) for t in ids]
    ends = [i for i, t in enumerate(toks)
            if (t.rstrip().endswith((".", "!", "?")) or "\n" in t)
            and not (i + 1 < len(toks) and toks[i + 1].strip()[:1].isdigit())]
    starts = [0] + [e + 1 for e in ends]
    def expand(idx, L):
        runs = []
        for i in sorted(i for i in set(idx) if 8 <= i < L - 16):
            if runs and i == runs[-1][1] + 1: runs[-1][1] = i
            else: runs.append([i, i])
        keep = set(idx)
        for a, b in runs:
            s = starts[bisect.bisect_right(starts, a) - 1]
            j = bisect.bisect_left(ends, b)
            e = ends[j] if j < len(ends) else L - 1
            keep.update(range(max(s, 8), min(e, L - 17) + 1))
        return sorted(keep)
    def fn(c):
        L = c.layers[0].keys.shape[2]
        assert L == len(ids), f"cache {L} != tokens {len(ids)}"
        chosen = None
        for k in range((B - 24) // 8, 0, -1):          # most spans that still fit after stretching
            raw = select_snap_span_indices(c, model, q_ids, num_mid_spans=k)
            new = expand(_as_list(raw), L)
            if len(new) <= B: chosen = (raw, new); break
        if chosen is None:
            raw = select_snap_span_indices(c, model, q_ids, num_mid_spans=1)
            chosen = (raw, _as_list(raw))
        raw, new = chosen
        return _like(raw, pad_to_budget(new, L, B))
    return fn

def build_hard_context(tokenizer, ident, target, seed, n_sent=14, n_decoys=3):
    rng = random.Random(seed)
    sents = [_sent(rng) for _ in range(n_sent)]
    fpos = min(n_sent - 1, int(target * n_sent))
    others = [i for i in range(n_sent) if i != fpos]
    for i in rng.sample(others, n_decoys):
        d = f"RG-{rng.randint(100,999)}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
        sents[i] = f"DECOY RECORD: The archived project identifier is {d}."
    sents[fpos] = f"IMPORTANT RECORD: The assigned project identifier is {ident}."
    return " ".join(sents)

def compress_cache_position_aware(cache):
    """Keep start, middle, and recent KV positions."""
    result = copy.deepcopy(cache)

    for layer in result.layers:
        if layer.keys is None or layer.values is None:
            continue

        length = layer.keys.shape[2]
        if length <= TOTAL_KEEP:
            continue

        indices = select_position_aware_indices(length)
        index_tensor = torch.tensor(
            indices, device=layer.keys.device, dtype=torch.long
        )

        layer.keys = layer.keys.index_select(
            2, index_tensor
        ).contiguous()
        layer.values = layer.values.index_select(
            2, index_tensor.to(layer.values.device)
        ).contiguous()

    return result

def first_line(a): return a.strip().split("\n")[0]

def id_token_span(tokenizer, ctx, ident):
    enc = tokenizer(ctx, return_offsets_mapping=True)
    c0 = ctx.index(ident); c1 = c0 + len(ident)
    toks = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > s and s < c1 and e > c0]
    return toks, len(enc["input_ids"])

def kv_bytes(cache):
    return sum(l.keys.numel() * l.keys.element_size() + l.values.numel() * l.values.element_size()
               for l in cache.layers if getattr(l, "keys", None) is not None)

def make_case(seed, n_sent):
    rng = random.Random(seed)
    projs = rng.sample(PROJ, 4)
    used = {"id": set(), "mgr": set(), "bud": set()}
    def uniq(kind, fn):
        while True:
            v = fn()
            if v not in used[kind]:
                used[kind].add(v); return v
    rid = lambda: f"RG-{rng.randint(100,999)}-{rng.choice('ABCDEFGHJKMNPQRSTUVWXYZ')}"
    person = lambda: f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    def money():
        n = rng.randrange(20000, 9000000, 250)
        s = f"${n:,}"
        return s + rng.choice([".25", ".50", ".75"]) if rng.random() < 0.4 else s
    facts, ans, decoys = [], {}, {"id": [], "mgr": [], "bud": []}
    for pi, P in enumerate(projs):
        for kind, T, gen in (("id", T_ID, rid), ("mgr", T_MG, person), ("bud", T_BD, money)):
            v = uniq(kind, gen)
            facts.append(T[rng.randint(0, 2)].format(P=P, X=v))
            if pi == 0: ans[kind] = v
            else: decoys[kind].append(v)
    pos = rng.sample(range(2, n_sent - 2), len(facts))
    sents = ["Report " + " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 26))) + "."
             for _ in range(n_sent)]
    for p, f in zip(pos, facts): sents[p] = f
    qa = [(k, QTXT[k].format(P=projs[0]), ans[k]) for k in ("id", "mgr", "bud")]
    return dict(ctx=" ".join(sents), qa=qa, decoys=decoys)

def make_snap_policy(question):
    def policy(cache):
        q_ids = tokenizer(
            "\nQuestion: " + question + "\nAnswer:",
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to("cuda")

        return select_snap_span_indices(
            cache, model, q_ids,
            start_keep=8,
            num_mid_spans=3,
            span_size=8,
            recent_keep=16,
            pool=5
        )
    return policy

def pad_to_budget(idx, L, B):
    keep = set(_as_list(idx))
    lo, hi = 8, L - 16                      # middle zone; first 8 and last 16 are already kept
    runs = []
    for i in sorted(i for i in keep if lo <= i < hi):
        if runs and i == runs[-1][1] + 1: runs[-1][1] = i
        else: runs.append([i, i])
    while len(keep) < B and runs:
        grew = False
        for r in runs:
            for side in (-1, 1):
                if len(keep) >= B: break
                j = r[0] - 1 if side < 0 else r[1] + 1
                if lo <= j < hi and j not in keep:
                    keep.add(j); grew = True
                    if side < 0: r[0] = j
                    else: r[1] = j
        if not grew: break
    return sorted(keep)

def pat(v): return re.compile(r"(?<!\w)" + re.escape(v) + r"(?!\w)(?!\.\d)(?!,\d)")

def recent_48_policy(cache):
    L = int(cache.get_seq_length())
    return list(range(max(0, L - 48), L))

def recent_indices(L, budget=48):
    return list(range(L)) if L <= budget else list(range(L - budget, L))

def select_position_aware_indices(length):
    """Existing start-middle-recent policy: 16 start + 16 midpoint + 16 recent (budget = 48)."""
    if length <= 0:
        return []
    if length <= TOTAL_KEEP:
        return list(range(length))

    middle_start = (length - MIDDLE_KEEP) // 2
    indices = (
        list(range(START_KEEP))
        + list(range(middle_start, middle_start + MIDDLE_KEEP))
        + list(range(length - RECENT_KEEP, length))
    )
    assert len(indices) == TOTAL_KEEP and len(set(indices)) == TOTAL_KEEP and indices == sorted(indices)
    return indices

def snap_fn(B, w, q_ids):
    k = max(1, (B - 24) // w)
    def fn(c):
        L = c.layers[0].keys.shape[2]
        raw = select_snap_span_indices(c, model, q_ids, num_mid_spans=k, span_size=w)
        return _like(raw, pad_to_budget(raw, L, B))
    return fn

def snap_policy(cache):
    return select_snap_span_indices(
        cache, model, q_ids,
        start_keep=8,
        num_mid_spans=3,
        span_size=8,
        recent_keep=16,
        pool=5
    )

def snap_policy_adapter(cache):
    q_ids = tokenizer(
        "\nQuestion: " + test_question + "\nAnswer:",
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids.to("cuda")

    return select_snap_span_indices(
        cache,
        model,
        q_ids,
        start_keep=8,
        num_mid_spans=3,
        span_size=8,
        recent_keep=16,
        pool=5
    )

def streaming_48_policy(cache):
    L = int(cache.get_seq_length())
    if L <= 48:
        return list(range(L))
    return list(range(32)) + list(range(L - 16, L))

def streaming_indices(L, sink=4, budget=48):
    if L <= budget:
        return list(range(L))
    idx = list(range(sink)) + list(range(L - (budget - sink), L))
    assert len(idx) == budget == len(set(idx)) and idx == sorted(idx)
    return idx

def summarize(df, by):
    g = df.groupby(by).agg(n=("strict", "size"), strict=("strict", "sum"), exact=("exact", "sum"),
                           id_kept=("id_kept", "mean"), pos=("n_pos_pre", "mean"),
                           kv_mib=("kv_mib_pre", "mean"), sel_s=("selector_s", "mean"),
                           peak_mib=("peak_delta_mib", "mean"))
    g["lo"] = [wilson(a, b)[0] for a, b in zip(g.strict, g.n)]
    g["hi"] = [wilson(a, b)[1] for a, b in zip(g.strict, g.n)]
    return g.round(3)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return f"({c-h:.2f}, {c+h:.2f})"

def snap_tok_indices(cache, model, q_ids, B, start_keep=8, recent_keep=16, pool=5):
    L = cache.layers[0].keys.shape[2]
    if L <= B: return list(range(L))
    lo, hi = start_keep, L - recent_keep
    q, dev = q_ids.shape[1], q_ids.device
    work = copy.deepcopy(cache)
    with eager_attn(model):
        out = model(input_ids=q_ids, past_key_values=work, use_cache=True,
                    attention_mask=torch.ones(1, L + q, dtype=torch.long, device=dev),
                    position_ids=torch.arange(L, L + q, device=dev).unsqueeze(0),
                    output_attentions=True)
    n = len(out.attentions)
    sc = sum(a[0, :, :, lo:hi].float().sum(dim=(0, 1)) for a in out.attentions[n // 3:])
    sc = F.avg_pool1d(sc.view(1, 1, -1), kernel_size=pool, stride=1, padding=pool // 2).view(-1)
    top = sc.topk(B - start_keep - recent_keep).indices.add(lo).tolist()
    idx = sorted(set(range(start_keep)) | set(top) | set(range(L - recent_keep, L)))
    assert len(idx) == B
    return idx

def h2o_scores(ids):
    acc = {"s": None}
    def hook(mod, inp, out):
        w = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
        if w is None: return
        s = w[0].sum(dim=(0, 1), dtype=torch.float32)
        acc["s"] = s if acc["s"] is None else acc["s"] + s
    hs = [l.self_attn.register_forward_hook(hook) for l in model.model.layers]
    try:
        with eager_attn(model):
            model(input_ids=ids, use_cache=False)
    finally:
        for h in hs: h.remove()
    if acc["s"] is None: raise RuntimeError("no attention weights")
    return acc["s"]

def h2o_indices(c, scores, B, recent_keep=16):
    L = c.layers[0].keys.shape[2]
    assert len(scores) == L, (len(scores), L)
    if L <= B: return list(range(L))
    top = scores[: L - recent_keep].topk(B - recent_keep).indices.tolist()
    idx = sorted(set(top) | set(range(L - recent_keep, L)))
    assert len(idx) == B
    return idx

def h2o_fn(B, ctx):
    def fn(c):
        if ctx not in _h2o_cache:
            ids = tokenizer(ctx, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            _h2o_cache.clear(); _h2o_cache[ctx] = h2o_scores(ids)
        return h2o_indices(c, _h2o_cache[ctx], B)
    return fn

def tok_fn(B, q_ids):          # SnapKV-style: same question-conditioned scores, single tokens
    def fn(c):
        L = c.layers[0].keys.shape[2]
        raw = select_snap_span_indices(c, model, q_ids, num_mid_spans=B - 24, span_size=1)
        return _like(raw, pad_to_budget(raw, L, B))
    return fn

def w(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)); h /= d
    return f"({c-h:.2f}, {c+h:.2f})"

def step(cache, x, pos0):
    r = cache.layers[0].keys.shape[2]
    return model(input_ids=x, past_key_values=cache, use_cache=True,
                 attention_mask=torch.ones(1, r + x.shape[1], dtype=torch.long, device=device),
                 position_ids=torch.arange(pos0, pos0 + x.shape[1], device=device).unsqueeze(0))
