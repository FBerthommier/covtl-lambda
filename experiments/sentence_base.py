#!/usr/bin/env python3
"""Sentence base for the wav2vec2 probing study (English, engine SAMPA).

Design: place assimilation in English, TIMIT-style two-word sentences.
- Pairs (viable): canonical vs assimilated, same phoneme count => the two
  variants are frame-aligned 1:1 (causal intervention on one phoneme).
- Non-viable controls: same manipulation but the context does not license
  the assimilation (following consonant is alveolar).
- Training set: word-final nasal/stop + varied following consonant
  (pan / tan / kan) => 6 tokens per class, balanced over contexts.

Naming: 'N' = velar nasal (patched closure), 'j'/'w' = glides used by the
engine to spell diphthongs (right = r-a-j-t is avoided: no /r/ sentences,
the engine maps spelled r to the French uvular R).
"""

# ---------------------------------------------------------------------------
# Viable assimilation pairs
# ---------------------------------------------------------------------------
# target_idx: index of the manipulated phoneme in the key sequence
PAIRS = [
    dict(pid="P1", sentence="clean pan",
         canon="klin.pan", assim="klim.pan",
         target="n", surface="m", process="n->m", context="labial (before p)"),
    dict(pid="P2", sentence="mean boat",
         canon="min.bot", assim="mim.bot",
         target="n", surface="m", process="n->m", context="labial (before b)"),
    dict(pid="P3", sentence="clean can",
         canon="klin.kan", assim="kliN.kan",
         target="n", surface="N", process="n->N", context="velar (before k)"),
    dict(pid="P4", sentence="thin packet",
         canon="Tin.pakit", assim="Tim.pakit",
         target="n", surface="m", process="n->m", context="labial (before p)"),
    dict(pid="P5", sentence="that pen",
         canon="Dat.pen", assim="Dap.pen",
         target="t", surface="p", process="t->p", context="labial (before p)"),
    dict(pid="P6", sentence="that cake",
         canon="Dat.kek", assim="Dak.kek",
         target="t", surface="k", process="t->k", context="velar (before k)"),
]

# ---------------------------------------------------------------------------
# Non-viable controls (following consonant does not license the labial
# assimilation -- alveolar or velar; the 'unlic' variant applies the same
# phoneme substitution)
# ---------------------------------------------------------------------------
NONVIABLE = [
    dict(pid="NV1", sentence="clean tan",
         canon="klin.tan", unlic="klim.tan",
         target="n", surface="m", process="n->m", context="alveolar (before t)"),
    dict(pid="NV2", sentence="that time",
         canon="Dat.tajm", unlic="Dap.tajm",
         target="t", surface="p", process="t->p", context="alveolar (before t)"),
    dict(pid="NV3", sentence="clean can",
         canon="klin.kan", unlic="klim.kan",
         target="n", surface="m", process="n->m", context="velar (before k)"),
    dict(pid="NV4", sentence="mean goat",
         canon="min.got", unlic="mim.got",
         target="n", surface="m", process="n->m", context="velar (before g)"),
]

# ---------------------------------------------------------------------------
# Probe training set (canonical only): word-final target + varied carrier.
# 6 tokens per class, following consonant balanced 2x{p, t, k}.
# ---------------------------------------------------------------------------
TRAIN = []
_wordlists = {
    # class: [(word_sampa, word_orth), ...] 6 each
    "n":  [("fin", "fin"), ("win", "win"), ("kin", "keen"),
           ("bin", "bean"), ("sin", "seen"), ("tin", "teen")],
    "m":  [("sim", "seem"), ("tim", "team"), ("dim", "deem"),
           ("gem", "game"), ("sem", "same"), ("nem", "name")],
    "N":  [("kiN", "king"), ("siN", "sing"), ("lON", "long"),
           ("baN", "bang"), ("sON", "song"), ("haN", "hang")],
    "t":  [("put", "put"), ("sat", "sat"), ("fat", "fat"),
           ("mEt", "met"), ("bit", "bit"), ("hat", "hot")],
    "p":  [("kip", "keep"), ("dip", "deep"), ("slip", "sleep"),
           ("tSip", "cheap"), ("hap", "hop"), ("tap", "top")],
    "k":  [("kik", "kick"), ("bak", "back"), ("sik", "sick"),
           ("buk", "book"), ("luk", "look"), ("tuk", "took")],
}
_carriers = [("pan", "pan"), ("tan", "tan"), ("kan", "can")]
# place labels for the probes
PLACE = {"n": "alveolar", "m": "labial", "N": "velar",
         "t": "alveolar", "p": "labial", "k": "velar"}
MANNER = {"n": "nasal", "m": "nasal", "N": "nasal",
          "t": "stop", "p": "stop", "k": "stop"}

for cls, words in _wordlists.items():
    for i, (w, orth) in enumerate(words):
        car, car_orth = _carriers[i % 3]
        TRAIN.append(dict(
            tid=f"T_{cls}{i+1}",
            sentence=f"{orth} {car_orth}",
            sampa=f"{w}.{car}",
            target=cls, place=PLACE[cls], manner=MANNER[cls],
        ))


def all_items():
    """Yield every synthesis item: (label, sampa, group, meta)."""
    out = []
    for p in PAIRS:
        tidx = _target_index(p["canon"], p["target"])
        out.append((f"{p['pid']}_canon", p["canon"], "pair_canon", dict(p, variant="canonical", target_idx=tidx)))
        out.append((f"{p['pid']}_assim", p["assim"], "pair_assim", dict(p, variant="assimilated", target_idx=tidx)))
    for p in NONVIABLE:
        tidx = _target_index(p["canon"], p["target"])
        out.append((f"{p['pid']}_canon", p["canon"], "nv_canon", dict(p, variant="canonical", target_idx=tidx)))
        out.append((f"{p['pid']}_unlic", p["unlic"], "nv_unlic", dict(p, variant="unlicensed", target_idx=tidx)))
    for t in TRAIN:
        tidx = _target_index(t["sampa"], t["target"])
        out.append((t["tid"], t["sampa"], "train", dict(t, variant="train", target_idx=tidx)))
    return out


def _target_index(sampa, target):
    from vtl_synth.core.phonemes import greedy_sampa_split
    keys = []
    for tok in sampa.replace(".", " ").split():
        keys.extend(greedy_sampa_split(tok))
    # target = last key of the first word
    first_word = sampa.split(".")[0]
    fw_keys = greedy_sampa_split(first_word)
    assert fw_keys[-1] == target, (sampa, target, fw_keys)
    idx = len(fw_keys) - 1
    assert keys[idx] == target
    return idx


if __name__ == "__main__":
    from vtl_synth.core.phonemes import greedy_sampa_split
    bad = 0
    for label, sampa, group, meta in all_items():
        try:
            keys = []
            for tok in sampa.replace(".", " ").split():
                k = greedy_sampa_split(tok)
                assert k, f"unparseable token {tok!r} in {sampa!r}"
                keys.extend(k)
            idx = meta["target_idx"]
            print(f"{label:14s} {sampa:14s} -> {len(keys)} keys, target[{idx}]={keys[idx]}  {meta.get('sentence','')}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {label} {sampa}: {e}")
    print(f"\n{len(all_items())} items, {bad} failures")
    # check pair length matching (canonical vs assim same key count)
    for p in PAIRS + NONVIABLE:
        c = greedy_sampa_split(p["canon"].replace(".", " ").replace("  ", " "))
        a = greedy_sampa_split(p["assim" if "assim" in p else "unlic"].replace(".", " ").replace("  ", " "))
        # note: greedy on the whole string with the dot removed may merge across
        # the boundary - redo per token
        c = sum([greedy_sampa_split(t) for t in p["canon"].split(".")], [])
        a = sum([greedy_sampa_split(t) for t in p["assim" if "assim" in p else "unlic"].split(".")], [])
        assert len(c) == len(a), (p, len(c), len(a))
        diff = [i for i, (x, y) in enumerate(zip(c, a)) if x != y]
        print(f"{p['pid']}: {len(c)} keys, substitution at index {diff} -> {[c[i] for i in diff]}->{[a[i] for i in diff]}")
