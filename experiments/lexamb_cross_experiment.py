#!/usr/bin/env python3
"""Crossing experiment: our synthetic-continuum protocol applied to the
NATURAL stimuli released by Pouw et al. (their re-recording of the
Gaskell & Marslen-Wilson 2001 lexical-ambiguity materials, Google-Drive
archive lexamb-stimuli.zip linked from github.com/CharlottePouw/
assimilation).  The original Darcy et al. (2009) stimA recordings are
not public; these are the natural deliberate-assimilation materials the
authors themselves released.

Archive structure (decoded from Pouw et al. Appendix C + our CTC
alignment): 43 numbered items x 2 recorded passes = 86 sentences, all
in the VIABLE variant with the full biasing context sentence (e.g.
"It's best to start the day with a burst of activity.  I think a quick
rum picks you up.").  The two passes (_1/_2) are two takes of the same
speaker: their median f0 and LPC spectral envelopes are statistically
indistinguishable (f0 ~165 Hz both; F1 ratio 0.996); Pouw et al.'s male
replication speaker is not part of the shared archive.  38 items are
the set analyzed by Pouw et al. (Appendix C, N=38); 5 items (18, 19,
20, 42, 43) are the N->[N]-surface items they EXCLUDED because their
CTC measure cannot score velar nasals (wav2vec2 writes [N] as "n"+"g");
our {n, m, N} probe measure can.

Measures:
  A. CTC transcription compensation (wav2vec2-base-960h, as Pouw):
     verdict = nearest of {underlying, surface} spelling (Levenshtein)
     at the aligned target-word position.
  B. Probe verdicts (BOTH models, 13 layers): the same logistic probes
     trained on the 18 canonical synthetic training sentences
     (T_n/T_m/T_N, work/features/<tag>/) as every previous experiment
     here, applied to the window mean of hidden states over the
     CTC-aligned frames of the assimilated consonant (the last M/N run
     for n_m items; the last N run extended through an adjacent G run
     for n_ng items).  Only nasal items are probeable.

Stages:  a = alignment+features, b = probes, c = stats.
Outputs -> <KIT_ROOT>/work/lexamb/   (set COVTL_KIT_ROOT to relocate;
SRC defaults to work/lexamb/wavs, populated by repro/fetch_lexamb.py;
LEXAMB_SRC overrides the stimuli location)
"""
import json
import math
import os
import sys
import wave

import numpy as np

# --- Kit portability: root of the experiment kit (parent of repro/) ------
KIT_ROOT = os.environ.get(
    "COVTL_KIT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..")))
os.environ.setdefault("HF_HOME", os.path.join(KIT_ROOT, "hf_cache"))

SRC = os.environ.get("LEXAMB_SRC", os.path.join(KIT_ROOT, "work", "lexamb", "wavs"))
OUT = os.path.join(KIT_ROOT, "work", "lexamb")
FEAT = os.path.join(KIT_ROOT, "work", "features")   # synthetic training feats
MODEL_TAGS = ["w2v2_base960h", "w2v2_base"]
MODEL_IDS = {
    "w2v2_base960h": "facebook/wav2vec2-base-960h",
    "w2v2_base": "facebook/wav2vec2-base",
}
TRAIN_CLASSES = ["n", "m", "N"]
LAYER_NAMES = ["CNN"] + [f"T{i}" for i in range(1, 13)]
Z = 1.959963985

# zip number -> item metadata.
# grp: "an" = analyzed by Pouw (Appendix C), "ng" = N->[N] exploratory
# typ: n_m, d_b, d_g, t_p, t_k, n_ng
# und/surf: underlying / surface spelling; variants: extra observed CTC
# spellings of the target word; follower: viable-variant follower words.
ITEMS = {
    "01": dict(grp="an", typ="d_b", und="cod",  surf="cob",
               variants=["cub"], follower=["poached"], app=1),
    "03": dict(grp="an", typ="d_b", und="bride", surf="bribe",
               variants=[], follower=["made"], app=2),
    "04": dict(grp="an", typ="d_g", und="led",  surf="leg",
               variants=[], follower=["covered"], app=3),
    "05": dict(grp="an", typ="d_g", und="fad",  surf="fag",
               variants=[], follower=["causes"], app=4),
    "06": dict(grp="an", typ="d_g", und="thud", surf="thug",
               variants=[], follower=["caught"], app=5),
    "07": dict(grp="an", typ="d_g", und="mud",  surf="mug",
               variants=[], follower=["cracked"], app=6),
    "08": dict(grp="an", typ="d_g", und="road", surf="rogue",
               variants=[], follower=["cut"], app=7),
    "09": dict(grp="an", typ="n_m", und="phone", surf="foam",
               variants=["poam"], follower=["packaging"], app=8),
    "10": dict(grp="an", typ="n_m", und="scene", surf="seam",
               variants=[], follower=["beautifully"], app=9),
    "11": dict(grp="an", typ="n_m", und="bean", surf="beam",
               variants=[], follower=["promptly"], app=10),
    "12": dict(grp="an", typ="n_m", und="worn", surf="warm",
               variants=[], follower=["blanket"], app=11),
    "13": dict(grp="an", typ="n_m", und="run", surf="rum",
               variants=[], follower=["picks"], app=12),
    "14": dict(grp="an", typ="n_m", und="cone", surf="comb",
               variants=[], follower=["pretty"], app=13),
    "15": dict(grp="an", typ="n_m", und="turn", surf="term",
               variants=[], follower=["passed"], app=14),
    "16": dict(grp="an", typ="n_m", und="dane", surf="dame",
               variants=[], follower=["planned"], app=15),
    "18": dict(grp="ng", typ="n_ng", und="ban", surf="bang",
               variants=["bancost"], follower=["cost", "caused",
                                               "havoc"], app=None),
    "19": dict(grp="ng", typ="n_ng", und="ran", surf="rang",
               variants=["sandrorene"], follower=["carefully"], app=None),
    "20": dict(grp="ng", typ="n_ng", und="bun", surf="bung",
               variants=["bone"], follower=["costs"], app=None),
    "21": dict(grp="an", typ="t_p", und="cat", surf="cap",
               variants=[], follower=["by"], app=16),
    "22": dict(grp="an", typ="t_p", und="heat", surf="heap",
               variants=[], follower=["bowls"], app=17),
    "23": dict(grp="an", typ="t_p", und="mat", surf="map",
               variants=[], follower=["back"], app=18),
    "24": dict(grp="an", typ="t_p", und="sit", surf="sip",
               variants=[], follower=["meekly"], app=19),
    "25": dict(grp="an", typ="t_p", und="wit", surf="whip",
               variants=[], follower=["brilliantly"], app=20),
    "26": dict(grp="an", typ="t_p", und="grit", surf="grip",
               variants=[], follower=["motor", "motorways"], app=21),
    "27": dict(grp="an", typ="t_p", und="sleet", surf="sleep",
               variants=[], follower=["brought"], app=22),
    "28": dict(grp="an", typ="t_k", und="port", surf="pork",
               variants=[], follower=["guzzled"], app=23),
    "29": dict(grp="an", typ="t_k", und="bait", surf="bake",
               variants=["bak"], follower=["got"], app=24),
    "30": dict(grp="an", typ="t_k", und="late", surf="lake",
               variants=[], follower=["growth"], app=25),
    "31": dict(grp="an", typ="t_k", und="dart", surf="dark",
               variants=[], follower=["gradually"], app=26),
    "32": dict(grp="an", typ="t_k", und="bite", surf="bike",
               variants=[], follower=["guiltily"], app=27),
    "33": dict(grp="an", typ="t_k", und="net", surf="neck",
               variants=[], follower=["got"], app=28),
    "34": dict(grp="an", typ="t_k", und="wait", surf="wake",
               variants=[], follower=["grumpily"], app=29),
    "35": dict(grp="an", typ="t_k", und="rat", surf="rack",
               variants=[], follower=["grimy"], app=30),
    "36": dict(grp="an", typ="n_m", und="line", surf="lime",
               variants=[], follower=["print"], app=31),
    "37": dict(grp="an", typ="t_p", und="right", surf="ripe",
               variants=[], follower=["berries"], app=32),
    "38": dict(grp="an", typ="t_k", und="fort", surf="fork",
               variants=[], follower=["go"], app=33),
    "39": dict(grp="an", typ="d_g", und="bud", surf="bug",
               variants=[], follower=["curled"], app=34),
    "40": dict(grp="an", typ="d_g", und="lead", surf="league",
               variants=[], follower=["cricket"], app=35),
    "41": dict(grp="an", typ="n_m", und="gun", surf="gum",
               variants=[], follower=["peddlers"], app=36),
    "42": dict(grp="ng", typ="n_ng", und="pan", surf="pang",
               variants=["pen"], follower=["glistened"], app=None),
    "43": dict(grp="ng", typ="n_ng", und="fan", surf="fang",
               variants=["fanga", "fangavot"], follower=["off",
                                                        "goes"], app=None),
    "44": dict(grp="an", typ="t_k", und="beat", surf="beak",
               variants=[], follower=["goes"], app=37),
    "45": dict(grp="an", typ="t_k", und="oat", surf="oak",
               variants=[], follower=["gives"], app=38),
}

# chars whose CTC runs represent the assimilated consonant, per type
COMP_CHAR = {"n_m": "N", "d_b": "D", "d_g": "D", "t_p": "T", "t_k": "T",
             "n_ng": "N"}
SURF_CHAR = {"n_m": "M", "d_b": "B", "d_g": "G", "t_p": "P", "t_k": "K",
             "n_ng": "G"}       # n_ng CTC verdict exploratory by design


def load_wav_16k(path):
    with wave.open(path, "rb") as w:
        sr, n, ch, sw = (w.getframerate(), w.getnframes(),
                         w.getnchannels(), w.getsampwidth())
        raw = w.readframes(n)
    assert ch == 1 and sw == 2, (path, ch, sw)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sr != 16000:
        g = math.gcd(sr, 16000)
        from scipy.signal import resample_poly
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


def norm(y):
    return (y - y.mean()) / (y.std() + 1e-7)


def median_f0(y, sr=16000):
    """Autocorrelation f0 over energetic 40 ms frames (70-300 Hz)."""
    fl, hop = int(0.04 * sr), int(0.01 * sr)
    f0s = []
    for i0 in range(0, len(y) - fl, hop):
        seg = y[i0:i0 + fl]
        if np.sqrt((seg ** 2).mean()) < 0.01:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[fl - 1:]
        lo, hi = int(sr / 300), int(sr / 70)
        lag = int(np.argmax(ac[lo:hi])) + lo
        if ac[lag] > 0.4 * ac[0]:
            f0s.append(sr / lag)
    return float(np.median(f0s)) if f0s else float("nan")


def levenshtein(a, b):
    m = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    m[:, 0] = np.arange(len(a) + 1)
    m[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            m[i, j] = min(m[i - 1, j] + 1, m[i, j - 1] + 1,
                          m[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return int(m[-1, -1])


def greedy_units(logits, processor):
    """per-frame argmax -> collapsed [(char, f_start, f_end)] + words."""
    ids = logits.argmax(-1)[0].tolist()
    vocab = processor.tokenizer.get_vocab()
    id2c = {v: k for k, v in vocab.items()}
    units = []
    prev = None
    for f, i in enumerate(ids):
        c = id2c.get(i, "")
        if c in ("<pad>", ""):
            prev = None
            continue
        if c == prev:
            units[-1][2] = f
            continue
        units.append([c, f, f])
        prev = c
    words, cur = [], []
    for c, f0, f1 in units:
        if c == "|":
            if cur:
                words.append(cur)
            cur = []
        else:
            cur.append((c, f0, f1))
    if cur:
        words.append(cur)
    return units, words


def word_str(w):
    return "".join(c for c, _, _ in w).lower()


def locate(words, meta):
    """Find the target word.  Returns (wi, how) or (None, 'notfound')."""
    cands = [c.lower() for c in [meta["surf"], meta["und"]] +
             meta["variants"] if c]
    best = None
    for wi, w in enumerate(words):
        s = word_str(w)
        if s in cands and len(s) >= 3:
            best = (wi, "spelling")
    if best:
        return best
    for fol in meta["follower"]:
        fol = fol.lower()
        for wi, w in enumerate(words):
            if word_str(w) == fol and wi >= 1:
                best = (wi - 1, "follower")
    return best if best else (None, "notfound")


def cons_window(word_units, typ):
    """Frames of the assimilated-consonant run(s) inside the target word.
    n_m : last run whose char is M or N.
    n_ng: last N run, extended through an immediately adjacent G run.
    stops: last run whose char is the comp or surf char."""
    comp, surf = COMP_CHAR[typ], SURF_CHAR[typ]
    if typ == "n_m":
        wanted = ("M", "N")
    elif typ == "n_ng":
        wanted = ("N",)
    else:
        wanted = (comp, surf)
    idx = [i for i, (c, _, _) in enumerate(word_units)
           if c.upper() in wanted]
    if not idx:
        return None
    i0 = idx[-1]
    c, f0, f1 = word_units[i0]
    if typ == "n_ng" and i0 + 1 < len(word_units) and \
            word_units[i0 + 1][0].upper() == "G":
        f1 = word_units[i0 + 1][2]
        chars = c + word_units[i0 + 1][0]
    else:
        chars = c
    return chars, f0, f1


# ---------------------------------------------------------------- stage A
def stage_a():
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Processor

    os.makedirs(f"{OUT}/feats", exist_ok=True)
    proc = Wav2Vec2Processor.from_pretrained(MODEL_IDS["w2v2_base960h"])
    asr = Wav2Vec2ForCTC.from_pretrained(MODEL_IDS["w2v2_base960h"])
    asr.eval()
    ssl = Wav2Vec2Model.from_pretrained(MODEL_IDS["w2v2_base"])
    ssl.eval()

    align = {}
    for num, meta in sorted(ITEMS.items()):
        for spk in ("1", "2"):
            stem = f"{num}_{spk}"
            path = os.path.join(SRC, stem + ".wav")
            if not os.path.exists(path):
                print(f"!! {stem}: file absent")
                continue
            y16 = load_wav_16k(path)
            f0 = median_f0(y16)
            y = norm(y16)
            x = torch.from_numpy(y).unsqueeze(0)
            with torch.no_grad():
                out = asr(x, output_hidden_states=True)
                words_txt = proc.batch_decode(out.logits.argmax(-1))[0]
                units, words = greedy_units(out.logits, proc)
                hs96 = np.stack(
                    [h[0].numpy() for h in out.hidden_states], 0)
                out2 = ssl(x, output_hidden_states=True)
                hsB = np.stack(
                    [h[0].numpy() for h in out2.hidden_states], 0)

            wi, how = locate(words, meta)
            rec = dict(stem=stem, item=num, spk=spk, f0=round(f0, 1),
                       text=words_txt.strip(), how=how,
                       type=meta["typ"], grp=meta["grp"],
                       und=meta["und"], surf=meta["surf"])
            if wi is None:
                rec["flag"] = "notfound"
                align[stem] = rec
                print(f"!! {stem}: target NOT FOUND  ({words_txt.strip()})")
                continue
            w = words[wi]
            ws = word_str(w)
            cw = cons_window(w, meta["typ"])
            nxt = word_str(words[wi + 1]) if wi + 1 < len(words) else ""
            rec.update(target_word=ws, target_idx=wi, next_word=nxt,
                       word_chars=[[c, a, b] for c, a, b in w])
            if cw is None:
                rec["flag"] = "no_consonant_run"
                align[stem] = rec
                print(f"!! {stem}: no consonant run in '{ws}'")
                continue
            chars, f0c, f1c = cw
            rec.update(cons_chars=chars, f_start=f0c, f_end=f1c)
            # CTC verdict: nearest of underlying / surface spelling
            du = levenshtein(ws, meta["und"])
            ds = levenshtein(ws, meta["surf"])
            rec.update(dist_und=du, dist_surf=ds)
            if du < ds:
                rec["ctc_comp"] = True
            elif ds < du:
                rec["ctc_comp"] = False
            else:
                rec["ctc_comp"] = None
                rec["flag"] = "spelling_tie"
            # window means at several frame margins around the run
            rec.update(win_frames=[f0c, f1c])
            T = hs96.shape[1]
            for margin in (0, 1, 2):
                a0 = max(0, f0c - margin)
                a1 = min(T - 1, f1c + margin)
                rec[f"win_m{margin}"] = [a0, a1]
            for tag, hs in (("w2v2_base960h", hs96),
                            ("w2v2_base", hsB)):
                np.savez_compressed(
                    f"{OUT}/feats/{stem}.npz",
                    **{f"wm{m}": hs[:, max(0, f0c - m):
                                min(T - 1, f1c + m) + 1, :]
                       .mean(axis=1).astype(np.float32)
                       for m in (0, 1, 2)},
                    frames=np.array([f0c, f1c]),
                    n_frames=T)
            align[stem] = rec
            print(f"{stem}: f0={f0:5.0f} [{how:8s}] {ws:11s}"
                  f" cons={chars:2s} f={f0c:3d}-{f1c:3d} next={nxt:12s}"
                  f" d(und/surf)={du}/{ds} comp={rec['ctc_comp']}")
    with open(f"{OUT}/cross_alignment.json", "w") as f:
        json.dump(align, f, indent=1)
    print("stage A done ->", f"{OUT}/cross_alignment.json")


# ---------------------------------------------------------------- stage B
def stage_b():
    from sklearn.linear_model import LogisticRegression

    with open(f"{OUT}/cross_alignment.json") as f:
        align = json.load(f)

    results = {}
    for tag in MODEL_TAGS:
        Xtr, ytr = [], []
        for ci, cls in enumerate(TRAIN_CLASSES):
            for i in range(1, 7):
                d = np.load(f"{FEAT}/{tag}/T_{cls}{i}.npz",
                            allow_pickle=True)
                win = json.loads(str(d["windows"]))["target"]
                m = ((d["t_centers"] >= win[0]) &
                     (d["t_centers"] <= win[1]))
                Xtr.append(d["hs"][:, m, :].mean(axis=1))
                ytr.append(ci)
        Xtr = np.stack(Xtr, 0)                    # (18, 13, 768)
        ytr = np.array(ytr)
        res = {"train_acc": {}, "stimuli": {}}
        for L in range(13):
            clf = LogisticRegression(max_iter=5000, C=1.0)
            clf.fit(Xtr[:, L, :], ytr)
            lname = LAYER_NAMES[L]
            res["train_acc"][lname] = float(
                (clf.predict(Xtr[:, L, :]) == ytr).mean())
            for stem, rec in align.items():
                # probe every located nasal item (a spelling_tie flag
                # only concerns the CTC verdict, not the probe window)
                if "target_word" not in rec or \
                        rec["type"] not in ("n_m", "n_ng"):
                    continue
                fp = f"{OUT}/feats/{stem}.npz"
                if not os.path.exists(fp):
                    continue
                d = np.load(fp)
                p = clf.predict_proba(d["wm2"][L][None])[0]
                pred = TRAIN_CLASSES[int(np.argmax(p))]
                entry = res["stimuli"].setdefault(stem, {})
                entry[lname] = dict(
                    pred=pred,
                    proba={c: float(p[i]) for i, c in
                           enumerate(TRAIN_CLASSES)})
                # sensitivity: margins 0 and 1 (verdict only)
                for m in (0, 1):
                    pm = clf.predict_proba(d[f"wm{m}"][L][None])[0]
                    entry.setdefault(f"m{m}", {})[lname] = \
                        TRAIN_CLASSES[int(np.argmax(pm))]
        results[tag] = res
        print(f"[{tag}] train_acc:",
              {k: round(v, 2) for k, v in res["train_acc"].items()})
    with open(f"{OUT}/probe_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("stage B done ->", f"{OUT}/probe_results.json")


# ---------------------------------------------------------------- stage C
def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = (Z / den) * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def stage_c():
    with open(f"{OUT}/cross_alignment.json") as f:
        align = json.load(f)
    with open(f"{OUT}/probe_results.json") as f:
        probes = json.load(f)

    ok = {k: v for k, v in align.items() if "flag" not in v}
    # items with a probe window (located, nasal): flag 'spelling_tie'
    # only invalidates the CTC verdict, not the probe measure
    nasal = {k: v for k, v in align.items()
             if "target_word" in v and v["type"] in ("n_m", "n_ng")}
    stats = {"n_files": len(align), "n_scored": len(ok), "groups": {},
             "f0": {}, "ctc_detail": {}}

    # --- CTC transcription measure (960h) --------------------------------
    an = [r for r in ok.values() if r["grp"] == "an"]
    for name, rows in (("pool", an),
                       ("pass1", [r for r in an if r["spk"] == "1"]),
                       ("pass2", [r for r in an if r["spk"] == "2"]),
                       ("n_m", [r for r in an if r["type"] == "n_m"]),
                       ("stops", [r for r in an if r["type"] != "n_m"])):
        ks = [r for r in rows if r["ctc_comp"] is not None]
        k = sum(1 for r in ks if r["ctc_comp"])
        n = len(ks)
        lo, hi = wilson(k, n)
        stats["groups"][f"ctc_{name}"] = dict(
            k=k, n=n, rate=round(k / n, 4) if n else None,
            ci=[round(lo, 4), round(hi, 4)])
    # exploratory n_ng CTC (unreliable by design)
    ng = [r for r in ok.values() if r["grp"] == "ng"]
    ks = [r for r in ng if r["ctc_comp"] is not None]
    k = sum(1 for r in ks if r["ctc_comp"])
    lo, hi = wilson(k, len(ks))
    stats["groups"]["ctc_ng_exploratory"] = dict(
        k=k, n=len(ks), rate=round(k / len(ks), 4) if ks else None,
        ci=[round(lo, 4), round(hi, 4)])
    # per-type detail
    for r in ok.values():
        stats["ctc_detail"][r["stem"]] = dict(
            word=r["target_word"], comp=r["ctc_comp"],
            d_und=r.get("dist_und"), d_surf=r.get("dist_surf"))

    # --- probe measure ----------------------------------------------------
    MARGIN = "wm2"          # primary: +/-2 frames (~100 ms window)
    for tag in MODEL_TAGS:
        st = probes[tag]["stimuli"]
        for name, sel in (
                ("n_m_an", lambda r: r["grp"] == "an" and
                    r["type"] == "n_m"),
                ("n_ng_expl", lambda r: r["grp"] == "ng"),
                ("n_m_pass1", lambda r: r["grp"] == "an" and
                    r["type"] == "n_m" and r["spk"] == "1"),
                ("n_m_pass2", lambda r: r["grp"] == "an" and
                    r["type"] == "n_m" and r["spk"] == "2")):
            rows = [r for r in nasal.values() if sel(r) and r["stem"] in st]
            if not rows:
                continue
            g = {"n": len(rows), "layers": {}}
            for L in LAYER_NAMES:
                k = sum(1 for r in rows
                        if st[r["stem"]][L]["pred"] == "n")
                lo, hi = wilson(k, len(rows))
                g["layers"][L] = dict(
                    k=k, rate=round(k / len(rows), 4),
                    ci=[round(lo, 4), round(hi, 4)],
                    mean_pn=round(float(np.mean(
                        [st[r["stem"]][L]["proba"]["n"]
                         for r in rows])), 4),
                    mean_pN=round(float(np.mean(
                        [st[r["stem"]][L]["proba"]["N"]
                         for r in rows])), 4),
                    mean_pm=round(float(np.mean(
                        [st[r["stem"]][L]["proba"]["m"]
                         for r in rows])), 4))
            stats["groups"][f"probe_{tag}_{name}"] = g

        # window-margin sensitivity at the final layer (T12)
        sens = {}
        for name, sel in (("n_m_an", lambda r: r["grp"] == "an" and
                           r["type"] == "n_m"),
                          ("n_ng_expl", lambda r: r["grp"] == "ng")):
            rows = [r for r in nasal.values() if sel(r) and r["stem"] in st]
            if not rows:
                continue
            for key, lay in (("m0", "T12"), ("m1", "T12"), ("m2", "T12")):
                if key == "m2":
                    continue
                k = sum(1 for r in rows
                        if st[r["stem"]].get(key, {}).get(lay) == "n")
                sens[f"{name}_{key}"] = f"{k}/{len(rows)}"
        stats["sensitivity"] = stats.get("sensitivity", {})
        stats["sensitivity"][tag] = sens

    # --- f0 summary --------------------------------------------------------
    for spk in ("1", "2"):
        v = [r["f0"] for r in align.values()
             if r["spk"] == spk and r["f0"] == r["f0"]]
        stats["f0"][spk] = dict(median=round(float(np.median(v)), 1),
                                n=len(v))

    with open(f"{OUT}/cross_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats["groups"], indent=1)[:3000])
    print("stage C done ->", f"{OUT}/cross_stats.json")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("all", "a"):
        stage_a()
    if stage in ("all", "b"):
        stage_b()
    if stage in ("all", "c"):
        stage_c()
