"""Calibration report for per-example results written by djev.evaluate / djev.baseline_ar.

For every readout: accuracy, mean probability of the gold answer (RLCD Table 5), ECE,
NLL, Brier, and the same after temperature scaling fitted on the first half of the file
(even rows fit the temperature, odd rows are reported).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter


def records(path: str):
    """-> {readout: [(probs dict, gold)]} for single- or multi-question rows."""
    out: dict[str, list] = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        for qid, d in r["decisions"].items():
            gold = str(r["gold"][qid]).lower()
            by = d.get("probs_by_readout") or ({"final": d["probs"]} if d.get("probs") else {})
            for name, p in by.items():
                out.setdefault(name, []).append((p, gold))
    return out


def scale(p: dict[str, float], T: float) -> dict[str, float]:
    logs = {k: math.log(max(v, 1e-12)) / T for k, v in p.items()}
    m = max(logs.values())
    e = {k: math.exp(v - m) for k, v in logs.items()}
    z = sum(e.values())
    return {k: v / z for k, v in e.items()}


def metrics(rows, bins: int = 10) -> dict:
    n = len(rows)
    acc = nll = brier = pg = 0.0
    hist = [[0, 0.0, 0.0] for _ in range(bins)]
    preds = Counter()
    for p, g in rows:
        pred = max(p, key=p.get)
        conf = p[pred]
        preds[pred] += 1
        acc += pred == g
        pg += p.get(g, 0.0)
        nll -= math.log(max(p.get(g, 0.0), 1e-12))
        brier += sum((v - (k == g)) ** 2 for k, v in p.items())
        b = min(int(conf * bins), bins - 1)
        hist[b][0] += 1
        hist[b][1] += conf
        hist[b][2] += pred == g
    ece = sum(abs(c / k - a / k) * k for k, c, a in hist if k) / n
    return {"n": n, "accuracy": acc / n, "avg_prob_gold": pg / n, "ECE": ece, "NLL": nll / n,
            "Brier": brier / n, "pred_dist": dict(preds)}


def fit_temperature(rows) -> float:
    grid = [0.5 * 1.1 ** i for i in range(120)]  # 0.5 .. ~4.6e4
    return min(grid, key=lambda T: sum(-math.log(max(scale(p, T).get(g, 0.0), 1e-12)) for p, g in rows))


def by_source(path: str) -> dict:
    """Per-source accuracy (all questions) and exact-match rate (every question of a row right)."""
    out: dict[str, list] = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        ok = [str(d.get("answer")).lower() == str(r["gold"][q]).lower() for q, d in r["decisions"].items()]
        s = out.setdefault(r.get("source") or "all", [0, 0, 0, 0])
        s[0] += sum(ok); s[1] += len(ok); s[2] += all(ok); s[3] += 1
    return {k: {"accuracy": round(v[0] / v[1], 4), "exact_match": round(v[2] / v[3], 4), "rows": v[3]} for k, v in sorted(out.items())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    for f in a.files:
        print(json.dumps({"file": f, "by_source": by_source(f)}, ensure_ascii=False))
        for name, rows in records(f).items():
            val, test = rows[0::2], rows[1::2]  # interleaved, so mixed-source files split evenly
            T = fit_temperature(val)
            print(json.dumps({
                "file": f, "readout": name,
                "all": metrics(rows),
                "test_raw": metrics(test),
                "temperature": round(T, 3),
                "test_scaled": metrics([(scale(p, T), g) for p, g in test]),
            }, ensure_ascii=False))


if __name__ == "__main__":
    main()
