"""Inspect MTSD annotations. Runs on annotations only -- no images needed."""
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def load_split(root, name):
    for c in [root / "splits" / f"{name}.txt", root / f"{name}.txt"]:
        if c.exists():
            return [ln.strip() for ln in c.read_text().splitlines() if ln.strip()]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args()

    ann_dir = args.root / "annotations"
    if not ann_dir.exists():
        raise SystemExit(f"No annotations/ under {args.root}")
    args.out.mkdir(parents=True, exist_ok=True)

    for s in ("train", "val", "test"):
        print(f"split {s:5s}: {len(load_split(args.root, s)):>7,} images")

    keys = load_split(args.root, args.split)
    if not keys:
        keys = [p.stem for p in ann_dir.glob("*.json")]

    sample = json.loads((ann_dir / f"{keys[0]}.json").read_text())
    shown = dict(sample)
    shown["objects"] = sample.get("objects", [])[:2]
    print("\n--- sample annotation ---")
    print(json.dumps(shown, indent=2)[:2000])
    print("--- end sample ---\n")

    inst, flags = Counter(), Counter()
    imgs, sizes = defaultdict(set), defaultdict(list)
    n_pano = n_missing = boxes_total = 0
    rel_sizes = []

    for k in keys:
        p = ann_dir / f"{k}.json"
        if not p.exists():
            n_missing += 1
            continue
        a = json.loads(p.read_text())
        if a.get("ispano") or a.get("is_pano"):
            n_pano += 1
        W, H = a.get("width", 0), a.get("height", 0)
        for o in a.get("objects", []):
            b = o.get("bbox", {})
            try:
                w = float(b["xmax"]) - float(b["xmin"])
                h = float(b["ymax"]) - float(b["ymin"])
            except (KeyError, TypeError, ValueError):
                continue
            lab, side = o.get("label", "UNKNOWN"), max(w, h)
            inst[lab] += 1
            imgs[lab].add(k)
            sizes[lab].append(side)
            boxes_total += 1
            if W and H:
                rel_sizes.append(side / max(W, H))
            for fk, fv in (o.get("properties") or {}).items():
                if fv is True:
                    flags[fk] += 1

    rows = [(l, n, len(imgs[l]), round(median(sizes[l]), 1)) for l, n in inst.most_common()]
    with (args.out / "class_counts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "n_instances", "n_images", "median_box_px"])
        w.writerows(rows)

    tiny = sum(1 for s in rel_sizes if s < 0.02)
    L = [f"split analysed      : {args.split}",
         f"images              : {len(keys):,}  (missing json: {n_missing})",
         f"panoramas           : {n_pano:,}",
         f"boxes               : {boxes_total:,}",
         f"distinct labels     : {len(inst):,}",
         f"labels >=  50 inst  : {sum(1 for _, n in inst.items() if n >= 50)}",
         f"labels >= 100 inst  : {sum(1 for _, n in inst.items() if n >= 100)}",
         f"labels >= 500 inst  : {sum(1 for _, n in inst.items() if n >= 500)}",
         f"boxes < 2% of image : {tiny:,} ({100*tiny/max(boxes_total,1):.1f}%)",
         "", "superclass counts:"]
    sup = Counter()
    for l, n in inst.items():
        sup[l.split("--")[0]] += n
    L += [f"  {k:<20s} {v:>8,}" for k, v in sup.most_common()]
    L += ["", "property flags true:"]
    L += [f"  {k:<20s} {v:>8,}" for k, v in flags.most_common()]
    L += ["", "top 25 labels:"]
    L += [f"  {l:<55s} {n:>7,}  median {m:>5.0f}px" for l, n, _, m in rows[:25]]

    text = "\n".join(L)
    (args.out / "summary.txt").write_text(text)
    print(text)
    print(f"\nwrote {args.out/'class_counts.csv'} and {args.out/'summary.txt'}")


if __name__ == "__main__":
    main()
