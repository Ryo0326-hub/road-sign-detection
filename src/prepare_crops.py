"""Build the fine-grained classifier dataset: 153 classes + 'other'."""
import argparse, csv, json, random
from collections import Counter
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
random.seed(0)


def load_split(root, name):
    for c in [root / "splits" / f"{name}.txt", root / f"{name}.txt"]:
        if c.exists():
            return [ln.strip() for ln in c.read_text().splitlines() if ln.strip()]
    raise FileNotFoundError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True,
                    help="data/mtsd_yolo/images (the 1920px versions)")
    ap.add_argument("--counts", type=Path, default=Path("reports/class_counts.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/mtsd_crops"))
    ap.add_argument("--min-inst", type=int, default=100)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--min-src-px", type=int, default=16)
    ap.add_argument("--other-cap", type=int, default=3000)
    args = ap.parse_args()

    with args.counts.open() as f:
        keep = {r["label"] for r in csv.DictReader(f)
                if r["label"] != "other-sign" and int(r["n_instances"]) >= args.min_inst}
    print(f"keeping {len(keep)} named classes (>= {args.min_inst} inst) + 'other'")
    (args.out / "classes.txt").parent.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        img_dir = args.images / split
        stats, n_other = Counter(), 0
        for key in load_split(args.root, split):
            src = img_dir / f"{key}.jpg"
            ann = args.root / "annotations" / f"{key}.json"
            if not src.exists() or not ann.exists():
                continue
            a = json.loads(ann.read_text())
            W0, H0 = a.get("width"), a.get("height")
            try:
                im = Image.open(src); im.load()
            except Exception:
                continue
            if im.mode != "RGB":
                im = im.convert("RGB")
            W, H = im.size
            sx, sy = W / W0, H / H0

            for i, o in enumerate(a.get("objects", [])):
                p = o.get("properties") or {}
                if p.get("dummy") or p.get("ambiguous"):
                    continue          # unreadable class -> bad classifier label
                lab = o.get("label", "")
                if lab in keep:
                    cls = lab
                elif lab == "other-sign":
                    if n_other >= args.other_cap or random.random() > 0.05:
                        continue
                    cls, n_other = "other", n_other + 1
                else:
                    cls = "other"
                    if n_other >= args.other_cap:
                        continue
                    n_other += 1

                b = o["bbox"]
                x0, y0 = float(b["xmin"]) * sx, float(b["ymin"]) * sy
                x1, y1 = float(b["xmax"]) * sx, float(b["ymax"]) * sy
                w, h = x1 - x0, y1 - y0
                if min(w, h) < args.min_src_px:
                    continue
                px, py = w * args.pad, h * args.pad
                box = (max(0, x0 - px), max(0, y0 - py),
                       min(W, x1 + px), min(H, y1 + py))
                d = args.out / split / cls
                d.mkdir(parents=True, exist_ok=True)
                im.crop(box).resize((args.size, args.size), Image.LANCZOS) \
                  .save(d / f"{key}_{i}.jpg", quality=92)
                stats[cls] += 1

        print(f"{split}: {sum(stats.values()):,} crops across {len(stats)} classes "
              f"(other={stats['other']:,})")
        if split == "train":
            (args.out / "classes.txt").write_text("\n".join(sorted(stats)) + "\n")
            print("  10 smallest:", stats.most_common()[-10:])


if __name__ == "__main__":
    main()
