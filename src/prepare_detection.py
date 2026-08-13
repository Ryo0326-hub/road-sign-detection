"""MTSD -> YOLO single-class detection dataset, with downscaling."""
import argparse, json
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load_split(root, name):
    for c in [root / "splits" / f"{name}.txt", root / f"{name}.txt"]:
        if c.exists():
            return [ln.strip() for ln in c.read_text().splitlines() if ln.strip()]
    raise FileNotFoundError(f"no split '{name}' under {root}")


def process(key, *, ann_dir, img_dir, out, split, max_side, min_box, keep_pano):
    ap_, ip = ann_dir / f"{key}.json", img_dir / f"{key}.jpg"
    if not ap_.exists() or not ip.exists():
        return 0, 0, 1
    a = json.loads(ap_.read_text())
    if a.get("ispano") and not keep_pano:
        return 0, 0, 1
    try:
        im = Image.open(ip); im.load()
    except Exception:
        return 0, 0, 1
    if im.mode != "RGB":
        im = im.convert("RGB")

    W, H = im.size
    sc = min(1.0, max_side / max(W, H))
    if sc < 1.0:
        im = im.resize((round(W * sc), round(H * sc)), Image.LANCZOS)
    nW, nH = im.size

    lines, kept, drop = [], 0, 0
    for o in a.get("objects", []):
        p = o.get("properties") or {}
        # keep 'ambiguous' (real signs, unknown class); drop only non-signs
        if p.get("dummy"):
            drop += 1; continue
        b = o.get("bbox", {})
        try:
            x0, y0 = float(b["xmin"]) * sc, float(b["ymin"]) * sc
            x1, y1 = float(b["xmax"]) * sc, float(b["ymax"]) * sc
        except Exception:
            drop += 1; continue
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(nW), x1), min(float(nH), y1)
        bw, bh = x1 - x0, y1 - y0
        if bw < min_box or bh < min_box:
            drop += 1; continue
        lines.append(f"0 {(x0+bw/2)/nW:.6f} {(y0+bh/2)/nH:.6f} {bw/nW:.6f} {bh/nH:.6f}")
        kept += 1

    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    im.save(out / "images" / split / f"{key}.jpg", quality=88, optimize=True)
    (out / "labels" / split / f"{key}.txt").write_text("\n".join(lines))
    return kept, drop, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/mtsd_yolo"))
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--max-side", type=int, default=1920)
    ap.add_argument("--min-box-px", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep-pano", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        keys = load_split(args.root, split)
        if args.limit:
            keys = keys[: args.limit]
        fn = partial(process, ann_dir=args.root / "annotations", img_dir=args.images,
                     out=args.out, split=split, max_side=args.max_side,
                     min_box=args.min_box_px, keep_pano=args.keep_pano)
        kept = drop = skip = 0
        with Pool(args.workers) as pool:
            for i, (k, d, s) in enumerate(pool.imap_unordered(fn, keys, chunksize=32), 1):
                kept += k; drop += d; skip += s
                if i % 1000 == 0:
                    print(f"  {split}: {i:,}/{len(keys):,}  {kept:,} boxes", flush=True)
        print(f"{split}: {len(keys):,} imgs -> {kept:,} boxes kept, {drop:,} dropped, {skip:,} imgs skipped")

    Path("configs").mkdir(exist_ok=True)
    Path("configs/mtsd_det.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: traffic-sign\n")
    print("wrote configs/mtsd_det.yaml")


if __name__ == "__main__":
    main()
