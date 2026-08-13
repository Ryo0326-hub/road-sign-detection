"""Render YOLO labels back onto converted images so you can eyeball them."""
import argparse, random
from pathlib import Path
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("data/tmp_check"))
ap.add_argument("--split", default="val")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--out", type=Path, default=Path("reports/viz"))
a = ap.parse_args()

imgs = sorted((a.data / "images" / a.split).glob("*.jpg"))
withboxes = [p for p in imgs if (a.data / "labels" / a.split / f"{p.stem}.txt").read_text().strip()]
print(f"{len(imgs)} images, {len(withboxes)} have boxes")
a.out.mkdir(parents=True, exist_ok=True)

for p in random.sample(withboxes, min(a.n, len(withboxes))):
    im = Image.open(p).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    n = 0
    for ln in (a.data / "labels" / a.split / f"{p.stem}.txt").read_text().splitlines():
        _, cx, cy, bw, bh = (float(x) for x in ln.split())
        x0, y0 = (cx - bw / 2) * W, (cy - bh / 2) * H
        x1, y1 = (cx + bw / 2) * W, (cy + bh / 2) * H
        d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        n += 1
    im.save(a.out / f"{p.stem}.jpg", quality=85)
    print(f"  {p.stem}: {n} boxes, {W}x{H}")
print(f"\nwrote to {a.out}/ -- open them")
