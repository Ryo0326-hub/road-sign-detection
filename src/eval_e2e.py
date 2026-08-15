"""End-to-end: detector -> crops -> classifier. Matched against GT by IoU."""
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np, torch, timm
from PIL import Image
from torchvision import transforms as T
from ultralytics import YOLO


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / ua if ua > 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="runs/detect/runs/det_1536/weights/best.pt")
    ap.add_argument("--clf", default="weights/classifier_best.pt")
    ap.add_argument("--root", type=Path, default=Path("data/raw/mtsd_v2_fully_annotated"))
    ap.add_argument("--images", type=Path, default=Path("data/mtsd_yolo/images/val"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    det = YOLO(a.det)
    ck = torch.load(a.clf, map_location=dev, weights_only=False)
    classes = ck["classes"]
    clf = timm.create_model(ck["model"], pretrained=False, num_classes=len(classes))
    clf.load_state_dict(ck["state"]); clf.to(dev).eval()
    tf = T.Compose([T.Resize((64, 64)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    imgs = sorted(a.images.glob("*.jpg"))
    if a.limit: imgs = imgs[:a.limit]
    print(f"{len(imgs)} val images")

    n_gt = n_det = n_hit = n_cls = 0
    per_cls = Counter(); per_cls_ok = Counter()

    for n, p in enumerate(imgs, 1):
        ann = json.loads((a.root / "annotations" / f"{p.stem}.json").read_text())
        im = Image.open(p).convert("RGB"); W, H = im.size
        sx, sy = W / ann["width"], H / ann["height"]
        gt = []
        for o in ann.get("objects", []):
            pr = o.get("properties") or {}
            if pr.get("dummy") or pr.get("ambiguous"): continue
            lab = o["label"] if o["label"] in classes else "other"
            b = o["bbox"]
            gt.append((lab, [b["xmin"]*sx, b["ymin"]*sy, b["xmax"]*sx, b["ymax"]*sy]))
        n_gt += len(gt)

        r = det.predict(p, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        n_det += len(boxes)
        if not len(boxes) or not gt: continue

        crops = []
        for b in boxes:
            w, h = b[2]-b[0], b[3]-b[1]
            crops.append(tf(im.crop((max(0, b[0]-0.15*w), max(0, b[1]-0.15*h),
                                     min(W, b[2]+0.15*w), min(H, b[3]+0.15*h)))))
        with torch.no_grad():
            pred = clf(torch.stack(crops).to(dev)).argmax(1).cpu().tolist()

        used = set()
        for glab, gbox in gt:
            best, bi = 0.0, -1
            for i, b in enumerate(boxes):
                if i in used: continue
                v = iou(gbox, b)
                if v > best: best, bi = v, i
            per_cls[glab] += 1
            if best >= 0.5:
                used.add(bi); n_hit += 1
                if classes[pred[bi]] == glab:
                    n_cls += 1; per_cls_ok[glab] += 1
        if n % 500 == 0: print(f"  {n}/{len(imgs)}", flush=True)

    rec = n_hit / max(n_gt, 1)
    cacc = n_cls / max(n_hit, 1)
    print(f"\n{'='*46}\nGT signs            : {n_gt:,}")
    print(f"Detections          : {n_det:,}")
    print(f"Detection recall    : {rec:.4f}")
    print(f"Clf acc on detected : {cacc:.4f}")
    print(f"END-TO-END          : {n_cls/max(n_gt,1):.4f}   <-- headline")
    print(f"{'='*46}")
    
    named_gt = sum(v for k, v in per_cls.items() if k != "other")
    named_ok = sum(v for k, v in per_cls_ok.items() if k != "other")
    oth_gt, oth_ok = per_cls["other"], per_cls_ok["other"]
    print(f"\n--- breakdown ---")
    print(f"named GT  : {named_gt:5d}   correct {named_ok:5d}  = {named_ok/max(named_gt,1):.4f}")
    print(f"other GT  : {oth_gt:5d}   correct {oth_ok:5d}  = {oth_ok/max(oth_gt,1):.4f}")
    
    json.dump({"gt": n_gt, "recall": rec, "clf_acc": cacc,
               "e2e": n_cls/max(n_gt,1)}, open("reports/e2e.json", "w"))


if __name__ == "__main__":
    main()
