"""Fine-grained sign classifier: 154 classes, 64x64 crops."""
import argparse, json
from pathlib import Path
import numpy as np, torch, timm
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/mtsd_crops"))
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("weights"))
    a = ap.parse_args()

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)

    # no horizontal flip: mirrored arrows mean different signs
    tr_tf = T.Compose([
        T.RandomResizedCrop(64, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        T.RandomRotation(10),
        T.ColorJitter(0.4, 0.4, 0.3, 0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    va_tf = T.Compose([T.Resize((64, 64)), T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    tr = datasets.ImageFolder(a.data / "train", tr_tf)
    va = datasets.ImageFolder(a.data / "val", va_tf)
    print(f"train {len(tr):,} | val {len(va):,} | classes {len(tr.classes)}")

    cnt = np.bincount([y for _, y in tr.samples], minlength=len(tr.classes))
    w = (1.0 / np.maximum(cnt, 1))[[y for _, y in tr.samples]]
    sampler = WeightedRandomSampler(torch.DoubleTensor(w), len(w), replacement=True)

    nw = a.workers
    trl = DataLoader(tr, a.bs, sampler=sampler, num_workers=nw,
                     persistent_workers=nw > 0)
    val = DataLoader(va, 256, shuffle=False, num_workers=nw,
                     persistent_workers=nw > 0)

    model = timm.create_model(a.model, pretrained=True,
                              num_classes=len(tr.classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs,
                                                steps_per_epoch=len(trl))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    a.out.mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    best = 0.0

    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in trl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step(); sched.step()
            tot += loss.item() * x.size(0)

        model.eval()
        c1 = c5 = n = 0
        P, Y = [], []
        with torch.no_grad():
            for x, y in val:
                x, y = x.to(dev), y.to(dev)
                o = model(x)
                t5 = o.topk(5, 1).indices
                c1 += (t5[:, 0] == y).sum().item()
                c5 += (t5 == y[:, None]).any(1).sum().item()
                n += y.numel()
                P += t5[:, 0].cpu().tolist(); Y += y.cpu().tolist()
        top1, top5 = c1 / n, c5 / n
        print(f"ep {ep:3d}  loss {tot/len(tr):.4f}  top1 {top1:.4f}  top5 {top5:.4f}",
              flush=True)
        if top1 > best:
            best = top1
            torch.save({"model": a.model, "classes": tr.classes,
                        "state": model.state_dict()}, a.out / "classifier_best.pt")
            json.dump({"top1": top1, "top5": top5, "epoch": ep,
                       "preds": P, "labels": Y},
                      open("reports/classifier_val.json", "w"))
    print(f"\nbest top-1: {best:.4f}  ->  weights/classifier_best.pt")


if __name__ == "__main__":
    main()
