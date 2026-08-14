import json
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

d = json.load(open("reports/classifier_val.json"))
classes = [l.strip() for l in open("data/mtsd_crops/classes.txt")]
y, p = np.array(d["labels"]), np.array(d["preds"])
print(f"best epoch {d['epoch']}  top1 {d['top1']:.4f}  top5 {d['top5']:.4f}\n")

cm = confusion_matrix(y, p, labels=range(len(classes)))
errs = Counter()
for i in range(len(classes)):
    for j in range(len(classes)):
        if i != j and cm[i, j]:
            errs[(classes[i], classes[j])] += cm[i, j]

print("top 20 confusions (true -> predicted):")
for (t, pr), n in errs.most_common(20):
    print(f"  {n:3d}  {t}\n       -> {pr}")

sup = cm.sum(1)
acc = np.divide(cm.diagonal(), np.maximum(sup, 1))
order = np.argsort(acc)
print("\n15 worst classes:")
for i in order[:15]:
    if sup[i]:
        print(f"  {acc[i]:.2f}  (n={sup[i]:3d})  {classes[i]}")

open("reports/classification_report.txt", "w").write(
    classification_report(y, p, labels=range(len(classes)),
                          target_names=classes, zero_division=0))
print("\nwrote reports/classification_report.txt")
