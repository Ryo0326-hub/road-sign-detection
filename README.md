# Road Sign Detection &amp; Classification

A two-stage computer vision pipeline that locates traffic signs in street-level imagery and identifies them among 154 sign types. Built on the Mapillary Traffic Sign Dataset (MTSD): 41,909 images, 180,286 annotated signs, global coverage.

```
input image ──▶  detector (YOLO26-s, 1536px)  ──▶  sign crops  ──▶  classifier (EfficientNet-B0)  ──▶  labelled signs
                 "where are the signs?"                             "which sign is this?"
```

---

## Results

| Stage | Metric | Result |
|---|---|---|
| **Detector** | mAP@50 | **0.871** |
| | mAP@50-95 | 0.661 |
| | Recall | 0.861 |
| | Inference | 1.9 ms/image (MI300X) |
| **Classifier** | Top-1 (154 classes) | **0.890** |
| | Top-5 | 0.992 |
| **End-to-end** | Named sign types | **0.722** |
| | All signs incl. unnamed | 0.652 |

Evaluated on the official MTSD validation split (5,210 images, 16,584 signs).

**On the end-to-end figure.** Errors compound across stages: the system finds 86.1% of signs and correctly names 83.9% of those it finds. Both the named-type and all-signs numbers are reported because they answer different questions — the first is "can it identify the sign types it was trained on", the second is "what fraction of everything on the road does it get right".

---

## Why two stages

MTSD's class distribution is severely long-tailed, which makes a single multi-class detector the wrong tool:

- **66% of annotated boxes carry a generic `other-sign` label** — a real sign, but outside the 400-class taxonomy
- Named classes span 2,428 examples (`yield`) down to single digits
- 63.5% of signs occupy under 2% of image width; the median sign is ~45 px in a 3264 px-wide image

Splitting the problem lets the detector learn "signs look like this" from all 145,888 boxes — including every `other-sign` — while the classifier trains on a balanced subset of nameable classes. It also means new sign types can be added by retraining only the small classifier.

---

## Engineering decisions

**Class cut at 100 instances.** 153 of 401 classes clear the threshold, covering 80% of labelled signs. Cutting at 25 instances would cover 98.6% but leave tail classes with too few examples to learn rather than memorise. The cut is stated explicitly rather than buried.

**`ambiguous` kept for detection, dropped for classification.** MTSD flags 33% of boxes as ambiguous, meaning the *class* is unreadable — not that the box is wrong. Dropping them would teach the detector that small blurry signs are background, precisely the failure mode that matters most. For classification, an unreadable sign is genuinely bad label data.

**1536 px training resolution.** A median 45 px sign in a 3264 px image becomes ~21 px at 1536, near YOLO's practical floor. The resolution was chosen from the measured box-size distribution, not by default.

**No horizontal flip augmentation.** A mirrored "turn left" is a different sign. Ultralytics flips 50% of the time by default.

**Verified the annotation parser visually before training.** Converted 50 images, rendered YOLO labels back onto them, and inspected by eye. Scale-factor bugs in dataset conversion produce models that train without error and detect nothing.

---

## A bug found in the evaluation, not the model

The first classifier scored **93.2% top-1** in isolation but only **57.1% end-to-end**. Breaking the metric down by category located the cause:

| | v1 | v2 |
|---|---|---|
| Named-class accuracy | 0.830 | 0.839 |
| `other` accuracy | **0.405** | **0.528** |
| **End-to-end** | **0.571** | **0.652** |

The classifier had been trained on 6.8% `other` crops but evaluated on data that was 58% `other` — so it almost never predicted `other` and confidently mislabelled every unnamed sign. Rebuilding the crop dataset with a representative `other` proportion and retraining lifted end-to-end by **8.1 points** with no loss on named classes.

Standalone top-1 fell from 0.932 to 0.890 in the process. That was the right trade: the first number was measuring a distribution that doesn't exist in deployment.

---

## Repository layout

```
src/
  inspect_mtsd.py        annotation schema audit, class frequency analysis
  prepare_detection.py   MTSD JSON -> YOLO format, with downscaling
  prepare_crops.py       classifier dataset: 153 classes + other bucket
  viz_check.py           renders labels back onto images for verification
  train_classifier.py    EfficientNet-B0, class-balanced sampling
  eval_e2e.py            end-to-end evaluation with IoU matching
  classifier_report.py   confusion analysis, per-class accuracy
  make_report.py         generates the PDF demo report
configs/
  mtsd_det.yaml          Ultralytics dataset config
reports/                 metrics, curves, confusion analysis, demo PDF
```

---

## Reproducing

MTSD is **not** included — it is licensed for academic research and must be obtained from [mapillary.com/dataset/trafficsign](https://www.mapillary.com/dataset/trafficsign) under your own account.

```bash
pip install -r requirements.txt

# 1. audit the annotations (no images needed)
python src/inspect_mtsd.py --root data/raw/mtsd_v2_fully_annotated

# 2. build the detection dataset
python src/prepare_detection.py --root data/raw/mtsd_v2_fully_annotated \
    --images data/raw/images --out data/mtsd_yolo --workers 8

# 3. verify the conversion visually before training
python src/prepare_detection.py --out data/tmp_check --splits val --limit 50 ...
python src/viz_check.py

# 4. train the detector
yolo detect train model=yolo26s.pt data=configs/mtsd_det.yaml \
    imgsz=1536 epochs=45 batch=24 patience=12 fliplr=0.0

# 5. build crops and train the classifier
python src/prepare_crops.py --root data/raw/mtsd_v2_fully_annotated \
    --images data/mtsd_yolo/images --other-cap 15000
python src/train_classifier.py

# 6. evaluate end-to-end and generate the report
python src/eval_e2e.py --det weights/detector_best.pt --clf weights/classifier_best.pt
python src/make_report.py --pages 10
```

**Compute.** Detector trained on a single AMD Instinct MI300X (ROCm 7.2, PyTorch 2.10), ~41 epochs to early stopping. Classifier trained locally on Apple Silicon via the MPS backend — 44,401 crops at 64×64 is small enough that renting a GPU for it would have been wasteful.

---

## Limitations

- **Licence.** MTSD's free edition is academic-research only. A commercial deployment would need the commercial edition or a differently-licensed dataset.
- **Class coverage.** 154 of 401 sign types, covering 80% of labelled signs.
- **Unnamed signs** are the dominant error source at 52.8% end-to-end accuracy, and are visually unbounded by construction.
- **No local road data.** Sign designs are country-specific; the model is trained on globally sourced imagery. Accuracy on a specific road network would improve measurably with a few hundred locally labelled images.
- **Single-frame.** Building a georeferenced asset inventory requires multi-frame tracking so each physical sign is counted once, not once per frame.

## Next steps

Multi-frame tracking for inventory deduplication · confidence-threshold tuning for recall-oriented operation · sign condition assessment (faded, bent, occluded) · domain adaptation to a target road network.

---

**Stack** — PyTorch · Ultralytics YOLO26 · timm · ROCm · Pillow · scikit-learn · ReportLab
