"""Train the shape CNN - renders first, then fine-tune on our own photos.

Designed for a Kaggle Notebook (free GPU, dataset attached):

    !git clone https://github.com/<you>/lego-recognition && cd lego-recognition
    !pip install -q timm wandb
    !python scripts/train_cnn.py \
        --renders "/kaggle/input/images-of-lego-bricks" \
        --real-train data/real/train --real-test data/real/test \
        --model resnet18 --aug real --epochs 15 --ft-epochs 10 --out models/shape_cnn.pt

The core experiment (slide 5) = run this several times and compare the
accuracy on --real-test:
    A  --aug basic,  no --real-train          renders only
    B  --aug real,   no --real-train          + realistic augmentation
    C  --aug real,   --real-train ...         + fine-tuning on our photos
    (D the same with --pretrained, if the supervisors allow ImageNet weights)
Every run writes a JSON line to results/experiments.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader, Dataset  # noqa: E402

from lego.augment import augment, render_to_input  # noqa: E402
from lego.data import find_class_root, list_image_folder, read_grey, stratified_split  # noqa: E402
from lego.shape_cnn import MEAN, STD, build_model, save_checkpoint  # noqa: E402


class BrickDataset(Dataset):
    def __init__(self, paths, labels, size, is_render, aug_level, cache=False):
        self.paths, self.labels = paths, labels
        self.size, self.is_render, self.aug_level = size, is_render, aug_level
        self.cache = {} if cache else None

    def __len__(self):
        return len(self.paths)

    def _load(self, i):
        if self.cache is not None and i in self.cache:
            return self.cache[i]
        g = render_to_input(read_grey(self.paths[i]), self.size) if self.is_render else read_grey(self.paths[i], self.size)
        if self.cache is not None:
            self.cache[i] = g
        return g

    def __getitem__(self, i):
        rng = np.random.default_rng()
        g = augment(self._load(i), rng, self.aug_level)
        x = (g.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x[None]), self.labels[i]


def evaluate(model, loader, device, num_classes):
    model.eval()
    correct = top3 = n = 0
    conf = np.zeros((num_classes, num_classes), np.int64)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = sum(model(torch.rot90(x, k, (2, 3))).float().softmax(1) for k in range(4))
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            top3 += (logits.topk(3, 1).indices == y[:, None]).any(1).sum().item()
            n += len(y)
            for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
                conf[t, p] += 1
    tp = np.diag(conf)
    f1 = 2 * tp / np.maximum(conf.sum(0) + conf.sum(1), 1)
    present = conf.sum(1) > 0
    return {"acc": correct / max(n, 1), "top3": top3 / max(n, 1), "macro_f1": float(f1[present].mean()) if present.any() else 0.0, "n": n}


def run_stage(name, model, train_ds, val_loader, test_loader, args, device, epochs, lr, num_classes, log):
    loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        t0, tot, cnt = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            cnt += len(y)
        val = evaluate(model, val_loader, device, num_classes)
        msg = {"stage": name, "epoch": ep + 1, "loss": tot / cnt, **{f"val_{k}": v for k, v in val.items()}}
        if test_loader is not None:
            msg.update({f"real_{k}": v for k, v in evaluate(model, test_loader, device, num_classes).items()})
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in msg.items()}), f"{time.time() - t0:.0f}s", flush=True)
        log(msg)
        if val["acc"] > best:  # select on validation, never on the real test set
            best = val["acc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, help="Kaggle render folder (class subfolders)")
    ap.add_argument("--real-train", help="our labelled crops for fine-tuning (scripts/extract_crops.py)")
    ap.add_argument("--real-test", help="our labelled crops for testing - NEVER trained on")
    ap.add_argument("--model", default="resnet18", help="any timm model, e.g. resnet18, efficientnet_b0, convnext_nano")
    ap.add_argument("--pretrained", action="store_true", help="ImageNet weights (only if allowed!)")
    ap.add_argument("--aug", default="real", choices=["none", "basic", "real"])
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ft-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ft-lr", type=float, default=3e-4)
    ap.add_argument("--render-mix", type=float, default=1.0, help="renders per real crop during fine-tuning (keeps it from forgetting)")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="max images per class (quick tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/shape_cnn.pt")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--tag", default="", help="name of this run in the experiment table")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = find_class_root(Path(args.renders))
    paths, labels, classes = list_image_folder(root)
    if args.limit:
        keep, seen = [], {}
        for i, y in enumerate(labels):
            seen[y] = seen.get(y, 0) + 1
            if seen[y] <= args.limit:
                keep.append(i)
        paths, labels = [paths[i] for i in keep], [labels[i] for i in keep]
    print(f"renders: {len(paths)} images, {len(classes)} classes from {root}")
    if len(classes) < 2 or len(paths) < 2 * len(classes):
        sys.exit(f"Could not find the render dataset under {args.renders} (found {len(paths)} images, {len(classes)} classes). "
                 f"Check the path, or run:  !ls -R {args.renders} | head -50")
    tr, va = stratified_split(labels, 0.1, args.seed)
    r_train = BrickDataset([paths[i] for i in tr], [labels[i] for i in tr], args.img, True, args.aug)
    r_val = BrickDataset([paths[i] for i in va], [labels[i] for i in va], args.img, True, "none", cache=True)
    val_loader = DataLoader(r_val, batch_size=args.bs, num_workers=0)  # num_workers=0 so the cache persists

    test_loader = None
    if args.real_test:
        tp, tl, _ = list_image_folder(args.real_test, classes)
        test_loader = DataLoader(BrickDataset(tp, tl, args.img, False, "none", cache=True), batch_size=args.bs, num_workers=0)
        print(f"real test: {len(tp)} crops")

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="lego-recognition", name=args.tag or None, config=vars(args))

    def log(d):
        if run:
            run.log(d)

    model = build_model(args.model, len(classes), pretrained=args.pretrained).to(device)
    model = run_stage("renders", model, r_train, val_loader, test_loader, args, device, args.epochs, args.lr, len(classes), log)
    results = {"after_renders": evaluate(model, test_loader, device, len(classes)) if test_loader else None}

    if args.real_train:
        rp, rl, _ = list_image_folder(args.real_train, classes)
        real_ds = BrickDataset(rp, rl, args.img, False, "basic" if args.aug != "none" else "none")
        n_r = int(len(rp) * args.render_mix)
        idx = np.random.default_rng(args.seed).choice(len(r_train), size=min(n_r, len(r_train)), replace=False)
        mix = torch.utils.data.Subset(r_train, idx.tolist())
        print(f"fine-tuning on {len(rp)} real crops + {len(mix)} renders")
        model = run_stage("finetune", model, ConcatDataset([real_ds, mix]), val_loader, test_loader, args, device,
                          args.ft_epochs, args.ft_lr, len(classes), log)
        results["after_finetune"] = evaluate(model, test_loader, device, len(classes)) if test_loader else None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.out, model.cpu(), classes, args.model, args.img, extra={"args": vars(args), "results": results})
    print("saved", args.out)

    Path("results").mkdir(exist_ok=True)
    with open("results/experiments.jsonl", "a") as f:
        f.write(json.dumps({"tag": args.tag, "model": args.model, "pretrained": args.pretrained, "aug": args.aug,
                            "finetuned": bool(args.real_train), **results, "time": time.strftime("%F %T")}) + "\n")
    print(json.dumps(results, indent=2))
    if run:
        run.summary.update({k: v for k, v in results.items() if v})
        run.finish()


if __name__ == "__main__":
    main()
