"""
pipeline.py
-----------
Oxford 102 Flower Classification using ResNet50.

Matches the darkangrycoder approach:
  - All 8,189 images organised into per-class folders by flower name
  - Random train/valid/test split (no setid.mat)
  - ResNet50 pretrained on ImageNet, FC head replaced for 102 classes
  - Discriminative learning rates: backbone gets lr/10, head gets lr

Usage:
    python pipeline.py --mode prepare
    python pipeline.py --mode train
    python pipeline.py --mode train --epochs 5 --lr 0.001 --train_pct 0.90 --valid_pct 0.10
    python pipeline.py --mode evaluate
    python pipeline.py --mode predict --image path/to/flower.jpg
"""

import os
import json
import copy
import tarfile
import shutil
import argparse
import urllib.request

import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
data_dir    = "data"
image_dir   = os.path.join(data_dir, "jpg")
labeled_dir = os.path.join(data_dir, "labeled")
ckpt_dir    = "checkpoints"
best_ckpt   = os.path.join(ckpt_dir, "resnet50_best.pth")
final_ckpt  = os.path.join(ckpt_dir, "resnet50_final.pth")
num_classes = 102
seed        = 42
mean        = [0.485, 0.456, 0.406]
std         = [0.229, 0.224, 0.225]

urls = {
    "102flowers.tgz":  "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
    "imagelabels.mat": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def download_dataset():
    os.makedirs(data_dir, exist_ok=True)
    for filename, url in urls.items():
        dest = os.path.join(data_dir, filename)
        if os.path.exists(dest):
            print(f"  [skip] {filename}")
            continue
        print(f"  Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
    tgz = os.path.join(data_dir, "102flowers.tgz")
    if not os.path.isdir(image_dir):
        print("  Extracting images ...")
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(data_dir)
    n = len([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
    print(f"  ✓ {n} images in {image_dir}")


def organize_dataset(mapping_path="mapping.json"):
    if os.path.isdir(labeled_dir):
        n = sum(len(f) for _, _, f in os.walk(labeled_dir))
        print(f"  [skip] labeled directory already exists ({n} files)")
        return
    with open(mapping_path) as f:
        mapping = json.load(f)
    labels_mat = scipy.io.loadmat(os.path.join(data_dir, "imagelabels.mat"))
    labels = labels_mat["labels"].flatten()
    print(f"  Organising {len(labels)} images into class folders ...")
    for i, label in enumerate(labels):
        img_id    = i + 1
        name      = mapping[str(int(label))]
        safe_name = name.replace(" ", "_").replace("/", "-").replace("'", "")
        src       = os.path.join(image_dir, f"image_{img_id:05d}.jpg")
        dst_dir   = os.path.join(labeled_dir, safe_name)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"image_{img_id:05d}.jpg")
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    print(f"  ✓ Images organised into {labeled_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATALOADERS
# ══════════════════════════════════════════════════════════════════════════════

def get_transforms(augment=True):
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]) if augment else val_tf
    return {"train": train_tf, "valid": val_tf, "test": val_tf}


def get_dataloaders(batch_size=64, workers=2, augment=True, train_pct=0.80, valid_pct=0.10):
    tf      = get_transforms(augment=augment)
    full_ds = datasets.ImageFolder(root=labeled_dir)

    class_to_indices = {}
    for idx, (_, label) in enumerate(full_ds.samples):
        class_to_indices.setdefault(label, []).append(idx)

    train_idx, valid_idx, test_idx = [], [], []
    rng = torch.Generator().manual_seed(seed)

    for label, indices in class_to_indices.items():
        n       = len(indices)
        perm    = torch.randperm(n, generator=rng).tolist()
        indices = [indices[i] for i in perm]
        n_train = max(1, int(n * train_pct))
        n_valid = max(1, int(n * valid_pct))
        train_idx += indices[:n_train]
        valid_idx += indices[n_train:n_train + n_valid]
        test_idx  += indices[n_train + n_valid:]

    train_ds = datasets.ImageFolder(root=labeled_dir, transform=tf["train"])
    val_ds   = datasets.ImageFolder(root=labeled_dir, transform=tf["valid"])
    test_ds  = datasets.ImageFolder(root=labeled_dir, transform=tf["test"])

    dataloaders = {
        "train": DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True),
        "valid": DataLoader(Subset(val_ds,   valid_idx), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
        "test":  DataLoader(Subset(test_ds,  test_idx),  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
    }

    idx_to_name = {idx: name.replace("_", " ") for name, idx in full_ds.class_to_idx.items()}

    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "idx_to_name.json"), "w") as f:
        json.dump(idx_to_name, f, indent=2)

    print(f"  train : {len(train_idx)} images")
    print(f"  valid : {len(valid_idx)} images")
    print(f"  test  : {len(test_idx)} images")

    return dataloaders, idx_to_name


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model.to(device)


def load_checkpoint(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model = build_model(device)
    model.load_state_dict(state)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    dataloaders, idx_to_name = get_dataloaders(
        batch_size=args.batch_size,
        workers=args.workers,
        augment=True,
        train_pct=args.train_pct,
        valid_pct=args.valid_pct,
    )

    model     = build_model(device)
    criterion = nn.CrossEntropyLoss()

    # Discriminative learning rates — backbone gets lr/10, head gets full lr
    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params     = list(model.fc.parameters())
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": args.lr / 10},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler    = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    best_weights = copy.deepcopy(model.state_dict())
    best_acc     = 0.0
    history      = []   # per-epoch metrics for the JSON log

    for epoch in range(args.epochs):
        lrs = scheduler.get_last_lr()
        print(f"\nEpoch {epoch+1}/{args.epochs}  backbone_lr={lrs[0]:.2e}  head_lr={lrs[1]:.2e}")
        epoch_record = {"epoch": epoch + 1, "backbone_lr": lrs[0], "head_lr": lrs[1]}

        for phase in ["train", "valid"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_correct = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    preds   = outputs.argmax(dim=1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss    += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()

            n       = len(dataloaders[phase].dataset)
            ep_loss = running_loss / n
            ep_acc  = running_correct / n
            marker  = ""

            if phase == "valid" and ep_acc > best_acc:
                best_acc     = ep_acc
                best_weights = copy.deepcopy(model.state_dict())
                marker       = "  ← best"

            epoch_record[f"{phase}_loss"] = ep_loss
            epoch_record[f"{phase}_acc"]  = ep_acc

            print(f"  {phase:5s} | loss {ep_loss:.4f}  acc {ep_acc:.4f}{marker}")

        history.append(epoch_record)
        scheduler.step()

    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), final_ckpt)
    torch.save({
        "model_state_dict": best_weights,
        "best_val_acc":     best_acc,
        "num_classes":      num_classes,
        "idx_to_name":      idx_to_name,
    }, best_ckpt)

    # Save training history as JSON
    train_log = {
        "best_val_acc": best_acc,
        "epochs":       args.epochs,
        "lr":           args.lr,
        "batch_size":   args.batch_size,
        "train_pct":    args.train_pct,
        "valid_pct":    args.valid_pct,
        "history":      history,
    }
    log_path = args.training_log
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\n✓ Best val acc  : {best_acc:.4f}")
    print(f"✓ Checkpoints   : {best_ckpt}  |  {final_ckpt}")
    print(f"✓ Training log  : {log_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders, idx_to_name = get_dataloaders(
        batch_size=args.batch_size,
        workers=args.workers,
        augment=False,
        train_pct=args.train_pct,
        valid_pct=args.valid_pct,
    )
    model = load_checkpoint(args.ckpt, device)

    top1_correct, top5_correct, total = 0, 0, 0
    per_class = {}

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            pred1   = outputs.argmax(dim=1)
            top1_correct += (pred1 == labels).sum().item()
            _, pred5 = outputs.topk(5, dim=1)
            for i, lbl in enumerate(labels):
                top5_correct += int(lbl in pred5[i])
            for pred, lbl in zip(pred1, labels):
                c = lbl.item()
                if c not in per_class:
                    per_class[c] = {"correct": 0, "total": 0}
                per_class[c]["total"]   += 1
                per_class[c]["correct"] += int(pred == lbl)
            total += labels.size(0)

    top1 = top1_correct / total
    top5 = top5_correct / total

    print(f"\n{'='*50}")
    print(f"  Top-1 accuracy : {top1*100:.2f}%")
    print(f"  Top-5 accuracy : {top5*100:.2f}%")
    print(f"{'='*50}\n")

    rows = []
    for idx, stats in per_class.items():
        acc  = stats["correct"] / stats["total"] if stats["total"] else 0.0
        name = idx_to_name.get(idx, f"class_{idx}")
        rows.append((acc, name, stats["correct"], stats["total"]))
    rows.sort()

    print(f"{'Flower':<35} {'Acc':>7}  {'n':>6}")
    print("-" * 55)
    for acc, name, correct, total_c in rows:
        bar = "█" * int(acc * 20)
        print(f"{name:<35} {acc*100:6.1f}%  {total_c:>6}   {bar}")

    accs = [r[0] for r in rows]
    print(f"\nWorst : {min(accs)*100:.1f}%   Best : {max(accs)*100:.1f}%   Mean : {sum(accs)/len(accs)*100:.1f}%")

    # Save results to JSON for later analysis (e.g. fairness plots)
    results = {
        "top1": top1,
        "top5": top5,
        "n_test_images": total,
        "summary": {
            "worst_class_acc": min(accs),
            "best_class_acc":  max(accs),
            "mean_class_acc":  sum(accs) / len(accs),
        },
        "per_class": {
            name: {
                "acc":     acc,
                "correct": correct,
                "total":   total_c,
            }
            for acc, name, correct, total_c in rows
        },
    }

    out_path = args.results_out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def predict(args):
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt        = torch.load(args.ckpt, map_location=device)
    idx_to_name = ckpt.get("idx_to_name", {})
    model       = load_checkpoint(args.ckpt, device)

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tensor = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze()

    topk_probs, topk_idx = probs.topk(args.topk)
    print(f"\nImage : {args.image}")
    print(f"{'='*45}")
    for rank, (prob, idx) in enumerate(zip(topk_probs, topk_idx), 1):
        name = idx_to_name.get(idx.item(), f"class_{idx.item()}")
        bar  = "█" * int(prob.item() * 30)
        print(f"  {rank}. {name:<35} {prob.item()*100:5.2f}%  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    parser = argparse.ArgumentParser(description="Oxford 102 ResNet50 pipeline")
    parser.add_argument("--mode",       required=True, choices=["prepare", "train", "evaluate", "predict"])
    parser.add_argument("--ckpt",       default=best_ckpt)
    parser.add_argument("--image",      default=None,    help="Image path (predict mode)")
    parser.add_argument("--topk",       type=int,   default=5)
    parser.add_argument("--epochs",     type=int,   default=25)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--step_size",  type=int,   default=7)
    parser.add_argument("--gamma",      type=float, default=0.1)
    parser.add_argument("--workers",    type=int,   default=2)
    parser.add_argument("--train_pct",  type=float, default=0.80, help="Fraction of data for training")
    parser.add_argument("--valid_pct",  type=float, default=0.10, help="Fraction of data for validation")
    parser.add_argument("--results_out",  default="evaluation_results.json", help="JSON file to save eval results")
    parser.add_argument("--training_log", default="training_log.json", help="JSON file to save training history")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.mode == "prepare":
        print("\n[1/2] Downloading ...")
        download_dataset()
        print("\n[2/2] Organising ...")
        organize_dataset()
        print("\nDone. Run:  python pipeline.py --mode train")
    elif args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "predict":
        assert args.image, "--image is required for predict mode"
        predict(args)
