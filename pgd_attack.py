"""
pgd_attack.py
=============
Projected Gradient Descent (PGD) adversarial attack on the ResNet50
Oxford 102 classifier.

Mirrors the structure of the HW4 adv_attack.py but adapted for:
  - Oxford 102 (not CIFAR-10 / MNIST)
  - ResNet50 (not LeNet / ResNet18)
  - ImageNet normalisation (so epsilon must be applied in normalised space)

Usage:
    python pgd_attack.py --ckpt checkpoints/resnet50_best.pth
    python pgd_attack.py --epsilon 0.03 --niter 20 --stepsize 0.005
    python pgd_attack.py --epsilon 0.01 0.03 0.05 0.1 --niter 20

The script measures:
  - Clean test accuracy (baseline)
  - Adversarial accuracy at one or more epsilon budgets
  - Per-class adversarial accuracy (for the fairness analysis)

Results are saved to JSON for later plotting.
"""

import os
import json
import argparse
import collections

import torch
import torch.nn as nn

from pipeline_v2 import (
    get_dataloaders, load_checkpoint, mean, std, ckpt_dir
)


# ══════════════════════════════════════════════════════════════════════════════
# PGD attack
# ══════════════════════════════════════════════════════════════════════════════

def PGD(x, y, model, criterion,
        niter=20, epsilon=0.03, stepsize=0.005,
        randinit=True, device=None):
    """
    Projected Gradient Descent attack (L-infinity).

    Args:
        x         : clean inputs (B, C, H, W) — already normalised
        y         : true labels (B,)
        model     : target model
        criterion : loss function (typically CrossEntropyLoss)
        niter     : number of PGD steps
        epsilon   : L-inf perturbation budget (in normalised space)
        stepsize  : step size per iteration
        randinit  : whether to start from a random point inside the epsilon ball
        device    : torch device (auto-detected if None)

    Returns:
        x_adv : adversarial examples (B, C, H, W)
    """
    if device is None:
        device = next(model.parameters()).device

    x = x.to(device)
    y = y.to(device)
    x_clean = x.clone().detach()

    # Random initialisation inside the epsilon ball
    if randinit:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = (x_clean + delta).detach()
    else:
        x_adv = x_clean.clone().detach()

    for _ in range(niter):
        x_adv.requires_grad_(True)

        outputs = model(x_adv)
        loss    = criterion(outputs, y)

        grad = torch.autograd.grad(loss, x_adv)[0]

        # Step in the direction of the sign of the gradient (L-inf)
        x_adv = x_adv.detach() + stepsize * grad.sign()

        # Project back into the epsilon ball around x_clean
        delta = torch.clamp(x_adv - x_clean, min=-epsilon, max=epsilon)
        x_adv = (x_clean + delta).detach()

    return x_adv


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation under attack
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_adversarial(model, dataloader, criterion,
                          niter, epsilon, stepsize, randinit,
                          device):
    """
    Run PGD on every test batch and return overall + per-class accuracy.

    Returns:
        top1_acc  : float
        per_class : dict {class_idx: {"correct": int, "total": int}}
    """
    model.eval()
    correct, total = 0, 0
    per_class = collections.defaultdict(lambda: {"correct": 0, "total": 0})

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        x_adv = PGD(inputs, labels, model, criterion,
                    niter=niter, epsilon=epsilon, stepsize=stepsize,
                    randinit=randinit, device=device)

        with torch.no_grad():
            outputs = model(x_adv)
            preds   = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        for pred, lbl in zip(preds, labels):
            c = lbl.item()
            per_class[c]["total"]   += 1
            per_class[c]["correct"] += int(pred == lbl)

    return correct / total, dict(per_class)


def evaluate_clean(model, dataloader, device):
    """Standard clean-accuracy evaluation (no attack)."""
    model.eval()
    correct, total = 0, 0
    per_class = collections.defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(dim=1)

            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            for pred, lbl in zip(preds, labels):
                c = lbl.item()
                per_class[c]["total"]   += 1
                per_class[c]["correct"] += int(pred == lbl)

    return correct / total, dict(per_class)


# ══════════════════════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_pgd_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # Load test set with no augmentation — required for reproducible attacks
    dataloaders, idx_to_name = get_dataloaders(
        batch_size=args.batch_size,
        workers=args.workers,
        augment=False,
        train_pct=args.train_pct,
        valid_pct=args.valid_pct,
    )
    test_loader = dataloaders["test"]
    model       = load_checkpoint(args.ckpt, device)
    criterion   = nn.CrossEntropyLoss()

    # ── Clean baseline ────────────────────────────────────────────────────
    print("Clean baseline (no attack) ...")
    clean_acc, clean_per_class = evaluate_clean(model, test_loader, device)
    print(f"  Clean accuracy : {clean_acc*100:.2f}%\n")

    # ── PGD at each epsilon ───────────────────────────────────────────────
    results = {
        "clean_acc": clean_acc,
        "clean_per_class": {
            idx_to_name.get(c, f"class_{c}"): s for c, s in clean_per_class.items()
        },
        "attack_params": {
            "niter":    args.niter,
            "stepsize": args.stepsize,
            "randinit": args.randinit,
        },
        "epsilons": {},
    }

    for eps in args.epsilon:
        print(f"PGD attack  epsilon={eps}  niter={args.niter}  stepsize={args.stepsize} ...")
        adv_acc, adv_per_class = evaluate_adversarial(
            model, test_loader, criterion,
            niter=args.niter, epsilon=eps,
            stepsize=args.stepsize, randinit=args.randinit,
            device=device,
        )
        drop = clean_acc - adv_acc
        print(f"  Adversarial accuracy : {adv_acc*100:.2f}%  (drop: {drop*100:.2f} pts)\n")

        results["epsilons"][str(eps)] = {
            "adv_acc":   adv_acc,
            "drop":      drop,
            "per_class": {
                idx_to_name.get(c, f"class_{c}"): s for c, s in adv_per_class.items()
            },
        }

    # ── Save JSON ─────────────────────────────────────────────────────────
    out_path = args.results_out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"{'Epsilon':>10}  {'Adv Acc':>10}  {'Drop':>8}")
    print("-" * 50)
    print(f"{'clean':>10}  {clean_acc*100:>9.2f}%  {'—':>8}")
    for eps in args.epsilon:
        r = results["epsilons"][str(eps)]
        print(f"{eps:>10.4f}  {r['adv_acc']*100:>9.2f}%  {r['drop']*100:>7.2f}%")
    print("=" * 50)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    parser = argparse.ArgumentParser(description="PGD adversarial attack on Oxford 102 ResNet50")
    parser.add_argument("--ckpt",        default=os.path.join(ckpt_dir, "resnet50_best.pth"))
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--workers",     type=int,   default=2)
    parser.add_argument("--train_pct",   type=float, default=0.80)
    parser.add_argument("--valid_pct",   type=float, default=0.10)
    parser.add_argument("--niter",       type=int,   default=20,
                        help="Number of PGD iterations")
    parser.add_argument("--epsilon",     type=float, nargs="+",
                        default=[0.01, 0.03, 0.05, 0.1],
                        help="L-inf budget(s) — pass multiple for an epsilon sweep")
    parser.add_argument("--stepsize",    type=float, default=0.005,
                        help="PGD step size")
    parser.add_argument("--randinit",    action="store_true", default=True,
                        help="Random initialisation inside epsilon ball")
    parser.add_argument("--results_out", default="pgd_results.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_pgd_experiment(args)
