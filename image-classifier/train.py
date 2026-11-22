"""Fine-tune MobileNetV2 on your own images.

Dataset layout (one folder per category, any number of categories):

    dataset/
      cat/   img1.jpg img2.png ...
      dog/   ...
      car/   ...

or, if you want to control the validation set yourself:

    dataset/train/<class>/...   and   dataset/val/<class>/...

Run:
    python train.py --data dataset --epochs 10
Output:
    models/custom_model.pt   (picked up automatically by classifier.py)
"""
import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

WEIGHTS = models.MobileNet_V2_Weights.IMAGENET1K_V2


def safe_loader(path):
    """Skip unreadable/corrupt files instead of crashing the whole run."""
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        print(f"  warning: could not read {path}, using blank image")
        return Image.new("RGB", (224, 224), (128, 128, 128))


def build_loaders(data_dir, batch, val_fraction, seed):
    meta = WEIGHTS.transforms()
    norm = transforms.Normalize(meta.mean, meta.std)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), norm,
    ])

    root = Path(data_dir)
    if (root / "train").is_dir() and (root / "val").is_dir():
        train_ds = datasets.ImageFolder(root / "train", train_tf, loader=safe_loader)
        val_ds = datasets.ImageFolder(root / "val", val_tf, loader=safe_loader)
        if train_ds.classes != val_ds.classes:
            raise SystemExit("train/ and val/ must contain the same class folders.")
    else:
        # Single folder: split each class 80/20 so every class appears in both sets.
        train_full = datasets.ImageFolder(root, train_tf, loader=safe_loader)
        val_full = datasets.ImageFolder(root, val_tf, loader=safe_loader)
        rng = random.Random(seed)
        by_class = {}
        for idx, (_, label) in enumerate(train_full.samples):
            by_class.setdefault(label, []).append(idx)
        train_idx, val_idx = [], []
        for idxs in by_class.values():
            rng.shuffle(idxs)
            n_val = max(1, int(len(idxs) * val_fraction)) if len(idxs) > 1 else 0
            val_idx += idxs[:n_val]
            train_idx += idxs[n_val:]
        train_ds, val_ds = Subset(train_full, train_idx), Subset(val_full, val_idx)
        train_ds.classes = train_full.classes

    classes = train_ds.classes
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)
    return train_dl, val_dl, classes, len(train_ds), len(val_ds)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset", help="dataset folder")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--finetune", action="store_true",
                    help="also train the backbone (slower, better with 500+ images/class)")
    ap.add_argument("--out", default="models/custom_model.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl, classes, n_train, n_val = build_loaders(
        args.data, args.batch, args.val_fraction, args.seed)
    print(f"{len(classes)} classes, {n_train} train images, {n_val} validation images, device={device}")
    print("classes:", ", ".join(classes))

    model = models.mobilenet_v2(weights=WEIGHTS)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    for p in model.features.parameters():
        p.requires_grad = args.finetune
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    lr = args.lr * (0.1 if args.finetune else 1.0)
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best = -1.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        if not args.finetune:
            model.features.eval()  # keep frozen BatchNorm stats stable
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
        acc = evaluate(model, val_dl, device)
        print(f"epoch {epoch}/{args.epochs}  loss {running / n_train:.4f}  val accuracy {acc:.1%}")
        if acc >= best:
            best = acc
            torch.save({"arch": "mobilenet_v2", "classes": classes,
                        "state_dict": model.state_dict()}, args.out)
    print(f"Done. Best validation accuracy {best:.1%}. Saved to {args.out}")


if __name__ == "__main__":
    main()
