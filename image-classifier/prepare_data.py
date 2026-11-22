"""Download ready-made public datasets and export them as dataset/<class>/*.jpg

Examples:
    python prepare_data.py --list
    python prepare_data.py --datasets pets flowers eurosat --per-class 200
    python prepare_data.py --datasets cifar10 cifar100 --per-class 150 --out dataset

Then train:
    python train.py --data dataset
Raw downloads are cached in raw_data/ (safe to delete afterwards).
"""
import argparse
import random
import re
from collections import Counter
from pathlib import Path

from torchvision import datasets as tvd

CAT_BREEDS = {"abyssinian", "bengal", "birman", "bombay", "british shorthair", "egyptian mau",
              "maine coon", "persian", "ragdoll", "russian blue", "siamese", "sphynx"}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def default_name(label, classes):
    return slug(classes[label])


def pets_species(label, classes):
    return "cat" if classes[label].lower() in CAT_BREEDS else "dog"


def single(name):
    return lambda label, classes: name


# key: (description, approx download, factory(root), label->class-name function)
REGISTRY = {
    "pets": ("Oxford-IIIT Pets grouped into 'cat' and 'dog' (~7k images)", "800 MB",
             lambda r: tvd.OxfordIIITPet(r, split="trainval", download=True), pets_species),
    "pet_breeds": ("Oxford-IIIT Pets, 37 individual breeds", "800 MB",
                   lambda r: tvd.OxfordIIITPet(r, split="trainval", download=True), default_name),
    "flowers": ("Flowers102 merged into one class 'flower' (~6k images)", "330 MB",
                lambda r: tvd.Flowers102(r, split="test", download=True), single("flower")),
    "eurosat": ("EuroSAT satellite land use, 10 classes (forest, river, highway...)", "90 MB",
                lambda r: tvd.EuroSAT(r, download=True), default_name),
    "cifar10": ("CIFAR-10: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck (32x32)", "170 MB",
                lambda r: tvd.CIFAR10(r, train=True, download=True), default_name),
    "cifar100": ("CIFAR-100: 100 everyday classes (32x32)", "170 MB",
                 lambda r: tvd.CIFAR100(r, train=True, download=True), default_name),
    "stl10": ("STL-10: 10 classes at 96x96, sharper than CIFAR", "2.5 GB",
              lambda r: tvd.STL10(r, split="train", download=True), default_name),
    "food101": ("Food-101: 101 food types", "5 GB",
                lambda r: tvd.Food101(r, split="train", download=True), default_name),
}


def get_labels(ds):
    for attr in ("targets", "_labels", "labels"):
        if hasattr(ds, attr):
            return [int(x) for x in getattr(ds, attr)]
    raise RuntimeError("Cannot read labels for this dataset type.")


def export(key, per_class, out_dir, raw_dir, rng):
    desc, _, factory, namer = REGISTRY[key]
    print(f"\n[{key}] downloading/loading ({desc}) ...")
    ds = factory(str(raw_dir))
    classes = list(getattr(ds, "classes", []) or [])
    labels = get_labels(ds)

    groups = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(namer(lab, classes) if classes or key == "flowers" else str(lab), []).append(idx)

    for cname, idxs in groups.items():
        rng.shuffle(idxs)
        folder = out_dir / cname
        folder.mkdir(parents=True, exist_ok=True)
        saved = 0
        for idx in idxs[:per_class]:
            img = ds[idx][0].convert("RGB")
            img.save(folder / f"{key}_{idx:06d}.jpg", quality=92)
            saved += 1
        print(f"  {cname}: {saved} images")


def summary(out_dir):
    counts = Counter()
    for f in out_dir.rglob("*"):
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            counts[f.parent.name] += 1
    print(f"\nDataset ready in '{out_dir}': {len(counts)} classes, {sum(counts.values())} images")
    small = [c for c, n in counts.items() if n < 50]
    if small:
        print("Classes with fewer than 50 images (add more):", ", ".join(small))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show available datasets and exit")
    ap.add_argument("--datasets", nargs="+", choices=sorted(REGISTRY), help="which datasets to export")
    ap.add_argument("--per-class", type=int, default=200, help="max images per class")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--raw", default="raw_data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.list or not args.datasets:
        print("Available datasets:\n")
        for k, (desc, size, _, _) in REGISTRY.items():
            print(f"  {k:<11} {size:>7}  {desc}")
        print("\nUsage: python prepare_data.py --datasets pets flowers eurosat --per-class 200")
        return

    rng = random.Random(args.seed)
    out_dir, raw_dir = Path(args.out), Path(args.raw)
    out_dir.mkdir(exist_ok=True)
    for key in args.datasets:
        export(key, args.per_class, out_dir, raw_dir, rng)
    summary(out_dir)


if __name__ == "__main__":
    main()
