# Training guide

## Option A: ready-made datasets (automatic)
    python prepare_data.py --list
    python prepare_data.py --datasets pets flowers eurosat cifar10 --per-class 200
    python train.py --data dataset --epochs 10
    python app.py

## Option B: your own / Kaggle images
Put images in dataset/<class_name>/ (one folder per category), then run train.py and app.py.

Mix A and B freely: everything ends up in dataset/<class_name>/.
To go back to the ImageNet model, delete or rename models/custom_model.pt.
