"""Train the CIFAR-10 evaluation classifier.

The classifier is an ImageNet-initialized ResNet-18 fine-tuned for 50 epochs.
All random seeds are fixed for reproducibility. The best checkpoint is saved
under ``models/``.

Usage:
  python train_classifier.py [--seed 42]
"""

import argparse
import os
import random
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model.to(device)


def main(args):
    set_deterministic(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    data_root = os.path.join(SCRIPT_DIR, "data")
    train_ds = datasets.CIFAR10(root=data_root, train=True, transform=transform_train, download=True)
    test_ds = datasets.CIFAR10(root=data_root, train=False, transform=transform_test)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss()

    save_path = os.path.join(SCRIPT_DIR, "models", "cifar10_classifier.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(50):
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        train_acc = 100.0 * correct / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                correct += outputs.argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        test_acc = 100.0 * correct / total
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch + 1}/50: train={train_acc:.2f}% test={test_acc:.2f}% [{elapsed:.0f}s]")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print("  -> Saved (best)")

    print(f"\nBest test accuracy: {best_acc:.2f}% ({time.time() - t0:.0f}s total)")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
