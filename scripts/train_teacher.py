import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.teacher import TeacherNet
from utils.checkpoint import save_checkpoint


def build_loaders(data_root, batch_size, num_workers):
    train_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_t = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_t)
    test_ds = datasets.CIFAR100(root=data_root, train=False, download=True, transform=test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_loaders(args.data_root, args.batch_size, args.num_workers)

    model = TeacherNet(num_classes=100, channels=args.channels).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        scheduler.step()
        acc = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(os.path.join(args.out_dir, "teacher_best.pt"), model, optimizer, epoch, {"acc": acc})
        if epoch % args.save_every == 0:
            save_checkpoint(os.path.join(args.out_dir, f"teacher_epoch_{epoch}.pt"), model, optimizer, epoch, {"acc": acc})
        print(f"epoch={epoch} acc={acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out-dir", default="./runs/teacher")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--channels", type=int, nargs="+", default=[64, 128, 256, 512])
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
