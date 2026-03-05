import argparse
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.layer_dataset import LayerDataset
from models.student import StudentNet
from models.teacher import TeacherNet
from utils.checkpoint import save_checkpoint
from utils.metrics import masked_mse


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_width_for_ratio(base_width, depth, in_ch, num_layers, max_out_ch, target_ratio, teacher_params):
    lo, hi = 8, 512
    best = None
    for _ in range(20):
        mid = (lo + hi) // 2
        model = StudentNet(in_ch, num_layers, max_out_ch, width=mid, depth=depth)
        params = count_params(model)
        ratio = params / teacher_params
        if best is None or abs(ratio - target_ratio) < abs(best[1] - target_ratio):
            best = (mid, ratio)
        if ratio < target_ratio:
            lo = mid + 1
        else:
            hi = mid - 1
    return best[0], best[1]


def train_one(args, target_ratio, teacher_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LayerDataset(args.data_root)
    sampler = WeightedRandomSampler(dataset.weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)

    width, ratio = find_width_for_ratio(
        args.base_width,
        args.depth,
        dataset.max_in_channels,
        dataset.num_layers,
        dataset.max_out_channels,
        target_ratio,
        teacher_params,
    )

    model = StudentNet(dataset.max_in_channels, dataset.num_layers, dataset.max_out_channels, width=width, depth=args.depth)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    run_name = f"student_{int(target_ratio * 100)}pct_w{width}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y, mask, _layer_id in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = masked_mse(pred, y, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(loader), 1)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(os.path.join(run_dir, f"student_epoch_{epoch}.pt"), model, optimizer, epoch, {"loss": avg_loss, "ratio": ratio})
        print(f"ratio_target={target_ratio:.2f} ratio_actual={ratio:.4f} epoch={epoch} loss={avg_loss:.6f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./distill_data")
    parser.add_argument("--out-dir", default="./runs/student")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--base-width", type=int, default=64)
    parser.add_argument("--targets", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--teacher-channels", type=int, nargs="+", default=[64, 128, 256, 512])
    return parser.parse_args()


def main():
    args = parse_args()
    teacher = TeacherNet(num_classes=100, channels=args.teacher_channels)
    teacher_params = count_params(teacher)
    for target in args.targets:
        train_one(args, target, teacher_params)


if __name__ == "__main__":
    main()
