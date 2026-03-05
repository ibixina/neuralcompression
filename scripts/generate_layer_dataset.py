import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.teacher import TeacherNet
from utils.checkpoint import load_checkpoint


def build_loader(data_root, batch_size, num_workers):
    t = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR100(root=data_root, train=True, download=True, transform=t)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_loader(args.data_root, args.batch_size, args.num_workers)

    model = TeacherNet(num_classes=100, channels=args.channels).to(device)
    load_checkpoint(args.teacher_ckpt, model, map_location=device)
    model.eval()

    layer_inputs = None
    layer_outputs = None

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, layer_io = model.forward_with_activations(x)
            if layer_inputs is None:
                layer_inputs = [[] for _ in range(len(layer_io))]
                layer_outputs = [[] for _ in range(len(layer_io))]

            for i, (x_in, y_out) in enumerate(layer_io):
                layer_inputs[i].append(x_in.cpu())
                layer_outputs[i].append(y_out.cpu())

    os.makedirs(args.out_dir, exist_ok=True)

    layer_files = []
    max_in_c = 0
    max_out_c = 0
    max_h = 0
    max_w = 0

    for i in range(len(layer_inputs)):
        x = torch.cat(layer_inputs[i], dim=0)
        y = torch.cat(layer_outputs[i], dim=0)
        max_in_c = max(max_in_c, x.shape[1])
        max_out_c = max(max_out_c, y.shape[1])
        max_h = max(max_h, x.shape[2], y.shape[2])
        max_w = max(max_w, x.shape[3], y.shape[3])
        rel_path = f"layer_{i}.pt"
        torch.save({"x": x, "y": y}, os.path.join(args.out_dir, rel_path))
        layer_files.append(rel_path)

    manifest = {
        "num_layers": len(layer_files),
        "max_in_channels": max_in_c,
        "max_out_channels": max_out_c,
        "max_h": max_h,
        "max_w": max_w,
        "layer_files": layer_files,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out-dir", default="./distill_data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--teacher-ckpt", required=True)
    parser.add_argument("--channels", type=int, nargs="+", default=[64, 128, 256, 512])
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
