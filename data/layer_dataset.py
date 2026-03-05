import json
import os
import torch
from torch.utils.data import Dataset


class LayerDataset(Dataset):
    def __init__(self, root):
        self.root = root
        manifest_path = os.path.join(root, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.num_layers = self.manifest["num_layers"]
        self.max_in_channels = self.manifest["max_in_channels"]
        self.max_out_channels = self.manifest["max_out_channels"]
        self.max_h = self.manifest["max_h"]
        self.max_w = self.manifest["max_w"]
        self.layer_files = self.manifest["layer_files"]

        self.layers = []
        self.index = []
        for layer_id, rel_path in enumerate(self.layer_files):
            data = torch.load(os.path.join(root, rel_path), map_location="cpu")
            x = data["x"]
            y = data["y"]
            self.layers.append({"x": x, "y": y})
            for i in range(x.shape[0]):
                self.index.append((layer_id, i))

        self.layer_sizes = [layer["x"].shape[0] for layer in self.layers]
        self.weights = self._build_weights()

    def _build_weights(self):
        weights = []
        for layer_id, idx in self.index:
            weights.append(1.0 / float(self.layer_sizes[layer_id]))
        return torch.tensor(weights, dtype=torch.float)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        layer_id, local_idx = self.index[idx]
        x = self.layers[layer_id]["x"][local_idx]
        y = self.layers[layer_id]["y"][local_idx]
        orig_out_c = y.shape[0]
        orig_h, orig_w = y.shape[1], y.shape[2]

        # Pad input channels to max_in_channels
        if x.shape[0] < self.max_in_channels:
            pad_c = self.max_in_channels - x.shape[0]
            pad = torch.zeros((pad_c, x.shape[1], x.shape[2]), dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)

        if x.shape[1] < self.max_h or x.shape[2] < self.max_w:
            pad_h = self.max_h - x.shape[1]
            pad_w = self.max_w - x.shape[2]
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

        # Append one-hot layer id as channels
        one_hot = torch.zeros((self.num_layers, x.shape[1], x.shape[2]), dtype=x.dtype)
        one_hot[layer_id, :, :] = 1.0
        x = torch.cat([x, one_hot], dim=0)

        if y.shape[1] < self.max_h or y.shape[2] < self.max_w:
            pad_h = self.max_h - y.shape[1]
            pad_w = self.max_w - y.shape[2]
            y = torch.nn.functional.pad(y, (0, pad_w, 0, pad_h))

        # Pad output channels to max_out_channels and build channel mask
        if y.shape[0] < self.max_out_channels:
            pad_c = self.max_out_channels - y.shape[0]
            pad = torch.zeros((pad_c, y.shape[1], y.shape[2]), dtype=y.dtype)
            y = torch.cat([y, pad], dim=0)

        channel_mask = torch.zeros((self.max_out_channels, 1, 1), dtype=y.dtype)
        channel_mask[:orig_out_c] = 1.0
        spatial_mask = torch.zeros((1, self.max_h, self.max_w), dtype=y.dtype)
        spatial_mask[:, :orig_h, :orig_w] = 1.0

        mask = channel_mask * spatial_mask
        return x, y, mask, layer_id
