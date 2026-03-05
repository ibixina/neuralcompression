import torch


def save_checkpoint(path, model, optimizer=None, epoch=None, metrics=None):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
