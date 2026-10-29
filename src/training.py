"""Subject-aware raw-signal CNN training and prediction."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from .evaluation import group_splits
from .models import build_cnn


def train_cnn(train, test, out_dir="results", seed=42, smoke=False):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y, groups = train["y"], train["subjects"]
    fit_idx, valid_idx = next(group_splits(y, groups, n_splits=5, random_state=seed))
    # Standardize each of the nine sensor channels using only the fit participants.
    mean = train["signals"][fit_idx].mean(axis=(0, 1), keepdims=True)
    std = train["signals"][fit_idx].std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    x_fit = ((train["signals"][fit_idx] - mean) / std).astype(np.float32)
    x_valid = ((train["signals"][valid_idx] - mean) / std).astype(np.float32)
    x_test = ((test["signals"] - mean) / std).astype(np.float32)
    # Dataset uses 0-based class IDs while published labels remain 1..6.
    y_fit = (y[fit_idx] - 1).astype(np.int64); y_valid = (y[valid_idx] - 1).astype(np.int64)
    fit_loader = DataLoader(TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)), batch_size=128, shuffle=True)
    valid_loader = DataLoader(TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid)), batch_size=256, shuffle=False)
    model = build_cnn(x_fit.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    max_epochs, patience = (3, 2) if smoke else (40, 9)
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    best_score = -float("inf"); best_state = None; best_epoch = 0; stale = 0
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.train(); loss_total = 0.0; correct = 0; seen = 0
        for xb, yb in fit_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True); logits = model(xb); loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            loss_total += loss.item() * len(yb); correct += (logits.argmax(1) == yb).sum().item(); seen += len(yb)
        history["loss"].append(loss_total / seen); history["accuracy"].append(correct / seen)
        model.eval(); val_loss = 0.0; val_correct = 0; val_seen = 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device); logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(yb); val_correct += (logits.argmax(1) == yb).sum().item(); val_seen += len(yb)
        val_loss /= val_seen; val_accuracy = val_correct / val_seen
        history["val_loss"].append(val_loss); history["val_accuracy"].append(val_accuracy)
        # Primary selection metric: validation macro F1, not accuracy.
        from sklearn.metrics import f1_score
        with torch.no_grad():
            val_predictions = torch.cat([model(xb.to(device)).argmax(1).cpu() for xb, _ in valid_loader]).numpy() + 1
        val_macro_f1 = f1_score(y_valid + 1, val_predictions, average="macro")
        if val_macro_f1 > best_score:
            best_score = float(val_macro_f1); best_epoch = epoch + 1; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
        print(f"CNN epoch {epoch+1:02d}/{max_epochs}: loss={history['loss'][-1]:.4f}, val_loss={val_loss:.4f}, val_macro_f1={val_macro_f1:.4f}", flush=True)
        if stale >= patience: break
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state); model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x_test), 256):
            xb = torch.from_numpy(x_test[start:start+256]).to(device)
            predictions.extend((model(xb).argmax(1).cpu().numpy() + 1).tolist())
    out = Path(out_dir); (out / "models").mkdir(parents=True, exist_ok=True); (out / "metrics").mkdir(parents=True, exist_ok=True)
    artifact = {"state_dict": model.cpu().state_dict(), "n_channels": x_fit.shape[-1], "mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist(), "best_epoch": best_epoch, "best_validation_macro_f1": best_score, "architecture": "Conv1D(32,64,96)+BatchNorm+MaxPool+GlobalAveragePool+Dropout(0.35)", "seed": seed}
    torch.save(artifact, out / "models" / "cnn_best.pt")
    np.savez_compressed(out / "metrics" / "cnn_history.npz", **{k: np.asarray(v) for k, v in history.items()})
    (out / "metrics" / "cnn_training.json").write_text(json.dumps({"best_epoch": best_epoch, "best_validation_macro_f1": best_score, "training_seconds": training_seconds, "validation_subjects": sorted(map(int, np.unique(groups[valid_idx]))), "fit_subjects": sorted(map(int, np.unique(groups[fit_idx]))), "device": str(device), "epochs": len(history["loss"]), "early_stopping_patience": patience}, indent=2))
    return np.asarray(predictions), training_seconds, history, (mean, std), (x_test, model, x_valid, y_valid + 1)


def perturbation_importance(model_bundle, signals, y, device=None):
    """Absolute occlusion drop in true-class log probability for each channel/time bin."""
    import torch
    model = model_bundle.cpu().eval(); x = torch.from_numpy(np.asarray(signals[:128], dtype=np.float32)); labels = torch.from_numpy(np.asarray(y[:128] - 1, dtype=np.int64))
    with torch.no_grad():
        base = torch.log_softmax(model(x), dim=1).gather(1, labels[:, None]).mean().item()
        scores = np.zeros((x.shape[2], 8), dtype=float)
        for channel in range(x.shape[2]):
            for bin_idx in range(8):
                start, end = bin_idx * 16, (bin_idx + 1) * 16
                occluded = x.clone(); occluded[:, start:end, channel] = 0
                score = torch.log_softmax(model(occluded), dim=1).gather(1, labels[:, None]).mean().item()
                scores[channel, bin_idx] = base - score
    return scores
