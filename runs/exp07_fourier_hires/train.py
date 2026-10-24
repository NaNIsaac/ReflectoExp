from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


class LogMSELoss(nn.Module):
    """
    Loss for the wide dynamic range of XRR data.
    If the input is already on a log scale, high angles can be weighted more.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        # plain MSE: the parameters are already normalized, which handles their scale differences
        # weight a specific parameter (e.g. thickness) here if wanted
        return self.mse(preds, targets)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str | Path,
        lr: float = 2e-4,  # a slightly lower LR is stable for the Fourier feature model
        weight_decay: float = 1e-4,
        patience: int = 20,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.checkpoint_dir = Path(save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Loss Function
        self.criterion = nn.MSELoss()

        # Scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=7, factor=0.5)

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.early_stop_patience = patience

        self.history = {"train": [], "val": [], "lr": []}

        # self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None
        self.scaler = None
        print(f"🚀 Trainer initialized on {self.device}")
        print(f"   - Augmentation: {'Enabled' if getattr(train_loader.dataset, 'augment', False) else 'Disabled'}")

    def _train_epoch(self, epoch: int, total_epochs: int) -> float:
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)

        # Debug mode
        # torch.autograd.set_detect_anomaly(True)

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # # Debug mode
            # if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            #     print("🚨 [FATAL] Input data contains NaN or Inf!")
            #     continue
            # if torch.isnan(targets).any() or torch.isinf(targets).any():
            #     print("🚨 [FATAL] Target data contains NaN or Inf!")
            #     continue

            self.optimizer.zero_grad(set_to_none=True)

            # [Physics-Informed Training]
            if self.scaler:
                with torch.amp.autocast("cuda"):
                    preds = self.model(inputs)
                    loss = self.criterion(preds, targets)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(inputs)
                loss = self.criterion(preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        running_loss = 0.0
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            preds = self.model(inputs)
            loss = self.criterion(preds, targets)
            running_loss += loss.item()
        return running_loss / len(self.val_loader)

    def train(self, epochs: int, resume_from: str | Path | None = None):
        start_epoch = self._load_checkpoint(resume_from) if resume_from else 1

        train_size = len(self.train_loader.dataset)
        print(f"\n[Training Start] Total Samples: {train_size} (Expanded)")
        print("-" * 70)

        for epoch in range(start_epoch, epochs + 1):
            train_loss = self._train_epoch(epoch, epochs)
            val_loss = self._validate()

            self.history["train"].append(train_loss)
            self.history["val"].append(val_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"E{epoch:03d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.1e}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best.pt", epoch, val_loss)
                print("  >>> Best Model Saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    print(f"\n🛑 Early stopping at epoch {epoch}")
                    break

            self._save_checkpoint("last.pt", epoch, val_loss)

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        path = self.checkpoint_dir / filename
        model_config = getattr(self.model, "config", {})

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": self.best_val_loss,
                "patience_counter": self.patience_counter,
                "history": self.history,
                "config": {"model_args": model_config},
            },
            path,
        )

    def _load_checkpoint(self, filepath: str | Path) -> int:
        filepath = Path(filepath)
        if not filepath.exists():
            return 1
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint.get("history", self.history)

        # val_loss in last.pt is the last epoch's value, not the minimum. Using it would loosen the criterion
        # after a resume, and a worse model would overwrite best.pt.
        # Prefer the saved best_val_loss; if absent (older ckpt), recover it from the history.
        best = checkpoint.get("best_val_loss")
        if best is None:
            val_hist = checkpoint.get("history", {}).get("val", [])
            best = min(val_hist) if val_hist else checkpoint.get("val_loss", float("inf"))
        self.best_val_loss = best
        self.patience_counter = checkpoint.get("patience_counter", 0)

        print(f"🔄 Resuming from Epoch {checkpoint['epoch'] + 1} (best val = {self.best_val_loss:.6f}, patience {self.patience_counter})")
        return checkpoint["epoch"] + 1
