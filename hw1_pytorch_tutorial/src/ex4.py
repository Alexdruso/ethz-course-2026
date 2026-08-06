# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: hw0-pytorch
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 4: Transformers on Images + GLU-MLP Ablations (ViT × GLU Variants)
#
# ## In this exercise you will combine two influential ideas:
#
# Vision Transformers (ViT) from “An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale” (Dosovitskiy et al., 2020) https://arxiv.org/pdf/2010.11929:
# ViT shows that you can treat an image like a sequence of tokens by splitting it into non-overlapping patches (e.g. 16×16 in the paper), embedding each patch into a vector, adding positional information, and then applying standard Transformer blocks for classification.
#
# Gated MLPs (GLU variants) from “GLU Variants Improve Transformer” (Shazeer, 2020) https://arxiv.org/pdf/2002.05202:
# Shazeer proposes replacing the standard Transformer feed-forward layer (FFN/MLP) with gated linear unit (GLU) variants such as GEGLU and SwiGLU, which often improves training dynamics and final performance under comparable compute/parameter budgets.
#
# ## What you will do
#
# You will implement a tiny ViT-style classifier for MNIST, then run a controlled ablation where you replace the MLP inside each Transformer block:
#
# Baseline FFN (GELU):
# Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)
#
# GLU-family MLPs (choose at least two and justify):
#
# GEGLU, SwiGLU, other activation functions
#
# Your goal is to evaluate whether these GLU variants change:
#
# - convergence speed (loss vs steps),
#
# - final test accuracy,
#
# - and/or stability across runs.
#
# ## Key ViT concepts you will implement
#
# - To convert MNIST images into Transformer tokens, you will:
#   Patchify each 28×28 image into non-overlapping P×P patches.
#   If P=4, then you get a 7×7 patch grid → 49 tokens per image.
#
# - Embed patches with a linear layer: patch vectors → d_model.
#
# - Add positional embeddings so the model knows where each patch came from.
#
# - Apply n_layers Transformer encoder blocks.
#
# - Pool token features (e.g., mean pooling) and project to 10 classes.
#
# ## Key GLU concept you will implement
#
# GLU-style MLPs replace a standard FFN with a gating mechanism:
# compute two projections a and b, apply a nonlinearity to a (variant-dependent), multiply elementwise: act(a) * b, project back to d_model.
# To keep the comparison fair, use the 2/3 width rule from Shazeer.
#
# What we provide vs what you implement
#
# ### We provide:
#
# - MNIST loading + dataloaders
#
# - a minimal training loop structure (AdamW)
#
# - a suggested small model configuration that runs on CPU
#
# ### You implement:
#
# - patch tokenization (patchify)
#
# - patch embedding + positional embedding strategy
#
# - a pre-LN Transformer encoder block using nn.MultiheadAttention
#
# - at least two GLU MLP variants + one FFN baseline
#
# - metric logging sufficient to support your conclusion
#
# ## Deliverables
#
# Run at least 3 variants (baseline + the activation functions you choose for GLU) and report:
#
# - final and best test accuracy
#
# - number of trainable parameters
#
# - a plot or printed summary of loss/accuracy over epochs
#
# - a short discussion of your results

# %%
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# %%
def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert images to patch tokens."""

    result = x.clone()

    *batches, c, h, w = result.shape

    assert h % patch_size == 0, (
        "patchify supports only non overlapping patches of the same size"
    )

    assert w % patch_size == 0, (
        "patchify supports only non overlapping patches of the same size"
    )

    h_n = h // patch_size
    w_n = w // patch_size

    result = result.reshape(*batches, c, h_n, patch_size, w_n, patch_size)

    result = result.permute(*(_ for _ in range(len(batches))), -4, -2, -5, -3, -1)

    return result.reshape(
        *batches, (h * w) // patch_size**2, c * patch_size * patch_size
    )


# %%
class PatchEmbed(nn.Module):
    def __init__(self, patch_dim: int, d_model: int):
        super().__init__()
        self._patch_dim = patch_dim
        self._d_model = d_model

        self._projection = torch.nn.Linear(in_features=patch_dim, out_features=d_model)

    def forward(self, x_patches: torch.Tensor) -> torch.Tensor:
        return self._projection(x_patches)


class PositionalEmbedding(nn.Module):
    def __init__(self, num_tokens: int, d_model: int):
        super().__init__()
        self._embedding = torch.nn.Embedding(
            embedding_dim=d_model, num_embeddings=num_tokens
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(x.shape[-2], device=x.device)

        embeddigs = self._embedding(indices)

        return x + embeddigs


# %%
# TODO: Define the variants you want to compare against each other from the GLU paper. Justify your choice.
class FeedForward(nn.Module):
    """
    Standard Transformer FFN:
      x -> Linear(d_model->d_ff) -> GELU -> Dropout -> Linear(d_ff->d_model) -> Dropout
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()

        self._project_d_ff = torch.nn.Linear(
            in_features=d_model, out_features=d_ff, bias=False
        )

        self._activation = torch.nn.GELU()

        self._dropout = torch.nn.Dropout(p=dropout)

        self._project_d_model = torch.nn.Linear(
            in_features=d_ff, out_features=d_model, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        result = self._project_d_ff(x)

        result = self._activation(result)

        result = self._dropout(result)

        return self._project_d_model(result)


class GLUFeedForward(nn.Module):
    """GLU-family FFN"""

    def __init__(self, d_model: int, d_ff_gated: int, dropout: float, variant: str):
        super().__init__()

        self._project_d_ff_gated_1 = torch.nn.Linear(
            in_features=d_model, out_features=d_ff_gated, bias=False
        )

        self._project_d_ff_gated_2 = torch.nn.Linear(
            in_features=d_model, out_features=d_ff_gated, bias=False
        )

        if variant == "glu":
            self._activation = torch.nn.Sigmoid()
        elif variant == "bilinear":
            self._activation = torch.nn.Identity()
        elif variant == "reglu":
            self._activation = torch.nn.ReLU()
        elif variant == "geglu":
            self._activation = torch.nn.GELU()
        elif variant == "swiglu":
            self._activation = torch.nn.SiLU()
        else:
            raise ValueError(f"{variant=} not supported")

        self._dropout = torch.nn.Dropout(p=dropout)

        self._project_d_model = torch.nn.Linear(
            in_features=d_ff_gated, out_features=d_model, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        projection_1 = self._project_d_ff_gated_1(x)

        projection_2 = self._project_d_ff_gated_2(x)

        result = self._activation(projection_1) * projection_2

        result = self._dropout(result)

        return self._project_d_model(result)


# %%
class TransformerEncoderBlock(nn.Module):
    """
    Pre-LN encoder block:
      x = x + Dropout(SelfAttn(LN(x)))
      x = x + Dropout(MLP(LN(x)))
    """

    def __init__(self, d_model: int, n_heads: int, mlp: nn.Module, dropout: float):
        super().__init__()

        self._layer_norm_1 = torch.nn.LayerNorm(normalized_shape=d_model)

        self._attention = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )

        self._dropout_1 = torch.nn.Dropout(p=dropout)

        self._layer_norm_2 = torch.nn.LayerNorm(normalized_shape=d_model)

        self._mlp = mlp

        self._dropout_2 = torch.nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        intermediate_result = self._layer_norm_1(x)

        intermediate_result, _ = self._attention.forward(
            query=intermediate_result,
            key=intermediate_result,
            value=intermediate_result,
        )

        intermediate_result = self._dropout_1(intermediate_result)

        intermediate_result = intermediate_result + x

        result = self._layer_norm_2(intermediate_result)

        result = self._mlp(result)

        result = self._dropout_2(result)

        return result + intermediate_result


# %%
class TinyViT(nn.Module):
    """
    Tiny ViT-style classifier for MNIST.
    - patchify -> patch embed -> pos embed -> blocks -> mean pool -> head
    """

    def __init__(
        self,
        patch_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        mlp_kind: str,
    ):
        super().__init__()
        assert 28 % patch_size == 0
        grid = 28 // patch_size
        self.num_tokens = grid * grid
        self.patch_size = patch_size
        patch_size * patch_size

        self._project = PatchEmbed(
            patch_dim=patch_size * patch_size, d_model=d_model
        )

        self._positional_embedding = PositionalEmbedding(
            num_tokens=self.num_tokens, d_model=d_model
        )

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp=(
                        FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
                        if mlp_kind == "ffn"
                        else GLUFeedForward(
                            d_model=d_model,
                            d_ff_gated=d_ff,
                            dropout=dropout,
                            variant=mlp_kind,
                        )
                    ),
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.prediction = torch.nn.Linear(
            in_features=self.num_tokens * d_model,
            out_features=10,  # mnist
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        result = patchify(x=x, patch_size=self.patch_size)

        result = self._project(result)

        result = self._positional_embedding(result)

        for block in self.blocks:
            result = block(result)

        result = result.flatten(start_dim=1)

        logits = self.prediction(result)

        return logits


# %%
@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    batch_size: int = 128
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 0.01
    device: str = "cuda"
    log_every: int = 50  # steps between in-epoch progress lines


# %%
def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


def grad_global_norm(model: nn.Module) -> float:
    """L2 norm of the concatenated gradient — a cheap stability probe."""

    squares = [
        p.grad.detach().pow(2).sum()
        for p in model.parameters()
        if p.grad is not None
    ]

    if not squares:
        return 0.0

    return torch.stack(squares).sum().sqrt().item()


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, seconds = divmod(seconds, 60)

    return f"{int(minutes)}m{seconds:04.1f}s"


# %%
def train_one_run(
    mlp_kind: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
) -> dict:
    model.to(cfg.device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    total_params, trainable_params = count_parameters(model)
    steps_per_epoch = len(train_loader)

    print(f"  params      : {total_params:,} total | {trainable_params:,} trainable")
    print(
        f"  optim       : AdamW lr={cfg.lr:.1e} wd={cfg.weight_decay} "
        f"| batch={cfg.batch_size} | epochs={cfg.epochs} | device={cfg.device}"
    )
    print(
        f"  data        : {steps_per_epoch} train batches "
        f"| {len(test_loader)} test batches"
    )
    print("  " + "-" * 74)

    # Per-step series (length == epochs * steps_per_epoch)
    train_losses: list[float] = []
    grad_norms: list[float] = []

    # Per-epoch series (length == epochs)
    epoch_train_losses: list[float] = []
    epoch_train_accs: list[float] = []
    test_losses: list[float] = []
    test_accs: list[float] = []
    epoch_times: list[float] = []

    best_acc = 0.0
    best_epoch = 0
    run_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        # Train loop
        model.train()
        epoch_start = time.perf_counter()
        window_start = epoch_start

        running_loss = 0.0
        running_correct = 0.0
        running_total = 0.0

        window_loss = 0.0
        window_correct = 0.0
        window_total = 0.0

        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            logits = model(xb)
            loss = loss_fn.forward(
                logits,
                yb
            )

            opt.zero_grad()
            loss.backward()
            norm = grad_global_norm(model)
            opt.step()

            batch_loss = loss.item()
            batch_correct = (logits.detach().argmax(dim=-1) == yb).float().sum().item()
            batch_total = yb.numel()

            train_losses.append(batch_loss)
            grad_norms.append(norm)

            running_loss += batch_loss * batch_total
            running_correct += batch_correct
            running_total += batch_total

            window_loss += batch_loss * batch_total
            window_correct += batch_correct
            window_total += batch_total

            step = i + 1
            if step % cfg.log_every == 0 or step == steps_per_epoch:
                elapsed = time.perf_counter() - window_start
                throughput = window_total / max(elapsed, 1e-9)
                print(
                    f"  [{mlp_kind:>7}] e{epoch + 1}/{cfg.epochs} "
                    f"step {step:>4}/{steps_per_epoch} "
                    f"| loss {window_loss / window_total:.4f} "
                    f"(run {running_loss / running_total:.4f}) "
                    f"| acc {window_correct / window_total:.3f} "
                    f"| gnorm {norm:6.3f} "
                    f"| {throughput:7.1f} img/s"
                )
                window_start = time.perf_counter()
                window_loss = 0.0
                window_correct = 0.0
                window_total = 0.0

        train_time = time.perf_counter() - epoch_start
        epoch_train_losses.append(running_loss / running_total)
        epoch_train_accs.append(running_correct / running_total)

        # Evaluation loop NOTE: Should be no need to change this
        model.eval()
        correct = 0.0
        total = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                logits = model(xb)
                test_loss += loss_fn.forward(logits, yb).item() * yb.numel()
                correct += (logits.argmax(dim=-1) == yb).float().sum().item()
                total += yb.numel()

        test_accs.append(correct / total)
        test_losses.append(test_loss / total)
        epoch_times.append(time.perf_counter() - epoch_start)

        is_best = test_accs[-1] > best_acc
        if is_best:
            best_acc = test_accs[-1]
            best_epoch = epoch + 1

        print(
            f"  [{mlp_kind:>7}] epoch {epoch + 1}/{cfg.epochs} done in "
            f"{format_duration(epoch_times[-1])} "
            f"(train {format_duration(train_time)}) "
            f"| train loss {epoch_train_losses[-1]:.4f} acc {epoch_train_accs[-1]:.4f} "
            f"| test loss {test_losses[-1]:.4f} acc {test_accs[-1]:.4f}"
            f"{'  <- best' if is_best else ''}"
        )

    total_time = time.perf_counter() - run_start
    print(
        f"  [{mlp_kind:>7}] finished in {format_duration(total_time)} "
        f"| best test acc {best_acc:.4f} @ epoch {best_epoch} "
        f"| final test acc {test_accs[-1]:.4f} "
        f"| final train loss {epoch_train_losses[-1]:.4f}"
    )

    return {
        "mlp_kind": mlp_kind,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "steps_per_epoch": steps_per_epoch,
        "train_losses": train_losses,
        "grad_norms": grad_norms,
        "epoch_train_losses": epoch_train_losses,
        "epoch_train_accs": epoch_train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs,
        "epoch_times": epoch_times,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "total_time": total_time,
    }


# %%
# Categorical palette: three hues that stay separable under colour-vision
# deficiency; assigned to variants in fixed order, never cycled.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
# Dash patterns as a second channel: GLU variants often sit on top of each
# other, and colour alone would hide the curve underneath.
SERIES_DASHES = ((1, 0), (6, 2), (2, 2))
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


def ema(values: list[float], alpha: float = 0.02) -> list[float]:
    """Exponential moving average — makes per-step curves readable."""

    smoothed: list[float] = []
    average = values[0] if values else 0.0

    for value in values:
        average = alpha * value + (1 - alpha) * average
        smoothed.append(average)

    return smoothed


def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e3e2df", linewidth=0.8)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c4")

    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.title.set_color(INK_PRIMARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def plot_ablation(results: list[dict], cfg: TrainConfig, out_path):
    """Four panels: convergence, accuracy, gradient stability, final error."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, ScalarFormatter

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=SURFACE)
    fig.suptitle(
        "TinyViT on MNIST — FFN vs GLU-family MLPs "
        f"({cfg.epochs} epochs, lr={cfg.lr:.0e}, batch={cfg.batch_size}, "
        f"seed={cfg.seed})",
        color=INK_PRIMARY,
        fontsize=14,
        fontweight="bold",
    )

    ax_loss, ax_acc = axes[0]
    ax_grad, ax_err = axes[1]

    for ax in (ax_loss, ax_acc, ax_grad, ax_err):
        _style_axis(ax)

    # (a) convergence speed: per-step train loss, raw behind an EMA
    ax_loss.set_title("Training loss vs step", loc="left", fontsize=11)
    ax_loss.set_xlabel("optimizer step")
    ax_loss.set_ylabel("cross-entropy (log scale)")
    ax_loss.set_yscale("log")

    # (b) final performance: test accuracy per epoch
    ax_acc.set_title("Test accuracy vs epoch", loc="left", fontsize=11)
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")

    # (c) stability: how large the gradients stay
    ax_grad.set_title("Gradient global norm (EMA) vs step", loc="left", fontsize=11)
    ax_grad.set_xlabel("optimizer step")
    ax_grad.set_ylabel("L2 norm")

    # (d) headline: final test error, zero-based so the bars are honest
    ax_err.set_title("Final test error rate", loc="left", fontsize=11)
    ax_err.set_ylabel("error (%)")

    epochs_axis = list(range(1, cfg.epochs + 1))

    for index, run in enumerate(results):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        dashes = SERIES_DASHES[index % len(SERIES_DASHES)]
        label = run["mlp_kind"]
        steps = range(1, len(run["train_losses"]) + 1)

        ax_loss.plot(steps, run["train_losses"], color=color, alpha=0.15, linewidth=0.7)
        ax_loss.plot(
            steps,
            ema(run["train_losses"]),
            color=color,
            linewidth=2,
            dashes=dashes,
            label=label,
        )

        ax_acc.plot(
            epochs_axis,
            run["test_accs"],
            color=color,
            linewidth=2,
            dashes=dashes,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=label,
        )
        ax_acc.annotate(
            f"{run['test_accs'][-1]:.3f}",
            xy=(epochs_axis[-1], run["test_accs"][-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=9,
            va="center",
        )

        ax_grad.plot(
            steps,
            ema(run["grad_norms"]),
            color=color,
            linewidth=2,
            dashes=dashes,
            label=label,
        )

    # Plain decimals on the log axis (0.1, 0.2, 0.5, 1, 2) instead of 10^-1.
    ax_loss.yaxis.set_major_formatter(ScalarFormatter())
    ax_loss.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0)))
    ax_loss.yaxis.set_minor_formatter(ScalarFormatter())

    ax_acc.set_xticks(epochs_axis)
    ax_acc.margins(x=0.08)

    labels = [run["mlp_kind"] for run in results]
    errors = [100 * (1 - run["test_accs"][-1]) for run in results]
    colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(results))]

    bars = ax_err.bar(labels, errors, color=colors, width=0.55)
    ax_err.set_ylim(0, max(errors) * 1.35)

    for bar, run, error in zip(bars, results, errors):
        ax_err.annotate(
            f"{error:.2f}%\n{run['total_params']:,} params",
            xy=(bar.get_x() + bar.get_width() / 2, error),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            color=INK_SECONDARY,
            fontsize=9,
        )

    for ax in (ax_loss, ax_acc, ax_grad):
        legend = ax.legend(frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    return out_path


def print_summary(results: list[dict]) -> None:
    header = (
        f"{'variant':>8} | {'d_ff':>5} | {'params':>9} | {'final acc':>9} | "
        f"{'best acc':>9} | {'best@':>5} | {'test loss':>9} | {'s/epoch':>8}"
    )

    print("\n" + "=" * len(header))
    print("SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for run in results:
        seconds_per_epoch = sum(run["epoch_times"]) / len(run["epoch_times"])
        print(
            f"{run['mlp_kind']:>8} | {run.get('d_ff', 0):>5} | "
            f"{run['total_params']:>9,} | {run['test_accs'][-1]:>9.4f} | "
            f"{run['best_acc']:>9.4f} | {run['best_epoch']:>5} | "
            f"{run['test_losses'][-1]:>9.4f} | {seconds_per_epoch:>8.1f}"
        )

    print("=" * len(header))

    winner = max(results, key=lambda run: run["best_acc"])
    print(
        f"best variant: {winner['mlp_kind']} "
        f"(best test acc {winner['best_acc']:.4f} @ epoch {winner['best_epoch']})"
    )


if __name__ == "__main__":
    # %%
    from pathlib import Path

    cfg = TrainConfig(seed=0, batch_size=128, epochs=5, lr=3e-4, weight_decay=0.01)

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        print("cuda requested but unavailable — falling back to cpu")
        cfg = TrainConfig(
            seed=cfg.seed,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            device="cpu",
            log_every=cfg.log_every,
        )

    tfm = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    patch_size = 4
    d_model = 64
    n_heads = 4
    n_layers = 2
    d_ff = 256
    dropout = 0.1

    # Baseline + two GLU variants. GEGLU and SwiGLU are the two Shazeer reports
    # as strongest; together they isolate the gate nonlinearity (GELU vs SiLU)
    # while holding the gating structure fixed.
    runs = ["ffn", "geglu", "swiglu"]
    results = []

    for run_index, kind in enumerate(runs):
        # Shazeer's 2/3 width rule: a gated MLP carries three weight matrices
        # instead of two, so shrink d_ff to keep the parameter budget fair.
        run_d_ff = d_ff if kind == "ffn" else int(round(2 * d_ff / 3 / 8)) * 8

        # Same seed per run => identical init stream and shuffling order, so any
        # difference in the curves comes from the MLP and not from luck.
        torch.manual_seed(cfg.seed)

        model = TinyViT(
            patch_size=patch_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=run_d_ff,
            dropout=dropout,
            mlp_kind=kind,
        )

        print("\n" + "=" * 78)
        print(f"Run {run_index + 1}/{len(runs)} · mlp={kind}")
        print("=" * 78)
        print(
            f"  arch        : patch={patch_size} tokens={model.num_tokens} "
            f"d_model={d_model} heads={n_heads} layers={n_layers} "
            f"d_ff={run_d_ff} dropout={dropout}"
        )

        out = train_one_run(kind, model, train_loader, test_loader, cfg)
        out["d_ff"] = run_d_ff
        results.append(out)

    print_summary(results)

    figure_dir = Path(__file__).resolve().parent / "figures"
    figure_dir.mkdir(exist_ok=True)
    figure_path = plot_ablation(results, cfg, figure_dir / "ex4_glu_ablation.png")
    print(f"\nsaved plot -> {figure_path}")
