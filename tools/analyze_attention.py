"""Attention Gate mechanism and focus diagnostics for the trained Attention ResUNet.

This script is intentionally self-contained at the analysis level, but reuses the
model definition and preprocessing from Attention_resunet.py. It produces:

* a representative image plate with input/labels/prediction, alpha maps and
  channel-aggregated encoder/decoder activation energies;
* per-image and per-gate CSV metrics;
* a counterfactual inference report in which alpha is temporarily forced to one.

The counterfactual is a dependency diagnostic, not a retrained no-attention
ablation. A fair performance claim requires training a no-gate model with the
same split, seed, optimizer and schedule.
"""

from __future__ import annotations

import contextlib
import csv
import io
import math
import os
import re
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF


# -----------------------------------------------------------------------------
# Figure contract
# -----------------------------------------------------------------------------
# Core conclusion: the trained gates should assign higher spatial alpha to
# useful oil-containing regions and reduce irrelevant skip activations, while
# preserving the decoder's semantic feature stream.
# Archetype: image plate + quant, with a mechanism-focused image plate.
# Backend: Python/matplotlib only.
# Output: editable SVG/PDF plus 600-dpi TIFF and a PNG preview.
# Statistics: all 10 held-out test images; per-image means; bootstrap 95% CI.
# Reviewer risk: alpha is class-agnostic and is not an oil probability; the
# alpha=1 intervention is not a retrained no-gate baseline.


# Mandatory editable-text settings from the Python figure workflow.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(os.environ.get("OILSEG_TEST_IMAGES", ROOT / "data/test/images"))
LABEL_DIR = Path(os.environ.get("OILSEG_TEST_LABELS", ROOT / "data/test/masks"))
WEIGHT_PATH = Path(os.environ.get("OILSEG_WEIGHTS", ROOT / "weights/attention_resunet.pth"))
OUT_DIR = ROOT / "outputs" / "attention_gate_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GATE_NAMES = ("att4", "att3", "att2")
GATE_COLORS = {"att4": "#0F4D92", "att3": "#42949E", "att2": "#9A4D8E"}
GATE_LABELS = {"att4": "att4 (32×32)", "att3": "att3 (64×64)", "att2": "att2 (128×128)"}
CLASS_NAMES = ["rock", "pore", "heterogeneous oil", "throat oil", "wall oil", "dead-end oil"]
COLOR_MATCH_CONFIG = [
    (255, 255, 255, 80),
    (128, 128, 128, 40),
    (255, 0, 255, 50),
    (0, 255, 0, 60),
    (0, 0, 255, 70),
    (255, 255, 0, 50),
]


def natural_key(path: Path):
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


from oilseg import inference as AR
from oilseg.model import AttentionResUNet

def load_model(device: torch.device):
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"Model weight not found: {WEIGHT_PATH}")
    model = AttentionResUNet(num_classes=6).to(device)
    try:
        state = torch.load(WEIGHT_PATH, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def rgb2label(mask_rgb: np.ndarray) -> np.ndarray:
    """Match the RGB-label conversion used by the training script."""

    mask_i = mask_rgb.astype(np.int16)
    label = np.zeros(mask_i.shape[:2], dtype=np.int64)
    for idx, (r, g, b, threshold) in enumerate(COLOR_MATCH_CONFIG):
        delta = mask_i - np.asarray([r, g, b], dtype=np.int16)
        dist = np.sqrt(np.sum(delta.astype(np.int32) ** 2, axis=2))
        label[dist <= threshold] = idx
    return label


def per_class_iou(pred: np.ndarray, target: np.ndarray, num_classes: int = 6) -> np.ndarray:
    values = np.full(num_classes, np.nan, dtype=float)
    for cls in range(num_classes):
        p = pred == cls
        t = target == cls
        union = np.logical_or(p, t).sum()
        if union:
            values[cls] = np.logical_and(p, t).sum() / union
    return values


def image_miou(pred: np.ndarray, target: np.ndarray) -> float:
    values = per_class_iou(pred, target)
    return float(np.nanmean(values)) if np.isfinite(values).any() else 0.0


def upsample_map(tensor: torch.Tensor, height: int, width: int) -> np.ndarray:
    """Convert [1,1,h,w] or [1,h,w] to a full-resolution float array."""

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    array = F.interpolate(
        tensor.float().cpu(), size=(height, width), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    return array


def top_fraction_metrics(alpha: np.ndarray, oil_mask: np.ndarray, fraction: float = 0.10):
    flat_alpha = alpha.ravel()
    flat_oil = oil_mask.ravel().astype(bool)
    k = max(1, int(math.ceil(fraction * flat_alpha.size)))
    selected = np.argpartition(flat_alpha, flat_alpha.size - k)[-k:]
    oil_fraction = float(flat_oil.mean())
    top_precision = float(flat_oil[selected].mean())
    enrichment = top_precision / oil_fraction if oil_fraction > 0 else np.nan
    return oil_fraction, top_precision, enrichment


def bootstrap_ci(values, seed: int = 20260829, n_boot: int = 5000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan)
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[sample_idx].mean(axis=1)
    return tuple(np.percentile(boot_means, [2.5, 97.5]))


def attach_hooks(model, capture: bool = False, force_open: bool = False):
    captures = {name: {} for name in GATE_NAMES}
    handles = []
    for name in GATE_NAMES:
        gate = getattr(model, name)
        if capture:

            def gate_hook(module, args, kwargs, output, gate_name=name):
                # The project calls AttentionBlock with keyword arguments
                # (g=d4, x=e3), so request kwargs explicitly from PyTorch's
                # forward hook API.
                g_input = kwargs.get("g", args[0] if len(args) > 0 else None)
                x_input = kwargs.get("x", args[1] if len(args) > 1 else None)
                if g_input is None or x_input is None:
                    raise RuntimeError(f"Could not capture inputs for {gate_name}")
                captures[gate_name]["g"] = g_input.detach()
                captures[gate_name]["x"] = x_input.detach()
                captures[gate_name]["xg"] = output.detach()

            def alpha_hook(module, inputs, output, gate_name=name):
                captures[gate_name]["alpha"] = output.detach()

            handles.append(gate.register_forward_hook(gate_hook, with_kwargs=True))
            handles.append(gate.psi.register_forward_hook(alpha_hook))
        if force_open:

            def open_hook(module, inputs, output):
                return torch.ones_like(output)

            handles.append(gate.psi.register_forward_hook(open_hook))
    return captures, handles


def infer_one(model, image_path: Path, device: torch.device, capture: bool = False, force_open: bool = False):
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    height, width = image_np.shape[:2]
    label_path = LABEL_DIR / image_path.name
    if not label_path.exists():
        raise FileNotFoundError(f"Missing matching label: {label_path}")
    target = rgb2label(np.asarray(Image.open(label_path).convert("RGB")))
    tensor = TF.to_tensor(image).unsqueeze(0).to(device)
    padded, pad_h, pad_w = AR.pad_to_multiple(tensor, 32)

    captures, handles = attach_hooks(model, capture=capture, force_open=force_open)
    try:
        with torch.inference_mode():
            logits = model(padded)
    finally:
        for handle in handles:
            handle.remove()

    pred_padded = logits.argmax(dim=1).squeeze(0)
    pred = AR.crop_to_original(pred_padded, int(pad_h), int(pad_w)).cpu().numpy()
    return {
        "name": image_path.name,
        "image_np": image_np,
        "target": target,
        "pred": pred,
        "miou": image_miou(pred, target),
        "class_iou": per_class_iou(pred, target),
        "captures": captures,
        "original_shape": (height, width),
    }


def gate_metrics(result):
    height, width = result["original_shape"]
    target = result["target"]
    oil_mask = target >= 2
    rows = []
    for name in GATE_NAMES:
        c = result["captures"][name]
        alpha = np.clip(upsample_map(c["alpha"], height, width), 0.0, 1.0)
        raw_energy = upsample_map(c["x"].abs().mean(dim=1, keepdim=True), height, width)
        gated_energy = upsample_map(c["xg"].abs().mean(dim=1, keepdim=True), height, width)
        decoder_energy = upsample_map(c["g"].abs().mean(dim=1, keepdim=True), height, width)
        background = ~oil_mask
        oil_fraction, top10_precision, top10_enrichment = top_fraction_metrics(alpha, oil_mask)

        def mean_or_nan(array, mask):
            return float(array[mask].mean()) if mask.any() else np.nan

        raw_oil = mean_or_nan(raw_energy, oil_mask)
        raw_bg = mean_or_nan(raw_energy, background)
        gated_oil = mean_or_nan(gated_energy, oil_mask)
        gated_bg = mean_or_nan(gated_energy, background)
        rows.append(
            {
                "image": result["name"],
                "gate": name,
                "gate_resolution": GATE_LABELS[name].split("(")[1].rstrip(")"),
                "oil_fraction": oil_fraction,
                "alpha_oil": mean_or_nan(alpha, oil_mask),
                "alpha_background": mean_or_nan(alpha, background),
                "alpha_delta_oil_minus_background": mean_or_nan(alpha, oil_mask)
                - mean_or_nan(alpha, background),
                "alpha_top10_precision": top10_precision,
                "alpha_top10_enrichment": top10_enrichment,
                "raw_energy_oil": raw_oil,
                "raw_energy_background": raw_bg,
                "gated_energy_oil": gated_oil,
                "gated_energy_background": gated_bg,
                "energy_suppression_oil": 1.0 - gated_oil / raw_oil if raw_oil > 0 else np.nan,
                "energy_suppression_background": 1.0 - gated_bg / raw_bg if raw_bg > 0 else np.nan,
                "raw_energy_mean": float(raw_energy.mean()),
                "gated_energy_mean": float(gated_energy.mean()),
                "decoder_energy_mean": float(decoder_energy.mean()),
                "energy_suppression_global": 1.0 - gated_energy.sum() / raw_energy.sum()
                if raw_energy.sum() > 0
                else np.nan,
            }
        )
    return rows


def aggregate_summary(rows):
    summary = []
    keys = [
        "alpha_oil",
        "alpha_background",
        "alpha_delta_oil_minus_background",
        "alpha_top10_precision",
        "alpha_top10_enrichment",
        "energy_suppression_oil",
        "energy_suppression_background",
        "energy_suppression_global",
    ]
    for gate in GATE_NAMES:
        gate_rows = [row for row in rows if row["gate"] == gate]
        out = {"gate": gate, "n_images": len(gate_rows)}
        for key in keys:
            values = np.asarray([row[key] for row in gate_rows], dtype=float)
            out[f"{key}_mean"] = float(np.nanmean(values))
            out[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else np.nan
            low, high = bootstrap_ci(values, seed=20260829 + list(GATE_NAMES).index(gate))
            out[f"{key}_ci95_low"] = float(low)
            out[f"{key}_ci95_high"] = float(high)
        summary.append(out)
    return summary


def write_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_label(ax, label):
    ax.text(-0.04, 1.03, label, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=10)


def image_axis(ax, title, label=None):
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if label:
        add_label(ax, label)


def oil_rgb(mask, oil_color=(0.90, 0.22, 0.55), background=(0.96, 0.96, 0.94)):
    canvas = np.empty((*mask.shape, 3), dtype=float)
    canvas[...] = background
    canvas[mask] = oil_color
    return canvas


def save_figure(fig: plt.Figure, base_path: Path):
    # Use explicit margins because the two shared colorbars and the two
    # spanning quantitative axes are not fully compatible with tight_layout.
    fig.subplots_adjust(left=0.045, right=0.925, top=0.945, bottom=0.055)
    fig.savefig(str(base_path) + ".svg", bbox_inches="tight")
    fig.savefig(str(base_path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base_path) + ".tiff", dpi=600, bbox_inches="tight")
    fig.savefig(str(base_path) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_mechanism_figure(rep, all_rows, summary, base_path: Path):
    image_np = rep["image_np"]
    target = rep["target"]
    pred = rep["pred"]
    oil_mask = target >= 2
    pred_oil = pred >= 2
    height, width = rep["original_shape"]
    captures = rep["captures"]

    alpha_maps = {name: np.clip(upsample_map(captures[name]["alpha"], height, width), 0, 1) for name in GATE_NAMES}
    raw_maps = {name: upsample_map(captures[name]["x"].abs().mean(dim=1, keepdim=True), height, width) for name in GATE_NAMES}
    gated_maps = {name: upsample_map(captures[name]["xg"].abs().mean(dim=1, keepdim=True), height, width) for name in GATE_NAMES}
    decoder_maps = {name: upsample_map(captures[name]["g"].abs().mean(dim=1, keepdim=True), height, width) for name in GATE_NAMES}
    raw_att2 = raw_maps["att2"]
    gated_att2 = gated_maps["att2"]
    decoder_att2 = decoder_maps["att2"]
    removed_att2 = np.maximum(raw_att2 - gated_att2, 0)

    energy_values = np.concatenate([raw_att2.ravel(), gated_att2.ravel(), decoder_att2.ravel()])
    energy_low, energy_high = np.percentile(energy_values, [1, 99])
    if energy_high <= energy_low:
        energy_high = energy_low + 1e-6
    removed_high = np.percentile(removed_att2, 99.5)
    if removed_high <= 0:
        removed_high = 1e-6

    alpha_cmap = mpl.colormaps["magma"]
    energy_cmap = LinearSegmentedColormap.from_list("energy_gray", ["#F4F3EE", "#B9B9B9", "#3F3F3F"])
    removed_cmap = LinearSegmentedColormap.from_list("removed_teal", ["#F4F3EE", "#B8DEDD", "#2B777B"])

    fig = plt.figure(figsize=(14.2, 11.6), facecolor="white")
    fig.suptitle(
        f"Attention Gate mechanism diagnostic — representative test image {Path(rep['name']).stem}",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1.0, 1.0, 1.0, 0.82], hspace=0.27, wspace=0.18)
    axes = np.empty((4, 4), dtype=object)
    for r in range(4):
        for c in range(4):
            axes[r, c] = fig.add_subplot(gs[r, c])

    # Row 1: establish image, target and output.
    axes[0, 0].imshow(image_np)
    image_axis(axes[0, 0], "input image", "a")
    axes[0, 1].imshow(oil_rgb(oil_mask))
    image_axis(axes[0, 1], "ground-truth oil union", "b")
    axes[0, 2].imshow(oil_rgb(pred_oil, oil_color=(0.18, 0.62, 0.62)))
    axes[0, 2].contour(oil_mask.astype(float), levels=[0.5], colors=["#E84A8A"], linewidths=0.35)
    image_axis(axes[0, 2], f"predicted oil union (mIoU={rep['miou']:.3f})", "c")
    axes[0, 3].imshow(image_np)
    axes[0, 3].imshow(alpha_maps["att2"], cmap=alpha_cmap, vmin=0, vmax=1, alpha=0.55)
    axes[0, 3].contour(oil_mask.astype(float), levels=[0.5], colors=["#FFFFFF"], linewidths=0.45)
    image_axis(axes[0, 3], "att2 α overlaid on input", "d")

    # Row 2: attention maps at all three gate resolutions.
    for col, name in enumerate(GATE_NAMES):
        axes[1, col].imshow(alpha_maps[name], cmap=alpha_cmap, vmin=0, vmax=1)
        axes[1, col].contour(oil_mask.astype(float), levels=[0.5], colors=["#55E0D5"], linewidths=0.40)
        image_axis(axes[1, col], f"{GATE_LABELS[name]}: spatial α", chr(ord("e") + col))
    axes[1, 3].imshow(1.0 - alpha_maps["att2"], cmap=removed_cmap, vmin=0, vmax=1)
    axes[1, 3].contour(oil_mask.astype(float), levels=[0.5], colors=["#55E0D5"], linewidths=0.40)
    image_axis(axes[1, 3], "att2 suppression (1−α)", "h")

    # Row 3: aggregate channel activation energy, avoiding arbitrary channel choice.
    axes[2, 0].imshow(raw_att2, cmap=energy_cmap, vmin=energy_low, vmax=energy_high)
    axes[2, 0].contour(oil_mask.astype(float), levels=[0.5], colors=["#55E0D5"], linewidths=0.40)
    image_axis(axes[2, 0], "att2 raw encoder energy E|x|", "i")
    axes[2, 1].imshow(gated_att2, cmap=energy_cmap, vmin=energy_low, vmax=energy_high)
    axes[2, 1].contour(oil_mask.astype(float), levels=[0.5], colors=["#55E0D5"], linewidths=0.40)
    image_axis(axes[2, 1], "att2 gated encoder energy E|x′|", "j")
    axes[2, 2].imshow(decoder_att2, cmap=energy_cmap, vmin=energy_low, vmax=energy_high)
    axes[2, 2].contour(oil_mask.astype(float), levels=[0.5], colors=["#55E0D5"], linewidths=0.40)
    image_axis(axes[2, 2], "att2 decoder energy E|g|", "k")
    axes[2, 3].imshow(removed_att2, cmap=removed_cmap, vmin=0, vmax=removed_high)
    axes[2, 3].contour(oil_mask.astype(float), levels=[0.5], colors=["#55E0D5"], linewidths=0.40)
    image_axis(axes[2, 3], "removed skip energy E|x|−E|x′|", "l")

    # Quantitative panels: all 10 test images, not just the representative.
    ax_delta = fig.add_subplot(gs[3, 0:2])
    ax_enrich = fig.add_subplot(gs[3, 2:4])
    axes[3, 0].remove()
    axes[3, 1].remove()
    axes[3, 2].remove()
    axes[3, 3].remove()

    x_positions = np.arange(len(GATE_NAMES))
    for idx, gate in enumerate(GATE_NAMES):
        gate_rows = [r for r in all_rows if r["gate"] == gate]
        deltas = np.asarray([r["alpha_delta_oil_minus_background"] for r in gate_rows], dtype=float)
        enrich = np.asarray([r["alpha_top10_enrichment"] for r in gate_rows], dtype=float)
        color = GATE_COLORS[gate]
        jitter = np.linspace(-0.085, 0.085, len(deltas)) if len(deltas) > 1 else np.array([0.0])
        ax_delta.scatter(np.full(len(deltas), idx) + jitter, deltas, s=15, color=color, alpha=0.45, linewidths=0)
        ax_enrich.scatter(np.full(len(enrich), idx) + jitter, enrich, s=15, color=color, alpha=0.45, linewidths=0)
        d_mean = float(np.nanmean(deltas))
        d_low, d_high = bootstrap_ci(deltas, seed=20260829 + idx)
        e_mean = float(np.nanmean(enrich))
        e_low, e_high = bootstrap_ci(enrich, seed=20260929 + idx)
        ax_delta.errorbar(idx, d_mean, yerr=[[d_mean - d_low], [d_high - d_mean]], fmt="o", color=color, ms=5, capsize=3, lw=1.4)
        ax_enrich.errorbar(idx, e_mean, yerr=[[e_mean - e_low], [e_high - e_mean]], fmt="o", color=color, ms=5, capsize=3, lw=1.4)
    ax_delta.axhline(0, color="#767676", lw=0.8, ls="--")
    ax_delta.set_xticks(x_positions, ["att4", "att3", "att2"])
    ax_delta.set_ylabel("mean α(oil) − mean α(non-oil)")
    ax_delta.set_title("attention contrast across all test images (n=10)", fontsize=8, pad=4)
    ax_delta.set_ylim(bottom=min(-0.05, ax_delta.get_ylim()[0]))
    ax_delta.grid(axis="y", color="#D8D8D8", lw=0.5, alpha=0.7)
    ax_delta.text(0.01, 0.03, "points: individual images; dot ± 95% bootstrap CI", transform=ax_delta.transAxes, fontsize=6.5, color="#606060")
    add_label(ax_delta, "m")

    ax_enrich.axhline(1, color="#767676", lw=0.8, ls="--")
    ax_enrich.set_xticks(x_positions, ["att4", "att3", "att2"])
    ax_enrich.set_ylabel("oil enrichment in top 10% α")
    ax_enrich.set_title("does high α preferentially select oil pixels?", fontsize=8, pad=4)
    ax_enrich.grid(axis="y", color="#D8D8D8", lw=0.5, alpha=0.7)
    ax_enrich.text(0.01, 0.03, "1.0 = no enrichment; dashed line = pixel-proportion baseline", transform=ax_enrich.transAxes, fontsize=6.5, color="#606060")
    add_label(ax_enrich, "n")

    # Shared colorbars for image plates.
    sm_alpha = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=alpha_cmap)
    sm_alpha.set_array([])
    fig.colorbar(sm_alpha, ax=[axes[0, 3], axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3]], fraction=0.018, pad=0.012, label="spatial attention α")
    sm_energy = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(energy_low, energy_high), cmap=energy_cmap)
    sm_energy.set_array([])
    fig.colorbar(sm_energy, ax=[axes[2, 0], axes[2, 1], axes[2, 2]], fraction=0.018, pad=0.012, label="mean |activation|")

    fig.text(
        0.01,
        0.008,
        "α is a class-agnostic spatial gate, not an oil probability. Energy maps are mean absolute activation across channels; contours mark the union of the four oil classes.",
        fontsize=7,
        color="#4D4D4D",
    )
    save_figure(fig, base_path)


def write_report(summary, counter_rows, rep_name, output_path: Path):
    def fmt(value):
        return "NA" if not np.isfinite(value) else f"{value:.4f}"

    normal_miou = np.asarray([row["normal_miou"] for row in counter_rows], dtype=float)
    open_miou = np.asarray([row["alpha_one_miou"] for row in counter_rows], dtype=float)
    lines = [
        "# Attention Gate 诊断结果",
        "",
        "## 结论",
        "",
        "这套诊断把 Attention Gate 的效果拆成两个可验证问题：",
        "",
        "1. 注意力图 α 是否在真实油区更高，而不是只在任意背景纹理上高？",
        "2. α 乘回编码器 skip 特征后，是否确实抑制了无关特征，同时保留解码器的语义支路？",
        "",
        f"代表性图像采用三层 Gate 的平均 α 对比度的中位数规则选择：`{rep_name}`。这比手工挑一张最高分图片更不容易产生 cherry-picking。",
        "",
        "## 运行口径",
        "",
        "- 权重：`weights/attention_resunet.pth`（训练脚本按验证 mIoU 保存的权重）。",
        "- 测试集：10 张单独测试图，逐图与同名标签配对。",
        "- Gate：`att4`、`att3`、`att2`；α 取 `psi` 模块经过 BN+Sigmoid 后的输出。",
        "- α 上采样到原图尺寸仅用于可视化和像素级统计；原始 Gate 仍在 32×32、64×64、128×128 等尺度上工作。",
        "- 油区定义：标签中的四类剩余油，即类别编号 2–5 的并集。",
        "- `oil enrichment` = 高 α 前 10% 像素中的油像素比例 ÷ 全图油像素比例。1.0 表示没有富集。",
        "- 误差线：10 张图的均值及固定随机种子的 bootstrap 95% CI。",
        "",
        "## 全测试集结果",
        "",
        "| Gate | mean α(oil) | mean α(non-oil) | Δα | top-10% oil enrichment | global energy suppression |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['gate']} | {fmt(row['alpha_oil_mean'])} | {fmt(row['alpha_background_mean'])} | {fmt(row['alpha_delta_oil_minus_background_mean'])} | {fmt(row['alpha_top10_enrichment_mean'])} | {fmt(row['energy_suppression_global_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## α=1 反事实干预",
            "",
            f"正常 Gate 的 10 图平均 mIoU：`{normal_miou.mean():.4f}`；把三个 Gate 的输出临时强制为 α=1 后：`{open_miou.mean():.4f}`。",
            "",
            "这个结果说明当前已训练权重依赖门控产生的特征分布，但不能写成“Attention 比无 Attention 提升了多少”。原因是解码器已经在训练过程中适应了门控特征；α=1 只是在推理时改变了输入分布，并不等价于重新训练一个无注意力基线。",
            "",
            "要做正式消融，需要重新训练一个结构、数据划分、增强、随机种子、优化器和学习率完全一致、但没有 Attention Gate 的模型，再比较独立测试集上的 mIoU、各类 IoU 和少数类召回率。",
            "",
            "## 如何解读图",
            "",
            "- 第一行建立输入、真实油区和最终预测的对应关系；",
            "- 第二行显示三层 α 以及 att2 的抑制图；",
            "- 第三行不挑任意单个通道，而是用所有通道的平均绝对激活能量比较原始 skip、过滤后 skip 和同层 decoder；",
            "- 最后一行把代表性图像之外的 10 张测试图全部纳入统计。",
            "",
            "## 重要限制",
            "",
            "Attention Gate 是单通道、类别无关的空间门控。α 高不等于该位置一定是油；孔道边界等对分割有用的非油区域也可能获得较高 α。因此最严谨的表述是“α 与油区有空间重叠/富集”，而不是“α 就是油概率图”。",
            "",
            "本次图像是基于现有预处理图和标签的诊断可视化，未修改模型权重或训练代码。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = sorted(
        [p for p in TEST_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}],
        key=natural_key,
    )
    if not image_paths:
        raise RuntimeError(f"No test images found in {TEST_DIR}")
    model = load_model(device)

    all_rows = []
    normal_results = {}
    print(f"device={device}; test_images={len(image_paths)}; weight={WEIGHT_PATH.name}")
    for image_path in image_paths:
        normal = infer_one(model, image_path, device, capture=True)
        normal_results[image_path.name] = normal
        all_rows.extend(gate_metrics(normal))
        print(f"  {image_path.name}: normal mIoU={normal['miou']:.4f}")

    summary = aggregate_summary(all_rows)
    write_csv(OUT_DIR / "attention_gate_metrics.csv", all_rows)
    write_csv(OUT_DIR / "attention_gate_summary.csv", summary)

    counter_rows = []
    for image_path in image_paths:
        normal = normal_results[image_path.name]
        opened = infer_one(model, image_path, device, capture=False, force_open=True)
        row = {
            "image": image_path.name,
            "normal_miou": normal["miou"],
            "alpha_one_miou": opened["miou"],
            "miou_change_alpha_one_minus_normal": opened["miou"] - normal["miou"],
        }
        for cls, value in enumerate(normal["class_iou"]):
            row[f"normal_iou_class_{cls}_{CLASS_NAMES[cls].replace(' ', '_')}"] = value
        for cls, value in enumerate(opened["class_iou"]):
            row[f"alpha_one_iou_class_{cls}_{CLASS_NAMES[cls].replace(' ', '_')}"] = value
        counter_rows.append(row)
    write_csv(OUT_DIR / "attention_gate_alpha_one_intervention.csv", counter_rows)

    # Select a median-contrast representative across all three gates.
    image_contrasts = []
    for image_path in image_paths:
        rows = [r for r in all_rows if r["image"] == image_path.name]
        image_contrasts.append((image_path.name, float(np.nanmean([r["alpha_delta_oil_minus_background"] for r in rows]))))
    image_contrasts.sort(key=lambda item: (item[1], natural_key(Path(item[0]))))
    representative_name = image_contrasts[len(image_contrasts) // 2][0]
    representative = normal_results[representative_name]

    # The normal pass was captured for metrics; rerun only the representative
    # once to ensure all hook arrays are fresh and isolated for plotting.
    representative = infer_one(model, TEST_DIR / representative_name, device, capture=True)
    make_mechanism_figure(representative, all_rows, summary, OUT_DIR / "attention_gate_diagnostic")
    write_report(summary, counter_rows, representative_name, OUT_DIR / "attention_gate_analysis.md")

    print(f"representative={representative_name}; mean_contrast={image_contrasts[len(image_contrasts)//2][1]:.4f}")
    print(f"outputs={OUT_DIR}")
    for row in summary:
        print(
            f"{row['gate']}: alpha_oil={row['alpha_oil_mean']:.4f}, "
            f"alpha_non_oil={row['alpha_background_mean']:.4f}, "
            f"delta={row['alpha_delta_oil_minus_background_mean']:.4f}, "
            f"top10_enrichment={row['alpha_top10_enrichment_mean']:.3f}"
        )
    print(f"counterfactual normal_mean_mIoU={np.mean([r['normal_miou'] for r in counter_rows]):.4f}")
    print(f"counterfactual alpha_one_mean_mIoU={np.mean([r['alpha_one_miou'] for r in counter_rows]):.4f}")


if __name__ == "__main__":
    main()
