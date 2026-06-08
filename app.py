# =============================================================================
# Prob-PSENN · Probabilistic Prototype-Based Self-Explainable Neural Network
# Production inference application for Hugging Face Spaces
# =============================================================================
# Architecture:  FeatureExtractor (CNN) → PrototypeNetwork (50 prototypes)
# Explainability: Prototype similarity · Activation heatmaps · Entropy analysis
# Interface:      Gradio Sketchpad with dark terminal aesthetic
# =============================================================================

import os
import sys
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend — required for HF Spaces
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import gradio as gr

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: %s", DEVICE)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH      = "prob_psenn_model.pth"
NUM_PROTOTYPES  = 50
EMBED_DIM       = 64
IMG_SIZE        = 28
MC_PASSES       = 20            # Monte Carlo dropout forward passes


# =============================================================================
# MODEL DEFINITION
# =============================================================================

class FeatureExtractor(nn.Module):
    """
    Lightweight CNN backbone.
    Input:  (B, 1, 28, 28)
    Output: (B, 64)  — L2-normalized embedding
    """

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # → (B, 16, 14, 14)

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # → (B, 32,  7,  7)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, EMBED_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc(x)
        return x


class PrototypeNetwork(nn.Module):
    """
    Prototype-based classifier.

    Forward pass:
      1. Extract CNN embedding.
      2. Cosine-similarity against NUM_PROTOTYPES learnable prototype vectors.
      3. Linear classifier over the similarity vector.
    """

    def __init__(self, num_prototypes: int = NUM_PROTOTYPES):
        super().__init__()

        self.feature_extractor = FeatureExtractor()

        # Learnable prototype matrix  (num_prototypes × EMBED_DIM)
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, EMBED_DIM)
        )

        # Final classifier  (num_prototypes → 10 digit classes)
        self.classifier = nn.Linear(num_prototypes, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.feature_extractor(x)
        embeddings = F.normalize(embeddings, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        similarity = torch.matmul(embeddings, prototypes.T)
        logits     = self.classifier(similarity)
        return logits


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(path: str = MODEL_PATH) -> PrototypeNetwork:
    """
    Load a trained PrototypeNetwork from a .pth state-dict file.

    Args:
        path: Path to the saved state-dict (.pth).

    Returns:
        Model in eval mode on DEVICE.

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError:      If the state-dict is incompatible.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model weights not found at '{path}'. "
            "Place 'prob_psenn_model.pth' in the same directory as app.py."
        )

    model = PrototypeNetwork(num_prototypes=NUM_PROTOTYPES).to(DEVICE)

    try:
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        log.info("Loaded model weights from '%s'", path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load state-dict: {exc}") from exc

    model.eval()
    return model


# Load model once at startup
try:
    MODEL = load_model()
except Exception as exc:
    log.error("Model loading failed: %s", exc)
    MODEL = None

# Pre-compute normalised prototypes (reused on every inference call)
_PROTOTYPES_NORM: torch.Tensor | None = None
if MODEL is not None:
    with torch.no_grad():
        _PROTOTYPES_NORM = F.normalize(MODEL.prototypes, dim=1)


# =============================================================================
# PREPROCESSING
# =============================================================================

# Must match training pipeline (ToTensor only — no extra normalisation)
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
])


def preprocess_image(img_array: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """
    Convert a raw numpy RGBA/RGB sketchpad image to a model-ready tensor.

    Steps:
      • Convert to grayscale (L).
      • Resize to 28×28.
      • Invert (Gradio draws white on black; MNIST is white digit on black).
      • Apply ToTensor transform.
      • Unsqueeze batch dimension.

    Args:
        img_array: Raw numpy image from Gradio Sketchpad (H×W×3 or H×W×4).

    Returns:
        Tuple of (tensor (1,1,28,28) on DEVICE, PIL-converted numpy array).
    """
    pil_img    = Image.fromarray(img_array).convert("L").resize((IMG_SIZE, IMG_SIZE))
    pil_img    = ImageOps.invert(pil_img)
    img_np     = np.array(pil_img)
    img_tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    return img_tensor, img_np


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

def run_inference(
    img_tensor: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single deterministic forward pass.

    Returns:
        probs_np       – softmax class probabilities (10,)
        similarities   – cosine similarities to all prototypes (NUM_PROTOTYPES,)
        proto_idx      – top-5 prototype indices
        proto_scores   – top-5 prototype similarity scores
    """
    with torch.no_grad():
        embedding    = F.normalize(MODEL.feature_extractor(img_tensor), dim=1)
        similarities = torch.matmul(embedding, _PROTOTYPES_NORM.T).squeeze()
        logits       = MODEL(img_tensor)
        probs        = torch.softmax(logits, dim=1)

    probs_np   = probs.cpu().numpy()[0]
    sims_np    = similarities.cpu().numpy()

    top5       = torch.topk(similarities, 5)
    proto_idx  = top5.indices.cpu().numpy()
    proto_scores = top5.values.cpu().numpy()

    return probs_np, sims_np, proto_idx, proto_scores


def run_mc_dropout(img_tensor: torch.Tensor, n_passes: int = MC_PASSES) -> dict:
    """
    Monte Carlo Dropout uncertainty estimation.
    Dropout layers remain active (train mode) for T stochastic forward passes.

    Args:
        img_tensor: Preprocessed input tensor (1,1,28,28).
        n_passes:   Number of stochastic forward passes (default 20).

    Returns:
        Dictionary with keys:
            mean_probs  – mean softmax probability per class (10,)
            variance    – variance per class (10,)
            uncertainty – scalar mean variance across classes
    """
    # Activate dropout without updating batch-norm statistics
    MODEL.feature_extractor.fc[3].train()   # Dropout layer index in fc

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = MODEL(img_tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)

    MODEL.eval()    # restore eval mode

    all_probs  = np.array(all_probs)            # (T, 10)
    mean_probs = all_probs.mean(axis=0)
    variance   = all_probs.var(axis=0)

    return {
        "mean_probs":  mean_probs,
        "variance":    variance,
        "uncertainty": float(variance.mean()),
    }


def compute_entropy(probs: np.ndarray) -> float:
    """
    Shannon entropy of a probability distribution.
    H = -Σ p_c · log(p_c)

    Thresholds:
        H < 0.5   → Low entropy  (high confidence)
        H < 1.5   → Moderate
        H ≥ 1.5   → High entropy (unreliable)
    """
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def get_activation_heatmap(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Compute the mean activation map from the final conv layer and upsample
    to 28×28 via bilinear zoom.

    Args:
        img_tensor: Preprocessed input (1,1,28,28).

    Returns:
        Normalised heatmap array (28,28) in [0, 1].
    """
    with torch.no_grad():
        feature_maps = MODEL.feature_extractor.conv(img_tensor)   # (1,32,7,7)

    act = feature_maps.mean(dim=1).squeeze().cpu().numpy()        # (7,7)
    act = (act - act.min()) / (act.max() - act.min() + 1e-8)

    heatmap = zoom(
        act,
        (IMG_SIZE / act.shape[0], IMG_SIZE / act.shape[1]),
    )
    return heatmap


# =============================================================================
# TEXT REPORT BUILDER
# =============================================================================

def build_report(
    pred: int,
    confidence: float,
    uncertainty: float,
    entropy: float,
    mc_uncertainty: float,
    probs_np: np.ndarray,
    proto_idx: np.ndarray,
    proto_scores: np.ndarray,
) -> str:
    """
    Build the monospaced AI explanation report shown in the Gradio textbox.
    """
    top3_idx = np.argsort(probs_np)[-3:][::-1]

    if confidence > 0.95:
        status = "● VERY HIGH"
    elif confidence > 0.80:
        status = "● GOOD"
    else:
        status = "● UNCERTAIN"

    if entropy < 0.5:
        entropy_label = "🟢 LOW  — high confidence"
    elif entropy < 1.5:
        entropy_label = "🟡 MODERATE — some ambiguity"
    else:
        entropy_label = "🔴 HIGH  — unreliable prediction"

    lines = [
        "PREDICTED DIGIT",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  {pred}",
        "",
        "CONFIDENCE METRICS",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  Confidence       {confidence * 100:5.1f}%",
        f"  Uncertainty      {uncertainty * 100:5.1f}%",
        f"  Entropy          {entropy:.4f}",
        f"  Reliability      {status}",
        "",
        "ENTROPY ANALYSIS",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  {entropy_label}",
        f"  MC Dropout Var.  {mc_uncertainty:.6f}",
        "",
        "TOP 3 PREDICTIONS",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  #1  Digit {top3_idx[0]}   {probs_np[top3_idx[0]] * 100:5.1f}%",
        f"  #2  Digit {top3_idx[1]}   {probs_np[top3_idx[1]] * 100:5.1f}%",
        f"  #3  Digit {top3_idx[2]}   {probs_np[top3_idx[2]] * 100:5.1f}%",
        "",
        "PROTOTYPE REASONING",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    for i in range(5):
        bar   = "█" * int(proto_scores[i] * 20)
        lines.append(
            f"  P{proto_idx[i]:02d}  {bar:<20}  {proto_scores[i]:.3f}"
        )

    return "\n".join(lines)


# =============================================================================
# PLOT BUILDERS
# =============================================================================

def plot_confidence(probs_np: np.ndarray, pred: int) -> plt.Figure:
    """Class confidence bar chart — dark terminal style."""
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    colors = ["#00ff88" if i == pred else "#1a3a2a" for i in range(10)]
    bars   = ax.bar(range(10), probs_np, color=colors, width=0.65,
                    edgecolor="none", zorder=3)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(range(10))
    ax.set_xticklabels([str(i) for i in range(10)],
                       color="#aaaaaa", fontsize=12, fontfamily="monospace")
    ax.tick_params(axis="y", colors="#444444", labelsize=9)
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, color="#1a1a1a", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("CLASS CONFIDENCE", color="#00ff88", fontsize=13,
                 fontfamily="monospace", pad=12, loc="left", weight="bold")

    ax.text(pred, probs_np[pred] + 0.04, f"{probs_np[pred] * 100:.1f}%",
            ha="center", va="bottom", color="#00ff88",
            fontsize=11, fontfamily="monospace", weight="bold")

    plt.tight_layout(pad=1.5)
    return fig


def plot_prototype_contributions(
    proto_idx: np.ndarray, proto_scores: np.ndarray
) -> plt.Figure:
    """Horizontal bar chart of top-5 prototype similarity scores."""
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    y_pos  = range(5)
    labels = [f"Prototype {proto_idx[i]:02d}" for i in range(5)]
    colors = ["#00ff88"] + ["#00aaff"] * 4     # top prototype highlighted

    hbars = ax.barh(list(y_pos), proto_scores, color=colors,
                    height=0.55, edgecolor="none")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, color="#aaaaaa", fontsize=11,
                       fontfamily="monospace")
    ax.tick_params(axis="x", colors="#444444", labelsize=9)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()
    ax.xaxis.grid(True, color="#1a1a1a", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("PROTOTYPE CONTRIBUTIONS", color="#00aaff", fontsize=13,
                 fontfamily="monospace", pad=12, loc="left", weight="bold")

    for i, score in enumerate(proto_scores):
        ax.text(score + 0.02, i, f"{score:.3f}",
                va="center", color="#888888", fontsize=9,
                fontfamily="monospace")

    plt.tight_layout(pad=1.5)
    return fig


def plot_visual_explanation(orig_np: np.ndarray, heatmap: np.ndarray) -> plt.Figure:
    """Three-panel explainability: input · activation · overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), facecolor="#0d0d0d")
    fig.subplots_adjust(wspace=0.05)

    for ax in axes:
        ax.set_facecolor("#0d0d0d")
        for spine in ax.spines.values():
            spine.set_edgecolor("#222222")

    axes[0].imshow(orig_np, cmap="gray", interpolation="nearest")
    axes[1].imshow(heatmap, cmap="inferno", interpolation="bilinear")
    axes[2].imshow(orig_np, cmap="gray", interpolation="nearest")
    axes[2].imshow(heatmap, cmap="jet", alpha=0.55, interpolation="bilinear")

    for ax, title in zip(axes, ["INPUT", "ACTIVATION", "OVERLAY"]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, color="#555555", fontsize=9,
                     fontfamily="monospace", pad=6)

    fig.suptitle("VISUAL EXPLANATION", color="#ff6b35", fontsize=13,
                 fontfamily="monospace", weight="bold", y=1.02)
    plt.tight_layout(pad=0.5)
    return fig


# =============================================================================
# MAIN PREDICT FUNCTION  (wired to Gradio)
# =============================================================================

def predict_digit(
    img,
) -> tuple[str, plt.Figure | None, plt.Figure | None, plt.Figure | None]:
    """
    Full inference pipeline called by the Gradio button.

    Args:
        img: Numpy array from the Gradio Sketchpad (RGB, H×W×3).

    Returns:
        Tuple of (report_text, fig_confidence, fig_prototypes, fig_xai).
        All figures are None on error.
    """
    # Guard: model not loaded
    if MODEL is None:
        return (
            "ERROR: Model not loaded. Check that 'prob_psenn_model.pth' "
            "exists in the app directory.",
            None, None, None,
        )

    # Guard: empty canvas
    if img is None:
        return "Draw a digit (0–9) on the canvas, then press ANALYSE →", None, None, None

    # Gradio Sketchpad returns dict in some versions
    if isinstance(img, dict):
        img = img.get("composite", img)

    # Check whether the canvas is blank (all-black or all-white)
    arr = np.array(img)
    if arr.std() < 5.0:
        return "Canvas appears empty. Draw a digit and press ANALYSE →", None, None, None

    try:
        # ── 1. Preprocess ────────────────────────────────────────────────
        img_tensor, orig_np = preprocess_image(arr)

        # ── 2. Deterministic inference ───────────────────────────────────
        probs_np, sims_np, proto_idx, proto_scores = run_inference(img_tensor)

        pred        = int(np.argmax(probs_np))
        confidence  = float(np.max(probs_np))
        uncertainty = 1.0 - confidence
        entropy     = compute_entropy(probs_np)

        # ── 3. MC Dropout uncertainty ────────────────────────────────────
        mc = run_mc_dropout(img_tensor, n_passes=MC_PASSES)
        mc_uncertainty = mc["uncertainty"]

        # ── 4. Activation heatmap ────────────────────────────────────────
        heatmap = get_activation_heatmap(img_tensor)

        # ── 5. Build outputs ─────────────────────────────────────────────
        report = build_report(
            pred, confidence, uncertainty, entropy,
            mc_uncertainty, probs_np, proto_idx, proto_scores,
        )

        fig1 = plot_confidence(probs_np, pred)
        fig2 = plot_prototype_contributions(proto_idx, proto_scores)
        fig3 = plot_visual_explanation(orig_np, heatmap)

        log.info(
            "Inference: pred=%d  conf=%.2f%%  entropy=%.4f",
            pred, confidence * 100, entropy,
        )

        return report, fig1, fig2, fig3

    except Exception as exc:
        log.exception("Inference error: %s", exc)
        return f"Error during inference: {exc}", None, None, None


# =============================================================================
# GRADIO UI
# =============================================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

:root {
    --bg:      #080808;
    --surface: #111111;
    --border:  #1e1e1e;
    --accent:  #00ff88;
    --accent2: #00aaff;
    --accent3: #ff6b35;
    --text:    #e0e0e0;
    --muted:   #555555;
    --mono:    'Space Mono', monospace;
    --sans:    'Inter', sans-serif;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: var(--sans) !important;
    color: var(--text) !important;
}

.gradio-container h1 {
    font-family: var(--mono) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    color: var(--accent) !important;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.75rem;
    margin-bottom: 0.25rem !important;
}

.gradio-container .description p {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.04em;
    line-height: 1.7;
}

.block, .panel, .form {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

label span, .label-wrap span {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
}

canvas {
    border: 1px solid #00ff8833 !important;
    border-radius: 4px !important;
    box-shadow: 0 0 24px #00ff8811 !important;
    cursor: crosshair !important;
}

textarea {
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    line-height: 1.85 !important;
    color: #cccccc !important;
    background: #0a0a0a !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 1rem !important;
    letter-spacing: 0.04em !important;
}

button.primary, button[variant="primary"] {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    padding: 0.6rem 1.6rem !important;
    transition: background 0.15s, box-shadow 0.15s !important;
}

button.primary:hover, button[variant="primary"]:hover {
    background: #00ff8818 !important;
    box-shadow: 0 0 16px #00ff8833 !important;
}

button.secondary, button[variant="secondary"] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    border-radius: 2px !important;
    transition: border-color 0.15s !important;
}

button.secondary:hover, button[variant="secondary"]:hover {
    border-color: var(--muted) !important;
    color: var(--text) !important;
}

.plot-component, .gr-plot {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 0.5rem !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
"""

DESCRIPTION_MD = """
# PROB-PSENN  //  PROBABILISTIC SELF-EXPLAINABLE AI

Prototype-Based Neural Network · Monte Carlo Dropout · Visual Attention Mapping
"""

INFO_MD = """
**How to use:**  Draw a digit (0–9) on the canvas → press **ANALYSE →**

The model explains its prediction through:
- **Confidence Scoring** — per-class softmax probabilities
- **Entropy Analysis** — distribution-aware uncertainty
- **Prototype Reasoning** — which learned prototypes drove the decision
- **Visual Heatmap** — which pixels activated the network most
"""


def build_interface() -> gr.Blocks:
    """Assemble and return the Gradio Blocks interface."""

    with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="Prob-PSENN") as demo:

        gr.Markdown(DESCRIPTION_MD)
        gr.Markdown(INFO_MD)

        with gr.Row():

            # ── Left column: input + report ──────────────────────────────
            with gr.Column(scale=1):
                sketch = gr.Sketchpad(
                    image_mode="RGB",
                    type="numpy",
                    label="INPUT  ·  Draw a digit 0–9",
                    height=300,
                )

                with gr.Row():
                    btn_predict = gr.Button("ANALYSE →", variant="primary")
                    btn_clear   = gr.ClearButton(
                        components=[sketch], value="CLEAR"
                    )

                report = gr.Textbox(
                    label="AI EXPLANATION REPORT",
                    lines=26,
                    interactive=False,
                    placeholder="Report will appear here after analysis…",
                )

            # ── Right column: visualisations ──────────────────────────────
            with gr.Column(scale=1):
                plot_conf  = gr.Plot(label="CLASS CONFIDENCE")
                plot_proto = gr.Plot(label="PROTOTYPE CONTRIBUTIONS")
                plot_xai   = gr.Plot(label="VISUAL EXPLANATION")

        # ── Wire up ───────────────────────────────────────────────────────
        btn_predict.click(
            fn=predict_digit,
            inputs=[sketch],
            outputs=[report, plot_conf, plot_proto, plot_xai],
        )

    return demo


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
