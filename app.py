import os
import sys
import tempfile
from PIL import Image
import streamlit as st

# Configure headless matplotlib environment
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# Add artvision package to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTVISION_DIR = os.path.join(BASE_DIR, "artvision")
if ARTVISION_DIR not in sys.path:
    sys.path.insert(0, ARTVISION_DIR)

# Streamlit Page Configuration
st.set_page_config(
    page_title="ArtVision | Multi-Task Artwork Analysis",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .badge-human {
        background-color: #dcfce7;
        color: #15803d;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-ai {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    .swatch-box {
        display: inline-block;
        width: 48px;
        height: 48px;
        border-radius: 8px;
        margin-right: 8px;
        border: 1px solid rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Import and verify predict_artwork module once."""
    from predict import predict_artwork
    return predict_artwork


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/art.png", width=90)
    st.markdown("### **ArtVision Framework**")
    st.markdown("""
    An explainable multi-task deep learning system built for museum-grade artwork analysis and synthetic AI art verification.
    
    **Core Capabilities:**
    - 🏛️ **Art Style & Movement** (27 classes)
    - 🛡️ **Authenticity Detection** (Human vs AI)
    - 👨‍🎨 **Artist Attribution** (Top master painters)
    - 🎭 **Genre & Mood** Classification
    - 🎨 **CIELAB Color Palette & Tone** (~53ms)
    - 🔍 **Grad-CAM Visual Explanations**
    """)
    st.markdown("---")
    st.markdown("**Backbone**: EfficientNet-B0")
    st.markdown("**Loss Balancing**: Homoscedastic Uncertainty")
    st.markdown("**Calibration**: Temperature Scaling")
    st.markdown("---")
    st.caption("ArtVision • 7th Sem ML Project • PDEU")


# Main Header
st.markdown('<div class="main-title">ArtVision: Artwork Analysis & Authentication</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-task neural classification, authenticity verification, color theory extraction, and Grad-CAM interpretability.</div>', unsafe_allow_html=True)

# Tabs
tab_analysis, tab_benchmarks, tab_architecture = st.tabs([
    "🎨 Artwork Analysis",
    "📊 Empirical Benchmarks",
    "🏛️ System Architecture"
])

with tab_analysis:
    col_input, col_results = st.columns([1, 1.3], gap="large")

    with col_input:
        st.markdown("#### 1. Input Artwork")
        upload_mode = st.radio("Choose input method:", ["Upload Image", "Sample Artwork"], horizontal=True)

        input_image = None
        temp_img_path = None

        if upload_mode == "Upload Image":
            uploaded_file = st.file_uploader("Upload an artwork file", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file).convert("RGB")
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                input_image.save(temp_file.name, "JPEG", quality=95)
                temp_img_path = temp_file.name
        else:
            sample_dir = os.path.join(ARTVISION_DIR, "frontend/src/test")
            sample_file = os.path.join(sample_dir, "fixtures_non_art.jpg")
            if os.path.exists(sample_file):
                input_image = Image.open(sample_file).convert("RGB")
                temp_img_path = sample_file
                st.info("Loaded sample artwork test fixture.")
            else:
                st.warning("Sample image not found; please upload an image.")

        if input_image is not None:
            st.image(input_image, caption="Input Artwork", use_container_width=True)

        explain_task = st.selectbox(
            "Grad-CAM Explanation Head:",
            ["style", "authenticity", "artist", "genre", "mood"],
            index=0,
            help="Select which neural prediction head to generate visual attribution heatmaps for."
        )

        analyze_btn = st.button("🚀 Run Multi-Task Analysis", type="primary", use_container_width=True, disabled=(input_image is None))

    with col_results:
        st.markdown("#### 2. Analysis Results")
        if analyze_btn and temp_img_path:
            with st.spinner("Executing calibrated multi-task inference & Grad-CAM..."):
                try:
                    predict_fn = load_predictor()
                    ckpt_path = os.path.join(ARTVISION_DIR, "runs/multitask_model/best_model.pth")
                    calib_path = os.path.join(ARTVISION_DIR, "runs/multitask_model/calibration.json")
                    color_cfg = os.path.join(ARTVISION_DIR, "configs/color.yaml")
                    pred_cfg = os.path.join(ARTVISION_DIR, "configs/predict.yaml")
                    explain_dir = os.path.join(ARTVISION_DIR, "runs/explain")
                    os.makedirs(explain_dir, exist_ok=True)

                    res = predict_fn(
                        image_path=temp_img_path,
                        checkpoint_path=ckpt_path,
                        calibration_path=calib_path,
                        color_config_path=color_cfg,
                        predict_config_path=pred_cfg,
                        device="cpu",
                        explain=True,
                        explain_task=explain_task,
                        explain_output_dir=explain_dir
                    )

                    # Top Cards: Authenticity & Style
                    auth = res.get("authenticity", {})
                    is_ai = auth.get("is_ai_generated", False)
                    auth_conf = auth.get("confidence", 0.0)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Authenticity Verification**")
                        if is_ai:
                            st.markdown(f'<div class="badge-ai">⚠️ Synthetic (AI Generated) — {auth_conf*100:.1f}%</div>', unsafe_allow_html=True)
                            st.caption(f"Estimated AI probability: {auth.get('probability_ai', 0.0)*100:.1f}%")
                        else:
                            st.markdown(f'<div class="badge-human">✅ Human Made Masterwork — {auth_conf*100:.1f}%</div>', unsafe_allow_html=True)
                            st.caption(f"Human authenticity confidence: {auth_conf*100:.1f}%")

                    with c2:
                        style_info = res.get("style", {})
                        st.markdown("**Style & Movement**")
                        st.markdown(f"### {style_info.get('prediction', 'Unknown')}")
                        st.caption(f"Historical Movement: **{res.get('movement', {}).get('prediction', 'N/A')}**")

                    st.markdown("---")

                    # Artist & Genre
                    c3, c4, c5 = st.columns(3)
                    with c3:
                        artist_info = res.get("artist", {})
                        st.markdown("**Artist Attribution**")
                        st.write(f"**{artist_info.get('prediction', 'Unknown')}**")
                        st.caption(f"Confidence: {artist_info.get('confidence', 0.0)*100:.1f}%")

                    with c4:
                        genre_info = res.get("genre", {})
                        st.markdown("**Genre / Subject**")
                        st.write(f"**{genre_info.get('prediction', 'General')}**")
                        st.caption(f"Confidence: {genre_info.get('confidence', 0.0)*100:.1f}%")

                    with c5:
                        mood_info = res.get("mood", {})
                        st.markdown("**Dominant Mood**")
                        st.write(f"**{mood_info.get('prediction', 'Neutral')}**")
                        st.caption(f"Confidence: {mood_info.get('confidence', 0.0)*100:.1f}%")

                    st.markdown("---")

                    # Color Palette & Tone
                    color_data = res.get("color", {})
                    st.markdown("##### 🎨 CIELAB Dominant Color Palette")
                    palette = color_data.get("palette", [])
                    if palette:
                        swatch_cols = st.columns(len(palette))
                        for i, swatch in enumerate(palette):
                            with swatch_cols[i]:
                                hex_val = swatch.get("hex", "#888888")
                                pct = swatch.get("percent", 0.0) * 100
                                st.markdown(f"""
                                <div style="background-color: {hex_val}; height: 50px; border-radius: 6px; border: 1px solid #ccc;"></div>
                                <div style="text-align: center; font-size: 0.8rem; margin-top: 4px;"><code>{hex_val}</code><br><b>{pct:.1f}%</b></div>
                                """, unsafe_allow_html=True)

                    st.write(
                        f"**Temperature**: `{color_data.get('temperature', 'N/A').title()}` • "
                        f"**Brightness**: `{color_data.get('brightness', 'N/A').title()}` • "
                        f"**Saturation**: `{color_data.get('saturation', 'N/A').title()}`"
                    )

                    st.markdown("---")

                    # Grad-CAM Heatmap
                    expl = res.get("explanation", {})
                    overlay_path = expl.get("overlay_path")
                    if overlay_path and os.path.exists(overlay_path):
                        st.markdown(f"##### 🔍 Grad-CAM Visual Attribution ({explain_task.title()} Head)")
                        cam_col1, cam_col2 = st.columns(2)
                        with cam_col1:
                            st.image(input_image, caption="Original Artwork", use_container_width=True)
                        with cam_col2:
                            st.image(Image.open(overlay_path), caption=f"Attribution Heatmap ({expl.get('target_prediction', '')})", use_container_width=True)
                        st.caption(f"Border Mass Fraction (BMF): `{expl.get('border_mass_fraction', 0.0):.4f}` • Indicates neural focus on core subject matter vs background.")

                except Exception as e:
                    st.error(f"Inference error: {str(e)}")
                    st.info("Tip: Ensure the model weights are loaded and sufficient container memory is available.")
        else:
            st.info("👈 Upload an artwork on the left and click **Run Multi-Task Analysis**.")


with tab_benchmarks:
    st.markdown("### 📊 Test Set Benchmark Performance (N = 816)")
    st.markdown("""
    Evaluated across independent test splits combining museum scans from WikiArt and generative benchmarks from ArtBench (Stable Diffusion and Latent Diffusion).
    """)

    metrics_data = [
        {"Task": "Authenticity (Human vs AI)", "Top-1 Accuracy": "97.06%", "Top-3 Accuracy": "100.0%", "Macro F1": "0.935", "Calibrated ECE": "0.0107", "Highlight": "AUROC: 0.995"},
        {"Task": "Derived Movement", "Top-1 Accuracy": "72.01%", "Top-3 Accuracy": "100.0%", "Macro F1": "0.720", "Calibrated ECE": "—", "Highlight": "7 Historical Eras"},
        {"Task": "Genre Recognition", "Top-1 Accuracy": "66.35%", "Top-3 Accuracy": "94.04%", "Macro F1": "0.565", "Calibrated ECE": "0.0664", "Highlight": "94.0% Top-3 Precision"},
        {"Task": "Artist Attribution", "Top-1 Accuracy": "65.05%", "Top-3 Accuracy": "82.61%", "Macro F1": "0.625", "Calibrated ECE": "0.0439", "Highlight": "82.6% Top-3 Precision"},
        {"Task": "Style Classification", "Top-1 Accuracy": "59.93%", "Top-3 Accuracy": "84.22%", "Macro F1": "0.434", "Calibrated ECE": "0.0442", "Highlight": "84.2% Top-3 Precision"},
        {"Task": "Color Extraction (CIELAB)", "Top-1 Accuracy": "—", "Top-3 Accuracy": "—", "Macro F1": "—", "Calibrated ECE": "—", "Highlight": "53.6 ms CPU Latency"},
    ]
    st.table(metrics_data)

    st.markdown("#### Publication-Grade Empirical Figures")
    fig1 = os.path.join(BASE_DIR, "showcase/evaluation/01_multitask_training_curves.png")
    fig2 = os.path.join(BASE_DIR, "showcase/evaluation/06_confusion_matrices_grid.png")
    fig3 = os.path.join(BASE_DIR, "showcase/results/02_gradcam_explainability_gallery.png")

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if os.path.exists(fig1):
            st.image(fig1, caption="Multi-Task Training Dynamics & Convergence", use_container_width=True)
    with b_col2:
        if os.path.exists(fig2):
            st.image(fig2, caption="Normalized Confusion Matrices Across Task Heads", use_container_width=True)

    if os.path.exists(fig3):
        st.image(fig3, caption="Grad-CAM Interpretability Gallery Across Masterworks and AI Fakes", use_container_width=True)


with tab_architecture:
    st.markdown("### 🏛️ Model Architecture & Multi-Task Formulation")
    st.markdown(r"""
    #### 1. Shared Feature Representation
    ArtVision utilizes an **EfficientNet-B0** convolutional neural network backbone with compound scaling:
    - Adaptive Average Pooling converts spatial feature maps into an embedding vector $\mathbf{z} \in \mathbb{R}^{1280}$.
    - Modular projection heads route $\mathbf{z}$ to task-specific classifiers.

    #### 2. Multi-Task Uncertainty Loss Weighting (Kendall et al.)
    Instead of hand-tuned fixed hyperparameters, ArtVision learns task-dependent observation noise $\sigma_t^2$ dynamically:
    $$\mathcal{L}_{\text{total}} = \sum_{t \in \mathcal{T}} \left( \frac{1}{2\sigma_t^2} \mathcal{L}_t + \log \sigma_t \right)$$

    #### 3. Dynamic Masked Backpropagation
    When an artwork lacks annotations for specific heads (e.g., AI images without artist labels), binary masks $m_t \in \{0, 1\}$ zero the gradient contribution:
    $$\mathcal{L} = \sum_{t} m_t \cdot w_t \cdot \mathcal{L}_t$$

    #### 4. Post-Hoc Temperature Scaling Calibration
    Softmax outputs are calibrated on the validation split via Platt scaling to ensure output probabilities represent true empirical confidence:
    $$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
    """)
