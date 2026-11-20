# 📡 SONAR-AI: Forward-Looking Sonar Anomaly Detector & AI Co-Pilot

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-2F81F7?style=for-the-badge&logo=streamlit&logoColor=white)](https://sonar-ai-wxqsxczhifkuerpt7zdsd6.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Trained%20in-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Gemini API](https://img.shields.io/badge/AI%20Co--Pilot-Gemini%20Vision-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)

An end-to-end **zero-label machine learning and generative AI system** for inspecting **Forward-Looking Sonar (FLS)** imagery. Built to operate without manual bounding-box annotations, the application combines an unsupervised Masked Convolutional Autoencoder (exported to ONNX for fast CPU inference), acoustic polar telemetry estimation, and a multimodal Gemini AI Co-Pilot that converts raw acoustic scans into structured maritime inspection logs.

🔗 **Live Application:** [https://sonar-ai-wxqsxczhifkuerpt7zdsd6.streamlit.app/](https://sonar-ai-wxqsxczhifkuerpt7zdsd6.streamlit.app/)

---

## 🎯 Problem Statement

Forward-Looking Sonar scans present unique computer vision challenges compared to standard RGB imagery:
- Heavy **acoustic speckle noise** and low visual contrast.
- **Scarcity of labeled datasets** for underwater targets (debris, cables, pipelines, hazards).
- Strict **fan-shaped acoustic geometry**, where irrelevant black borders dominate raw frames.

**SONAR-AI** addresses these challenges by training a **Masked Convolutional Autoencoder** exclusively on raw `.jpg` sonar frames. The model learns normal seabed backscatter and range-ring patterns inside the sonar fan wedge, automatically surfacing high-reflectivity foreground objects as reconstruction anomalies—requiring **zero manual labels**.

---

## ✨ Key Features

- **Zero-Label Anomaly Localization:** Detects bright underwater targets using pixel-wise reconstruction error strictly inside an eroded sonar fan mask.
- **Interactive Acoustic Controls:** Real-time adjustment of anomaly sensitivity (percentile threshold), heatmap colormaps (`Turbo`, `Inferno`, `Ocean`, `Jet`), and overlay opacity.
- **Automated Polar Telemetry:** Estimates target **normalized range (%)**, **bearing angle (degrees)**, **side (Port / Starboard / Centerline)**, target area, and CPU inference latency from the transducer apex.
- **Autoencoder Diagnostics View:** Side-by-side comparison of the raw scan, autoencoder background reconstruction, blended error heatmap, and localized target bounding box.
- **AI Maritime Inspection Co-Pilot:** Integrates **Gemini 3.5 Flash-Lite** (`google-genai`) with the annotated scan and computed telemetry to generate structured inspection logs covering:
  1. Acoustic Signature (reflectivity & shadow analysis)
  2. Plausible Underwater Object Types (with uncertainty awareness)
  3. Risk / Priority Level triage
  4. Recommended Next ROV/AUV Maneuver
- **One-Click Report Export:** Download generated inspection logs directly as `.md` files.

---

## 🏗️ System Architecture

```text
Raw FLS Scan (.jpg / .png)
        │
        ▼
[1. Sonar Preprocessing (OpenCV)]
    • Resize (256×256) & Grayscale intensity extraction
    • 3×3 Median filtering for speckle suppression
    • Eroded binary fan-wedge masking (ignores black borders)
        │
        ▼
[2. Unsupervised ML Engine (ONNX Runtime)]
    • Masked Convolutional Autoencoder (256→16×16 bottleneck→256)
    • Pixel-wise reconstruction error + Gaussian smoothing
    • Percentile thresholding & contour localization
        │
        ▼
[3. Polar Geometry Telemetry (NumPy/Math)]
    • Transducer apex estimation from fan mask
    • Normalized range (%) & bearing angle (°) calculation
        │
        ▼
[4. AI Inspection Co-Pilot (Gemini 3.5 Flash-Lite)]
    • Multimodal analysis of annotated frame + polar telemetry
    • Structured maritime inspection log + Markdown export
```

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Model Training** | PyTorch (NVIDIA T4 GPU on Google Colab) |
| **Model Deployment** | ONNX + ONNX Runtime (lightweight CPU inference, no PyTorch required in production) |
| **Image Processing** | OpenCV (`opencv-python-headless`), NumPy, Pillow |
| **Multimodal GenAI** | Google GenAI SDK (`gemini-3.5-flash-lite`) |
| **Web Interface** | Streamlit (Custom Slate & Cobalt dark theme) |
| **Cloud Hosting** | Streamlit Community Cloud |

---

## 📁 Repository Structure

```text
Sonar-AI/
├── .streamlit/
│   └── config.toml        # Custom Slate & Cobalt dark theme configuration
├── .gitignore             # Excludes .venv, cache files, and local API secrets
├── README.md              # Project documentation
├── requirements.txt       # Lightweight production dependencies
├── sonar_ae.onnx          # Exported Masked Convolutional Autoencoder weights
└── streamlit_app.py       # Main Streamlit application & AI Co-Pilot pipeline
```

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/navanil-das/Sonar-AI.git](https://github.com/navanil-das/Sonar-AI.git)
cd Sonar-AI
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# Windows (Command Prompt)
.venv\Scripts\activate

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Configure your Gemini API key
Get a free API key from [Google AI Studio](https://aistudio.google.com/) and create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```

### 5. Launch the app
```bash
python -m streamlit run streamlit_app.py
```
Open `http://localhost:8501` in your browser and upload any `.jpg` or `.png` Forward-Looking Sonar scan.

---

## 🧠 Model Training Summary

- **Architecture:** 4-layer strided Convolutional Encoder (`1 → 32 → 64 → 128 → 256`, compressing `256×256` input into a `16×16` latent bottleneck) paired with a symmetric Transposed-Convolution Decoder with Sigmoid output.
- **Loss Function:** **Fan-Masked Mean Squared Error (MSE)** computed strictly over active sonar beam pixels (`mask > 8`), preventing the network from wasting capacity on zero-intensity background padding.
- **Export:** Converted from PyTorch (`.pth`) to a single-file ONNX graph (`sonar_ae.onnx`, opset 18) for low-memory, sub-second CPU inference on cloud free tiers.

---

## 🔭 Limitations & Future Scope

- **Single-Frame Perspective:** Single-angle FLS frames carry inherent visual ambiguity; future iterations can integrate multi-frame tracking (e.g., ByteTrack) across ROV video feeds.
- **Multi-Target Localization:** Extend contour extraction to rank and inspect multiple distinct targets simultaneously per scan.
- **Batch Survey Mode:** Support uploading multi-frame survey folders with automated anomaly ranking and consolidated PDF mission reports.

---

## 👤 Author

**Navanil Das**  
GitHub: [@navanil-das](https://github.com/navanil-das)
