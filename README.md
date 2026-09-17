# 🌿 FloraGuard AI — Plant Disease Detection & Crop Diagnosis

An end-to-end Deep Learning & Computer Vision pipeline to detect and classify **38 crop diseases** from leaf imagery using the **PlantVillage** dataset, complete with an interactive, production-ready **FastAPI web application**.

---

## 🏆 Model Benchmark Leaderboard

| Model Architecture | Total Params | Model Size | Latency (ms) | Validation Acc | **Test Accuracy** | Test Loss | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large** | 4,250,710 | 49.0 MB | 8.7 ms | 99.71% | **99.79%** | 0.0080 | 🥇 **Production Champion** |
| **Pretrained ResNet-18** | 11,196,006 | 128.3 MB | 4.5 ms | 99.69% | **99.62%** | 0.0114 | 🥈 Runner-up |
| **Deeper CNN + AdamW** | 1,590,054 | 18.2 MB | 2.2 ms | 98.93% | **99.09%** | 0.0275 | 🥉 Best Custom CNN |
| **Deeper CNN (Baseline)** | 1,590,054 | 18.2 MB | 2.6 ms | 97.21% | **97.71%** | 0.0747 | Custom CNN Baseline |
| **Baseline 4-Layer CNN** | 399,142 | 4.6 MB | 2.2 ms | 95.91% | **96.01%** | 0.1206 | Initial Baseline |

---

## ✨ Key Capabilities

- 🔬 **Real-Time 38-Class Diagnosis**: Instant disease identification with high confidence calibration.
- 🔗 **Dual Input Modes**: Upload local leaf images or paste any direct web image URL address.
- 🛡️ **Botanical Foliage & OOD Verification**: Automatic rejection of non-plant screenshots/documents to prevent overconfident false positives.
- 🎛️ **Active Model Switching**: Switch dynamically between all 5 trained models in the UI.
- 💊 **Actionable Agronomy**: Integrated organic and chemical treatment remedies for every diagnosis.
- 📊 **Benchmark Leaderboard Tab**: Live view of parameter sizes, latency, and test accuracy metrics.

---

## 🚀 Quickstart

### 1. Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Launch FastAPI app
python3 -m uvicorn app.main:app --reload --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

### 2. Run with Docker
```bash
docker build -t floraguard-ai .
docker run -p 8000:7860 floraguard-ai
```

### 3. Cloud Deployment
Detailed 1-click deployment guides for **Hugging Face Spaces** and **Render.com** are available in [`DEPLOYMENT.md`](file:///Users/mac/Desktop/projects/plant-disease-detection-dl/DEPLOYMENT.md).

---

## 🏛️ Model Architectures & Experiment Pipeline

1. **Pretrained CNNs (Transfer Learning)**:
   - [`mobilenet_v3.py`](file:///Users/mac/Desktop/projects/plant-disease-detection-dl/src/experiments/mobilenet_v3.py) — MobileNetV3-Large backbone (~4.25M params, **99.79%** test accuracy).
   - [`resnet18.py`](file:///Users/mac/Desktop/projects/plant-disease-detection-dl/src/experiments/resnet18.py) — ResNet-18 backbone (~11.2M params, **99.62%** test accuracy).
2. **Custom Deep CNNs**:
   - [`deeper_cnn_adamw.py`](file:///Users/mac/Desktop/projects/plant-disease-detection-dl/src/experiments/deeper_cnn_adamw.py) — 5-layer CNN with AdamW & Cosine Annealing scheduler (**99.09%** test accuracy).
   - [`deeper_cnn.py`](file:///Users/mac/Desktop/projects/plant-disease-detection-dl/src/experiments/deeper_cnn.py) — 5-layer CNN baseline (**97.71%** test accuracy).
   - [`baseline.py`](file:///Users/mac/Desktop/projects/plant-disease-detection-dl/src/experiments/baseline.py) — 4-layer CNN baseline (**96.01%** test accuracy).

