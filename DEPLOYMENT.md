# 🚀 FloraGuard AI — Cloud Deployment Guide

This guide provides step-by-step instructions for deploying FloraGuard AI to the cloud for free.

---

## 🌟 Option 1: Hugging Face Spaces (Recommended for AI Portfolios — 100% Free)

Hugging Face Spaces provides free hosting specifically built for Machine Learning and Deep Learning web applications.

### Steps:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Fill in the details:
   - **Space Name**: `floraguard-ai` (or your choice)
   - **License**: `mit` / `apache-2.0`
   - **Space SDK**: Select **Docker** (Blank)
   - **Space hardware**: Free CPU tier (2 vCPU, 16GB RAM)
3. Clone your new Hugging Face Space repository locally:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/floraguard-ai hf-space
   ```
4. Copy the project files (including `Dockerfile`, `app/`, `experiments/`, `src/`, `requirements-prod.txt`) into the `hf-space` folder:
   ```bash
   cp -r Dockerfile app experiments src requirements-prod.txt app.py hf-space/
   ```
5. Commit and push:
   ```bash
   cd hf-space
   git add .
   git commit -m "Deploy FloraGuard AI"
   git push
   ```
6. Hugging Face will automatically build your Docker container and launch the live web app with a public URL!

---

## ☁️ Option 2: Render.com (Automatic GitHub Continuous Deployment)

Render allows you to deploy Docker containers directly connected to your GitHub repository.

### Steps:
1. Push your project code to a GitHub repository:
   ```bash
   git add .
   git commit -m "Add production deployment configurations"
   git push origin main
   ```
2. Go to [render.com](https://render.com) and sign in with your GitHub account.
3. Click **"New +"** $\rightarrow$ **"Web Service"**.
4. Connect your GitHub repository.
5. Choose **Docker** environment (Render will automatically detect the `Dockerfile` and `render.yaml`).
6. Select the **Free** instance type.
7. Click **"Create Web Service"**.
8. In 2–3 minutes, Render will provide a live URL like `https://floraguard-ai.onrender.com`.

---

## 🐳 Option 3: Run with Docker Locally

To test the containerized production build locally on your machine:

```bash
# Build the Docker image
docker build -t floraguard-ai .

# Run the container on port 8000
docker run -p 8000:7860 floraguard-ai
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 📦 What is Included for Deployment

- `Dockerfile`: Multi-stage Python 3.11 slim image with CPU-optimized PyTorch wheels (keeps container lightweight and fast).
- `requirements-prod.txt`: Lean dependencies list without training-only overhead.
- `.dockerignore`: Prevents raw datasets and virtual environments from bloating the build.
- `render.yaml`: Blueprint configuration for Render.
- `app.py`: Standard entrypoint configured with dynamic cloud `$PORT` routing.
