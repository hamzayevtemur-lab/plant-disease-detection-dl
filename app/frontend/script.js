// Global State
let selectedImageFile = null;
let selectedImageUrl = null;
let currentActiveModel = "";

// DOM Elements
const dropzone = document.getElementById("dropzone");
const dropzoneContent = document.getElementById("dropzoneContent");
const fileInput = document.getElementById("fileInput");
const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");
const removeImageBtn = document.getElementById("removeImageBtn");
const diagnoseBtn = document.getElementById("diagnoseBtn");
const btnText = document.getElementById("btnText");
const btnSpinner = document.getElementById("btnSpinner");
const imageUrlInput = document.getElementById("imageUrlInput");
const loadUrlBtn = document.getElementById("loadUrlBtn");

const emptyResultState = document.getElementById("emptyResultState");
const oodWarningBanner = document.getElementById("oodWarningBanner");
const oodWarningText = document.getElementById("oodWarningText");
const resultContent = document.getElementById("resultContent");

const cropName = document.getElementById("cropName");
const diseaseName = document.getElementById("diseaseName");
const healthBadge = document.getElementById("healthBadge");
const healthIcon = document.getElementById("healthIcon");
const healthStatus = document.getElementById("healthStatus");

const confidenceValue = document.getElementById("confidenceValue");
const latencyValue = document.getElementById("latencyValue");
const activeModelLabel = document.getElementById("activeModelLabel");

const symptomsText = document.getElementById("symptomsText");
const organicText = document.getElementById("organicText");
const chemicalText = document.getElementById("chemicalText");
const top5List = document.getElementById("top5List");
const modelSelect = document.getElementById("modelSelect");

// Initialize on Load
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadModels();
  loadBenchmark();
});

function setupEventListeners() {
  // Drag & drop
  ["dragenter", "dragover"].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  removeImageBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    clearImage();
  });

  modelSelect.addEventListener("change", (e) => {
    if (e.target.value) {
      switchActiveModel(e.target.value);
    }
  });

  // URL Enter key press
  if (imageUrlInput) {
    imageUrlInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleUrlInput();
      }
    });
  }
}

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please upload a valid image file (JPEG, PNG, WebP).");
    return;
  }
  selectedImageFile = file;
  selectedImageUrl = null;
  if (imageUrlInput) imageUrlInput.value = "";

  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    dropzoneContent.classList.add("hidden");
    previewContainer.classList.remove("hidden");
    diagnoseBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function handleUrlInput() {
  const url = imageUrlInput ? imageUrlInput.value.trim() : "";
  if (!url) {
    alert("Please enter a valid image URL.");
    return;
  }

  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    alert("Please enter a full URL starting with http:// or https://");
    return;
  }

  loadUrlBtn.textContent = "Loading...";
  loadUrlBtn.disabled = true;

  // Test load the image into an Image element to verify it can be displayed
  const testImg = new Image();
  testImg.crossOrigin = "anonymous";
  testImg.onload = () => {
    selectedImageUrl = url;
    selectedImageFile = null;
    fileInput.value = "";

    imagePreview.src = url;
    dropzoneContent.classList.add("hidden");
    previewContainer.classList.remove("hidden");
    diagnoseBtn.disabled = false;

    loadUrlBtn.textContent = "Load URL";
    loadUrlBtn.disabled = false;
  };

  testImg.onerror = () => {
    // Some CORS-restricted images might still load on the backend, so set it anyway with warning
    selectedImageUrl = url;
    selectedImageFile = null;
    fileInput.value = "";

    imagePreview.src = url;
    dropzoneContent.classList.add("hidden");
    previewContainer.classList.remove("hidden");
    diagnoseBtn.disabled = false;

    loadUrlBtn.textContent = "Load URL";
    loadUrlBtn.disabled = false;
  };

  testImg.src = url;
}

function clearImage() {
  selectedImageFile = null;
  selectedImageUrl = null;
  fileInput.value = "";
  if (imageUrlInput) imageUrlInput.value = "";
  imagePreview.src = "";
  previewContainer.classList.add("hidden");
  dropzoneContent.classList.remove("hidden");
  diagnoseBtn.disabled = true;
  
  resultContent.classList.add("hidden");
  oodWarningBanner.classList.add("hidden");
  emptyResultState.classList.remove("hidden");
}

async function runDiagnosis() {
  if (!selectedImageFile && !selectedImageUrl) return;

  // UI Loading State
  btnText.textContent = "Analyzing Leaf...";
  btnSpinner.classList.remove("hidden");
  diagnoseBtn.disabled = true;

  try {
    let response;
    if (selectedImageUrl) {
      response = await fetch("/predict-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: selectedImageUrl })
      });
    } else {
      const formData = new FormData();
      formData.append("file", selectedImageFile);
      response = await fetch("/predict", {
        method: "POST",
        body: formData
      });
    }

    if (!response.ok) {
      const errData = await response.json().catch(() => ({ detail: `Inference failed (${response.status})` }));
      throw new Error(errData.detail || `Inference failed (${response.status})`);
    }

    const data = await response.json();
    renderDiagnosisResults(data);
  } catch (err) {
    alert(`Error during diagnosis: ${err.message}`);
  } finally {
    btnText.textContent = "Analyze & Diagnose Leaf";
    btnSpinner.classList.add("hidden");
    diagnoseBtn.disabled = false;
  }
}

function renderDiagnosisResults(data) {
  emptyResultState.classList.add("hidden");
  resultContent.classList.remove("hidden");

  // Non-leaf / OOD Warning Banner
  if (data.is_valid_leaf === false) {
    oodWarningBanner.classList.remove("hidden");
    oodWarningText.textContent = data.validation_warning || "No plant foliage detected (0% green/leaf pixels found). The model cannot diagnose non-botanical screenshots or documents.";
    
    cropName.textContent = "NON-PLANT IMAGE";
    diseaseName.textContent = "Invalid / Non-Leaf Input";
    healthBadge.className = "health-badge diseased";
    healthIcon.textContent = "⚠️";
    healthStatus.textContent = "Rejected (Non-Plant)";
    
    confidenceValue.textContent = "0.0%";
    latencyValue.textContent = `${data.latency_ms} ms`;
    activeModelLabel.textContent = data.model_used || "mobilenet_v3";
    
    symptomsText.textContent = "The uploaded file is a document, UI screenshot, or non-botanical picture. Deep learning models trained on leaf pathology only operate on plant foliage.";
    organicText.textContent = "Please upload a clear, focused photograph of a crop leaf.";
    chemicalText.textContent = "Please upload a clear, focused photograph of a crop leaf.";
    
    top5List.innerHTML = `<div style="color: #9ca3af; font-size: 0.85rem; padding: 0.5rem 0;">No predictions generated for non-plant images.</div>`;
    return;
  }

  oodWarningBanner.classList.add("hidden");

  // Header Details
  cropName.textContent = data.crop;
  diseaseName.textContent = data.disease;

  if (data.is_healthy) {
    healthBadge.className = "health-badge healthy";
    healthIcon.textContent = "✓";
    healthStatus.textContent = "Healthy Foliage";
  } else {
    healthBadge.className = "health-badge diseased";
    healthIcon.textContent = "⚠️";
    healthStatus.textContent = "Disease Detected";
  }

  // Stats
  confidenceValue.textContent = `${data.confidence}%`;
  latencyValue.textContent = `${data.latency_ms} ms`;
  activeModelLabel.textContent = data.model_used || "mobilenet_v3";

  // Symptoms & Remedies
  symptomsText.textContent = data.symptoms;
  organicText.textContent = data.organic_treatment;
  chemicalText.textContent = data.chemical_treatment;

  // Top 5 Probabilities
  top5List.innerHTML = "";
  data.top5.forEach((item) => {
    const itemEl = document.createElement("div");
    itemEl.className = "top5-item";
    itemEl.innerHTML = `
      <div class="top5-header">
        <span class="top5-label">${item.crop} — ${item.disease}</span>
        <span class="top5-pct">${item.confidence}%</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" style="width: ${item.confidence}%"></div>
      </div>
    `;
    top5List.appendChild(itemEl);
  });
}

// Model Switcher API
async function loadModels() {
  try {
    const res = await fetch("/models");
    const data = await res.json();
    currentActiveModel = data.active_model;

    modelSelect.innerHTML = "";
    data.models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = `${m.name} (${m.test_accuracy})`;
      if (m.name === currentActiveModel) {
        opt.selected = true;
      }
      modelSelect.appendChild(opt);
    });
  } catch (err) {
    console.error("Error loading models:", err);
  }
}

async function switchActiveModel(modelName) {
  try {
    const res = await fetch("/models/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: modelName })
    });
    if (res.ok) {
      currentActiveModel = modelName;
      if (selectedImageFile) {
        runDiagnosis();
      }
    }
  } catch (err) {
    alert(`Could not switch model: ${err.message}`);
  }
}

// Benchmark Leaderboard API
async function loadBenchmark() {
  try {
    const res = await fetch("/benchmark");
    const data = await res.json();
    const tbody = document.getElementById("benchmarkTableBody");
    tbody.innerHTML = "";

    if (!data.results || data.results.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="text-center">No benchmark results found. Run src.training.benchmark to populate.</td></tr>`;
      return;
    }

    data.results.forEach((row, idx) => {
      const tr = document.createElement("tr");
      const isTop = idx === 0;
      const valAcc = (parseFloat(row.validation_accuracy || 0) * 100).toFixed(2) + "%";
      const testAcc = (parseFloat(row.test_accuracy || 0) * 100).toFixed(2) + "%";
      const latency = row.latency_ms ? `${row.latency_ms} ms` : "N/A";
      const size = row.size_mb ? `${row.size_mb} MB` : "N/A";
      const params = parseInt(row.parameters || 0).toLocaleString();

      tr.innerHTML = `
        <td><strong>${row.experiment}</strong></td>
        <td>${params}</td>
        <td>${size}</td>
        <td>${latency}</td>
        <td>${valAcc}</td>
        <td><strong style="color: #34d399;">${testAcc}</strong></td>
        <td>${parseFloat(row.test_loss || 0).toFixed(4)}</td>
        <td>${isTop ? '<span class="badge-top">🏆 Best Model</span>' : '<span style="color: #9ca3af;">Trained</span>'}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Error loading benchmark:", err);
  }
}

// Navigation Tabs
function switchTab(tabId) {
  const diagSection = document.getElementById("sectionDiagnose");
  const benchSection = document.getElementById("sectionBenchmark");
  const tabDiag = document.getElementById("tabDiagnose");
  const tabBench = document.getElementById("tabBenchmark");

  if (tabId === "diagnose") {
    diagSection.classList.remove("hidden");
    benchSection.classList.add("hidden");
    tabDiag.classList.add("active");
    tabBench.classList.remove("active");
  } else {
    diagSection.classList.add("hidden");
    benchSection.classList.remove("hidden");
    tabDiag.classList.remove("active");
    tabBench.classList.add("active");
    loadBenchmark();
  }
}
