# AI Skin: Intelligent Skin Diseases Detection — Demonstration Checklist

Use this checklist during your final-year academic project presentation and viva examination.

---

## 🎯 1. System Initialization & Architecture
- [ ] **Python Environment**: Confirm Python 3.11 virtual environment (`venv`) is activated.
- [ ] **Hardware Acceleration**: Verify Apple Silicon MPS (`mps:0`) is reported by `GET /api/health`.
- [ ] **Model Loading**: Confirm `models/mobilenetv2_skin_disease_best.pth` and `models/class_mapping.json` are loaded.
- [ ] **Frontend**: Confirm Vite server running on `http://localhost:5173`.

---

## 🔬 2. Live Clinical Demonstration Steps

### Step 1: Baseline Health Check
- Open `http://localhost:5173` in Chrome/Safari.
- Point out the **Navbar status badge**: shows `Backend Healthy`, `Apple Silicon (MPS)` acceleration, and `7 Diagnostic Classes`.

### Step 2: Upload Dermatoscopic Lesion
- Drag and drop `data/sample_images/sample_lesion.jpg` into the uploader area.
- Point out the **instant client-side preview** and image dimension display.

### Step 3: Run AI Diagnostic & Grad-CAM
- Click **"Run AI Diagnostic & Grad-CAM"**.
- Point out the real-time loading indicator (`Analyzing Lesion & Generating Grad-CAM...`).
- Inspect the **Primary Diagnostic Output Card**:
  - Predicted disease name and clinical risk tier (e.g. *Precancerous / High Risk*).
  - Confidence percentage (computed from softmax probabilities).
  - Multiclass probability distribution bars across all 7 classes.
  - Inference latency (measured in milliseconds) and compute device (`mps`).

### Step 4: Explainability & Grad-CAM Analysis
- Scroll to the **Grad-CAM Saliency Heatmap Viewer**.
- Toggle between the 3 view modes:
  - **Side by Side**: Demonstrates direct comparison between the dermatoscope image and AI attention regions.
  - **Grad-CAM Overlay**: Displays the alpha-blended heatmap highlighting high-salience features.
  - **Original**: Displays the uncropped native photograph.
- Highlight the **Target Layer**: Dynamically identified as `features.18.0`.

### Step 5: Counterfactual / Target-Class Inspection
- Click on another disease in the probability bars (e.g. **Melanoma** `mel`).
- Demonstrate how Grad-CAM re-queries the model's gradient flow to answer: *"What features in this lesion contribute to a suspicion of Melanoma?"*

### Step 6: Defensive Validation & Error Handling
- Attempt to upload an invalid file (e.g. text file or 0-byte image).
- Demonstrate that the system displays a clear, defensive validation alert without crashing.
- Click the **"Reset / New Analysis"** button to restore a clean state.

---

## 📊 3. Academic Viva Discussion Topics
- **Why MobileNetV2?** Depthwise separable convolutions provide high representational capacity with only 2.55M parameters, enabling fast edge inference on laptops and mobile devices without requiring expensive server clusters.
- **Why Two-Stage Transfer Learning?** Freezing the backbone in Stage 1 prevents catastrophic forgetting of ImageNet visual priors while initializing the custom head. Fine-tuning upper blocks in Stage 2 adapts high-level feature representations to dermatoscopic pathology.
- **How was Class Imbalance Handled?** Stratified patient-grouped splitting prevented lesion leakage across splits, and balanced cross-entropy loss weights penalized false negatives on critical minority classes (e.g. Melanoma).
- **What is the Role of Grad-CAM?** Clinicians cannot safely rely on black-box predictions. Grad-CAM provides visual verification that the model focused on authentic lesion borders rather than background artifacts (hair, ruler marks, gel bubbles).
- **Ethical & Safety Notice**: Clearly reiterate that AI Skin is a decision-support and preliminary screening tool that requires confirmation by expert histopathology.
