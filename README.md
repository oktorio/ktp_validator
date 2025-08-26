# KTP Validator

KTP photocopy/color detection **and** text-legibility scoring with a single **Final Score (0–100)**.  
Optimised for Indonesian ID cards (KTP) captured by mobile or scanner.

**File:** `ktp_validator_v1_0.py`  
**Python:** 3.10+  
**OS:** Windows / macOS / Linux

---

## ✨ What it does

1. **Document type classification**

   * `color_document` (KTP warna, given higher weight)
   * `grayscale_document`
   * `photocopy_on_colored_background`
   * Hard rejects: `black`, `white`
2. **Text legibility analysis (value side)**

   * Sharpness (variance of Laplacian)
   * Edge density, RMS contrast
   * OCR confidence (Tesseract, if available)
   * Text coverage, MSER text density, blockiness, censor/blur region detection
3. **Final Score (0–100)**

   * Combines legibility score with **document-type multiplier** and **legibility multiplier**
   * Penalises unclear text more strongly
   * Outputs `final.score`, `final.label`, and detailed reasons/multipliers

---

## ⚡ Quick start

```bash
# 1) (optional) create and activate a virtual env
# python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run on an image or a folder
python ktp_validator_v1_0.py path/to/image.jpg --verbose --save-debug debug_out
# or
python ktp_validator_v1_0.py path/to/folder --verbose --save-debug debug_out
```

> `--save-debug` stores rectified crops and masks (helpful for QA).  
> `--verbose` shows extra console info in plain mode.

---

## 📦 Requirements

* `opencv-python`, `numpy`, `pytesseract` (see `requirements.txt`)
* **Tesseract OCR** engine for OCR features (optional, but recommended)

  * Windows: install Tesseract and ensure `tesseract.exe` is on PATH.
  * macOS: `brew install tesseract`
  * Linux: `sudo apt-get install tesseract-ocr`

If Tesseract is not on PATH, you can point the script manually inside Python:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

> If Tesseract is missing, the script still runs; OCR confidence will be `null` and OCR-based features degrade gracefully.

---

## 🧪 Usage

```
python ktp_validator_v1_0.py <paths...> [--plain] [--verbose] [--save-debug DIR]
```

**Args**

* `paths` : one or more files or directories (recursive)
* `--plain` : print human-readable lines instead of JSON
* `--verbose` : more console details (applies to `--plain`)
* `--save-debug DIR` : write rectified image and masks to `DIR`

**Examples**

```bash
# JSON output (default)
python ktp_validator_v1_0.py samples/ktp.jpg

# Plain output for terminals/logs
python ktp_validator_v1_0.py samples --plain --verbose

# Save debug artifacts
python ktp_validator_v1_0.py samples/ --save-debug debug_out
```

---

## 📤 Output (JSON shape)

```json
{
  "file": "ktp.jpg",
  "label": "color_document",
  "stats": {
    "doc_meanS": 90.9,
    "doc_p90S": 132.0,
    "doc_colored_frac": 0.95,
    "bg_meanS": 64.0
  },
  "text_legibility": {
    "score": 60.5,
    "label": "fair",
    "sharpness_vlap": 122.3,
    "edge_density": 0.1296,
    "rms_contrast": 0.2277,
    "ocr_conf": null,
    "warped_size": [800, 500],
    "censor_area_frac": 0.0,
    "blockiness": 0.064,
    "mser_per_megapixel": 4117.6,
    "text_coverage": 0.309,
    "text_pixels_frac": 0.263,
    "value_pixels_frac": 0.263
  },
  "final": {
    "score": 68.5,
    "label": "fair",
    "multipliers": { "doc": 1.15, "text": 0.85, "combined": 0.978 },
    "reason": "color_document; text=fair"
  },
  "version": "1.0"
}
```

**Label thresholds**

* `excellent` ≥ 85  
* `good` ≥ 70  
* `fair` ≥ 55  
* otherwise `poor`

---

## 🎛️ Tuning (high-level knobs)

Open `ktp_validator_v1_0.py` and edit these dicts near the top.

### Document-type weight (warna vs fotokopi)

```python
DOC_MULTIPLIERS = {
    "color_document": 1.15,   # boost for KTP warna
    "grayscale_document": 0.85,
    "photocopy_on_colored_background": 0.70,
    "black": 0.00,
    "white": 0.00,
}
```

### Legibility penalty (final-stage)

```python
LEGIBILITY_MULTIPLIERS = {
    "excellent": 1.00,
    "good": 1.00,
    "fair": 0.85,   # stricter on unclear text
    "poor": 0.70,
}
```

---

## 🔬 Advanced tunables (fine control)

All under the **Tunables** section:

* **OCR gating**

  ```python
  OCR_LOW_CONF_THR = 40.0
  OCR_LOW_CONF_PENALTY_W = 20.0
  ```

* **Feature weights**

  ```python
  W_SHARP, W_EDGE, W_CONTR, W_OCR = 0.35, 0.20, 0.15, 0.30
  ```

* **Penalty weights**

  ```python
  MSER_PENALTY_W = 25.0
  MSER_OVER_PENALTY_W = 10.0
  BLOCKINESS_PENALTY_W = 8.0
  TEXT_COVERAGE_PENALTY_W = 12.0
  CENSOR_PENALTY_W = 45.0
  ```

* **Color/photocopy detection thresholds**

  ```python
  DOC_P90_SAT_THR = 50
  DOC_COLOR_FRACTION_THR = 0.12
  BG_MEAN_SAT_THR = 25
  ```

---

## 🧰 Debug artifacts

When `--save-debug DIR` is provided, the script writes:

* `<name>_rectified.png` : perspective-corrected card crop
* `<name>_textmask.png` : detected text strokes
* `<name>_valuemask.png` : value-side mask used for scoring

---

## 🛠️ Troubleshooting

* **`Failed to read image`**: path/format issue. Ensure image is readable by OpenCV.
* **OCR shows `null`**: Tesseract not installed or not on PATH.
* **Everything scores very low**: ensure ≥1000px width, good lighting, avoid reflections.
* **Detected as `black`/`white`**: entire image is near solid colour; check capture pipeline.

---

## 🧭 Roadmap ideas

* External `config.json` loader for tuning without code edits  
* Batch summary report (CSV/Excel)  
* Face/photo region quality checks (glare, shadow)

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

* OpenCV for image processing  
* Tesseract OCR (optional) for text confidence  

