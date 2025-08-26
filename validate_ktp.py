#!/usr/bin/env python3
import os, sys, argparse, json, math
import numpy as np
import cv2

# --- Optional OCR ---
try:
    import pytesseract
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False

# =================== Tunables ===================
NEAR_PURE_TOL = 3
NEAR_PURE_ALLOWED_FRACTION = 0.01

# Photocopy / color detection (robust)
DOC_P90_SAT_THR        = 50     # inside-doc p90(S) <= 50 => grayscale/photocopy
DOC_COLOR_FRACTION_THR = 0.12   # >=12% colored pixels inside => color_document
COLOR_PIXEL_S_THR      = 35     # pixel considered "colored" if S > 35
BG_MEAN_SAT_THR        = 25     # outside mean S > 25 => background colored

# Document geometry detection
MIN_DOC_AREA_FRACTION  = 0.20
ASPECT_MIN, ASPECT_MAX = 1.2, 2.2
INSET_FRACTION         = 0.08   # inset polygon inward to avoid colored borders

# Legibility score weights
W_SHARP, W_EDGE, W_CONTR, W_OCR = 0.40, 0.25, 0.20, 0.15
# Legibility normalization anchors
SHARP_LO_HI  = (80, 600)        # variance of Laplacian
EDGE_LO_HI   = (0.02, 0.09)     # Canny edge density
CONTR_LO_HI  = (0.05, 0.22)     # RMS contrast

# =================================================

def _norm_0_100(x, lo, hi):
    if hi <= lo: return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0, 1) * 100.0)

def near_black_white_label(bgr):
    lum = bgr.mean(axis=2)
    nb = (lum <= NEAR_PURE_TOL).sum()
    nw = (lum >= 255 - NEAR_PURE_TOL).sum()
    total = lum.size
    if nb / total >= 1 - NEAR_PURE_ALLOWED_FRACTION: return "black"
    if nw / total >= 1 - NEAR_PURE_ALLOWED_FRACTION: return "white"
    return None

def find_document_quad(bgr):
    h, w = bgr.shape[:2]
    scale = 900 / max(h, w)
    small = cv2.resize(bgr, (int(w*scale), int(h*scale)), cv2.INTER_AREA) if scale < 1 else bgr.copy()
    if scale >= 1: scale = 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1.0
    img_area = small.shape[0]*small.shape[1]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) != 4: continue
        area = cv2.contourArea(approx)
        if area < img_area * MIN_DOC_AREA_FRACTION: continue
        x,y,w2,h2 = cv2.boundingRect(approx)
        aspect = w2 / float(h2) if h2 else 0
        if ASPECT_MIN <= aspect <= ASPECT_MAX:
            rectness = area / (w2*h2 + 1e-6)
            score = area * rectness
            if score > best_score:
                best, best_score = approx, score

    if best is None: return None
    quad = (best.reshape(-1,2) / scale).astype(np.float32)
    return quad

def inset_polygon(poly, frac):
    c = poly.mean(axis=0, keepdims=True)
    return c + (poly - c) * (1.0 - frac)

def polygon_mask(img_shape, poly):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask

def sat_stats(bgr, mask=None):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[...,1].astype(np.float32)
    vals = s[mask>0] if mask is not None else s.reshape(-1)
    if vals.size == 0:
        return dict(mean=0.0, p90=0.0, colored_frac=0.0)
    return dict(
        mean=float(vals.mean()),
        p90=float(np.percentile(vals, 90)),
        colored_frac=float((vals > COLOR_PIXEL_S_THR).mean())
    )

def classify_document(bgr, verbose=False):
    label = near_black_white_label(bgr)
    if label:
        if verbose: print(f"[global] near-pure -> {label}")
        return label, None, None, None, None

    h, w = bgr.shape[:2]
    quad = find_document_quad(bgr)
    if quad is None:
        my, mx = int(h*0.12), int(w*0.10)
        poly = np.array([[mx, my],[w-mx, my],[w-mx, h-my],[mx, h-my]], dtype=np.float32)
    else:
        poly = quad

    poly_in = inset_polygon(poly, INSET_FRACTION)

    doc_mask = polygon_mask(bgr.shape, poly_in)
    bg_mask  = cv2.bitwise_not(polygon_mask(bgr.shape, poly))

    doc = sat_stats(bgr, doc_mask)
    bg  = sat_stats(bgr, bg_mask)

    if verbose:
        print(f"[doc] meanS={doc['mean']:.1f} p90S={doc['p90']:.1f} colored_frac={doc['colored_frac']:.3f}")
        print(f"[bg ] meanS={bg['mean']:.1f} p90S={bg['p90']:.1f}")

    if doc['p90'] <= DOC_P90_SAT_THR:
        return ("photocopy_on_colored_background" if bg['mean'] >= BG_MEAN_SAT_THR else "grayscale_document",
                poly, poly_in, doc, bg)
    else:
        if doc['colored_frac'] >= DOC_COLOR_FRACTION_THR:
            return ("color_document", poly, poly_in, doc, bg)
        else:
            return ("grayscale_document", poly, poly_in, doc, bg)

# ---------- Text legibility scoring ----------
def _variance_of_laplacian(gray):  return float(cv2.Laplacian(gray, cv2.CV_64F).var())
def _edge_density(gray):           return float((cv2.Canny(gray,60,180) > 0).mean())
def _rms_contrast(gray):
    g = gray.astype(np.float32)/255.0
    return float(g.std())

def _ocr_confidence(gray):
    if not _HAS_TESS: return None
    cfg = "--psm 6 --oem 3 -l ind+eng"
    d = pytesseract.image_to_data(gray, config=cfg, output_type=pytesseract.Output.DICT)
    confs = [int(c) for c in d.get("conf", []) if c not in ("-1", -1)]
    if not confs: return 0.0
    return float(np.median(confs))  # already 0..100

def _order_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl,tr,br,bl], dtype=np.float32)

def score_text_legibility(bgr, doc_poly):
    q = _order_quad(doc_poly)
    wA = np.linalg.norm(q[1]-q[0]); wB = np.linalg.norm(q[2]-q[3])
    hA = np.linalg.norm(q[3]-q[0]); hB = np.linalg.norm(q[2]-q[1])
    W = int(np.clip(max(wA,wB), 800, 1800))
    H = int(np.clip(max(hA,hB), 500, 1200))
    M = cv2.getPerspectiveTransform(q, np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32))
    top = cv2.warpPerspective(bgr, M, (W,H), flags=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    text_roi = gray[:, :int(W*0.72)]                # skip portrait area
    text_roi = cv2.fastNlMeansDenoising(text_roi, None, 7, 7, 21)
    text_norm = cv2.equalizeHist(text_roi)

    sharp = _variance_of_laplacian(text_norm)
    edges = _edge_density(text_norm)
    contr = _rms_contrast(text_norm)
    ocr_c = _ocr_confidence(text_norm)

    s_sharp = _norm_0_100(sharp, *SHARP_LO_HI)
    s_edges = _norm_0_100(edges, *EDGE_LO_HI)
    s_contr = _norm_0_100(contr, *CONTR_LO_HI)
    parts = [s_sharp*W_SHARP, s_edges*W_EDGE, s_contr*W_CONTR]
    denom = W_SHARP + W_EDGE + W_CONTR

    if ocr_c is not None:
        parts.append(ocr_c*W_OCR)
        denom += W_OCR

    score = float(np.clip(sum(parts)/denom, 0, 100))

    label = "excellent" if score>=85 else "good" if score>=70 else "fair" if score>=55 else "poor"
    return dict(
        score=round(score,1),
        label=label,
        sharpness_vlap=round(sharp,1),
        edge_density=round(edges,4),
        rms_contrast=round(contr,4),
        ocr_conf=(round(ocr_c,1) if ocr_c is not None else None),
        warped_size=[W,H]
    )

# -------------------- CLI ----------------------
def analyze_file(path, verbose=False, want_json=True):
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to read image")

    label, poly, poly_in, doc_stats, bg_stats = classify_document(bgr, verbose=verbose)

    # Choose the polygon used for text assessment (inner one to avoid borders)
    if poly_in is None:
        # construct a centered fallback if all-black/white
        h, w = bgr.shape[:2]
        my, mx = int(h*0.12), int(w*0.10)
        poly_in = np.array([[mx,my],[w-mx,my],[w-mx,h-my],[mx,h-my]], dtype=np.float32)

    text = score_text_legibility(bgr, poly_in)

    out = {
        "file": os.path.basename(path),
        "label": label,
        "stats": {
            "doc_meanS": None if doc_stats is None else round(doc_stats["mean"],1),
            "doc_p90S":  None if doc_stats is None else round(doc_stats["p90"],1),
            "doc_colored_frac": None if doc_stats is None else round(doc_stats["colored_frac"],3),
            "bg_meanS":  None if bg_stats  is None else round(bg_stats["mean"],1)
        },
        "text_legibility": text
    }
    if want_json:
        return json.dumps(out, ensure_ascii=False)
    else:
        lines = []
        lines.append(f"{out['file']}: {out['label']}")
        if verbose and doc_stats is not None:
            lines.append(f"  [doc] meanS={out['stats']['doc_meanS']} p90S={out['stats']['doc_p90S']} colored_frac={out['stats']['doc_colored_frac']}")
            lines.append(f"  [bg ] meanS={out['stats']['bg_meanS']}")
        t = out["text_legibility"]
        lines.append(f"  text_score={t['score']} ({t['label']}) sharp={t['sharpness_vlap']} edge={t['edge_density']} contr={t['rms_contrast']}"
                     + (f" ocr={t['ocr_conf']}" if t['ocr_conf'] is not None else ""))
        return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="KTP photocopy/color detection + text legibility score")
    ap.add_argument("paths", nargs="+", help="image file(s) or folder(s)")
    ap.add_argument("--plain", action="store_true", help="print human-readable instead of JSON")
    ap.add_argument("--verbose", action="store_true", help="print debug stats")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for n in names:
                    if n.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp")):
                        files.append(os.path.join(root, n))
        else:
            files.append(p)

    if not files:
        print(json.dumps({"error":"no_image_files_found"}))
        sys.exit(2)

    want_json = not args.plain
    for f in files:
        try:
            print(analyze_file(f, verbose=args.verbose, want_json=want_json))
        except Exception as e:
            err = {"file": os.path.basename(f), "error": str(e)}
            print(json.dumps(err) if want_json else f"{err['file']}: ERROR -> {err['error']}")

if __name__ == "__main__":
    main()
