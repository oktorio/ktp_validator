#!/usr/bin/env python3
# ktp_validator_v1_0.py
# Version: 1.0
# Desc   : KTP photocopy/color detection + value-text legibility + Final Score (0..100)
#          - Color KTP boost (color_document)
#          - Explicit final-stage penalty for unclear text (fair/poor)
#          - Outputs separate multipliers (doc vs text) and combined reason

__version__ = "1.5"

import os, sys, argparse, json
import numpy as np
import cv2

# ---------- Optional OCR ----------
try:
    import pytesseract
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False

# =================== Tunables ===================
NEAR_PURE_TOL = 3
NEAR_PURE_ALLOWED_FRACTION = 0.01

# Photocopy / color detection (robust)
DOC_P90_SAT_THR        = 50
DOC_COLOR_FRACTION_THR = 0.12
COLOR_PIXEL_S_THR      = 35
BG_MEAN_SAT_THR        = 25

# Document geometry detection
MIN_DOC_AREA_FRACTION  = 0.20
ASPECT_MIN, ASPECT_MAX = 1.2, 2.2
INSET_FRACTION         = 0.08

# Legibility score weights (OCR weighted more for real KTPs)
W_SHARP, W_EDGE, W_CONTR, W_OCR = 0.32, 0.26, 0.22, 0.20
SHARP_LO_HI  = (40, 360)
EDGE_LO_HI   = (0.02, 0.09)
CONTR_LO_HI  = (0.05, 0.22)

# Penalties
CENSOR_MIN_AREA_FRAC   = 0.010
CENSOR_UNIFORM_STD_THR = 8.0
CENSOR_PENALTY_W       = 45.0

PIXEL_GRID_SIZES       = [8, 12, 16, 20, 24]
BLOCKINESS_PENALTY_W   = 6.0   # lighter for patterned cards

MSER_MIN_COUNT_PER_MP  = 160.0
MSER_PENALTY_W         = 25.0
MSER_MAX_COUNT_PER_MP  = 12000.0
MSER_OVER_PENALTY_W    = 6.0

OCR_LOW_CONF_THR       = 30.0
OCR_LOW_CONF_PENALTY_W = 12.0

TEXT_COVERAGE_RANGE    = (0.01, 0.20)
TEXT_COVERAGE_PENALTY_W= 12.0

# ---- Final-score multipliers (easy knobs) ----
DOC_MULTIPLIERS = {
    "color_document": 1.40,  # boosted to give more weight to KTP warna
    "grayscale_document": 0.85,
    "photocopy_on_colored_background": 0.70,
    "black": 0.00,
    "white": 0.00,
}
LEGIBILITY_MULTIPLIERS = {
    "excellent": 1.00,
    "good": 1.00,
    "fair": 0.90,
    "poor": 0.80,
}
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
        if 1.2 <= aspect <= 2.2:
            rectness = area / (w2*h2 + 1e-6)
            score = area * rectness
            if score > best_score:
                best, best_score = approx, score
    if best is None: return None
    return (best.reshape(-1,2) / scale).astype(np.float32)

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
    return dict(mean=float(vals.mean()),
                p90=float(np.percentile(vals, 90)),
                colored_frac=float((vals > COLOR_PIXEL_S_THR).mean()))

def classify_document(bgr, verbose=False):
    label = near_black_white_label(bgr)
    if label:
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

def _ocr_confidence(gray):
    if not _HAS_TESS:
        return None
    try:
        cfg = "--psm 6 --oem 3 -l ind+eng"
        d = pytesseract.image_to_data(gray, config=cfg, output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in d.get("conf", []) if c not in ("-1", -1)]
        if not confs: return 0.0
        return float(np.median(confs))
    except Exception:
        return None

def _order_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl,tr,br,bl], dtype=np.float32)

# ----- Helpers -----
def _find_censor_fraction(gray):
    h, w = gray.shape[:2]; area = h*w
    g = cv2.GaussianBlur(gray, (5,5), 0)
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 35, 5)
    inv = 255 - thr
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8), iterations=1)
    inv = cv2.morphologyEx(inv,cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    censor_area = 0
    for i in range(1, num):
        x,y,wc,hc,aa = stats[i]
        if aa < area * CENSOR_MIN_AREA_FRAC: continue
        patch = g[y:y+hc, x:x+wc]
        if patch.size == 0: continue
        if patch.std() <= CENSOR_UNIFORM_STD_THR:
            censor_area += aa
    return censor_area / max(1, area)

def _blockiness_score(gray):
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    mag = (mag / (mag.max() + 1e-6))
    h, w = gray.shape[:2]
    scores = []
    for B in [8,12,16,20,24]:
        if min(h, w) < 3*B: continue
        vlines = np.sum(mag[:, ::B])
        hlines = np.sum(mag[::B, :])
        vn = vlines / (h * (w//B + 1))
        hn = hlines / (w * (h//B + 1))
        scores.append((vn + hn) * 0.5)
    return float(np.clip(max(scores) if scores else 0.0, 0, 1))

def _mser_text_density(gray):
    h, w = gray.shape[:2]; mpix = (h*w)/1_000_000.0
    if mpix <= 0: return 0.0
    try:
        mser = cv2.MSER_create()
        if hasattr(mser, "setDelta"):    mser.setDelta(5)
        if hasattr(mser, "setMinArea"):  mser.setMinArea(30)
        if hasattr(mser, "setMaxArea"):  mser.setMaxArea(5000)
        regions, _ = mser.detectRegions(gray)
        return len(regions) / mpix
    except Exception:
        thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV,35,10)
        thr = cv2.morphologyEx(thr,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=1)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(thr,connectivity=4)
        count = sum(1 for i in range(1,num) if 20<=stats[i,cv2.CC_STAT_AREA]<=5000)
        return count / mpix

def _text_coverage(gray):
    thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV,35,10)
    thr = cv2.morphologyEx(thr,cv2.MORPH_OPEN,np.ones((2,2),np.uint8),iterations=1)
    thr = cv2.morphologyEx(thr,cv2.MORPH_CLOSE,np.ones((2,2),np.uint8),iterations=1)
    return float((thr>0).mean())

def _text_stroke_mask(gray):
    g = cv2.equalizeHist(gray)
    binv = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV,35,10)
    binv = cv2.morphologyEx(binv,cv2.MORPH_OPEN,np.ones((2,2),np.uint8),iterations=1)
    cnts,_ = cv2.findContours(binv,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    h,w = binv.shape[:2]; area=h*w
    for c in cnts:
        if cv2.contourArea(c)>area*0.01:
            cv2.drawContours(binv,[c],-1,0,-1)
    return binv

def _value_mask_from_ocr(gray):
    if not _HAS_TESS:
        return None, 0
    try:
        cfg = "--psm 6 --oem 3 -l ind+eng"
        data = pytesseract.image_to_data(gray, config=cfg, output_type=pytesseract.Output.DICT)
    except Exception:
        return None, 0

    n = len(data.get("text", []))
    if n == 0:
        return None, 0

    keys = ("block_num","par_num","line_num","left","top","width","height","text","conf")
    for k in keys:
        if k not in data: return None, 0

    H, W = gray.shape[:2]
    mask = np.zeros_like(gray, dtype=np.uint8)
    boxes_count = 0

    i = 0
    while i < n:
        b = data["block_num"][i]; p = data["par_num"][i]; ln = data["line_num"][i]
        idxs = []
        j = i
        while j < n and data["block_num"][j]==b and data["par_num"][j]==p and data["line_num"][j]==ln:
            idxs.append(j); j += 1
        colon_x = None
        for k in idxs:
            t = (data["text"][k] or "").strip()
            if ":" in t:
                colon_x = data["left"][k] + data["width"][k]//2
                break
        if colon_x is not None:
            for k in idxs:
                x = data["left"][k]; y = data["top"][k]; w = data["width"][k]; h = data["height"][k]
                if x > colon_x + 2:
                    pad = max(1, int(0.15*h))
                    x0 = max(0, x - pad); y0 = max(0, y - pad)
                    x1 = min(W-1, x + w + pad); y1 = min(H-1, y + h + pad)
                    cv2.rectangle(mask, (x0,y0), (x1,y1), 255, -1)
                    boxes_count += 1
        i = j

    if boxes_count == 0:
        return None, 0
    return mask, boxes_count

def _value_mask_by_projection(text_mask):
    col = (text_mask>0).astype(np.uint8).sum(axis=0).astype(np.float32)
    k = max(3, int(0.02*len(col)))
    col = cv2.blur(col.reshape(1,-1), (1,k)).ravel()
    col /= (col.max()+1e-6)
    W = len(col)
    cands = np.arange(int(0.35*W), int(0.65*W))
    if len(cands)==0: return None
    scores = []
    for s in cands:
        right = col[s:].mean()
        left  = col[:s].mean()
        scores.append(right - 0.5*left)
    s = cands[int(np.argmax(scores))]
    mask = np.zeros((1,W), np.uint8); mask[:, s:] = 255
    return cv2.resize(mask, (text_mask.shape[1], 1), interpolation=cv2.INTER_NEAREST).repeat(text_mask.shape[0], axis=0)

def _expand_value_mask(mask):
    if mask is None:
        return None
    k = max(5, int(mask.shape[1] * 0.02))  # ~2% of width
    kernel = np.ones((1, k), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)
# ------------------------------------------------
# ------------------------------------------------
def _grid_energy(gray, B_list=(8,12,16,20,24)):
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    mag = mag / (mag.max() + 1e-6)
    H, W = gray.shape[:2]
    best = 0.0
    for B in B_list:
        if min(H, W) < 3*B: 
            continue
        v = mag[:, ::B].mean()
        h = mag[::B, :].mean()
        best = max(best, 0.5*(v+h))
    return float(best)


def score_text_legibility(bgr, doc_poly, save_debug_dir=None, base_name=""):
    q = _order_quad(doc_poly)
    wA = np.linalg.norm(q[1]-q[0]); wB = np.linalg.norm(q[2]-q[3])
    hA = np.linalg.norm(q[3]-q[0]); hB = np.linalg.norm(q[2]-q[1])
    W = int(np.clip(max(wA,wB),800,1800))
    H = int(np.clip(max(hA,hB),500,1200))
    M = cv2.getPerspectiveTransform(q,np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],np.float32))
    top = cv2.warpPerspective(bgr,M,(W,H),flags=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(top,cv2.COLOR_BGR2GRAY)
    EXCLUDE_HEADER_FRAC = 0.15
    y0 = int(H*EXCLUDE_HEADER_FRAC)
    text_roi = gray[y0:,:int(W*0.78)]
    text_roi = cv2.fastNlMeansDenoising(text_roi,None,7,7,21)
    text_norm = cv2.equalizeHist(text_roi)
    text_mask = _text_stroke_mask(text_norm)

    # Build "value-only" mask with robust fallbacks
    value_mask_ocr, nboxes = _value_mask_from_ocr(text_norm)
    if value_mask_ocr is not None:
        value_mask_ocr = _expand_value_mask(value_mask_ocr)

    if value_mask_ocr is None:
        value_mask = _value_mask_by_projection(text_mask)
        used_ocr = False
    else:
        value_mask = value_mask_ocr
        used_ocr = True

    if value_mask is None:
        value_mask = np.ones_like(text_mask, dtype=np.uint8) * 255
    value_frac_tmp = float((cv2.bitwise_and(text_mask, value_mask)).mean() / 255.0)

    if used_ocr and value_frac_tmp < 0.05:
        proj_mask = _value_mask_by_projection(text_mask)
        if proj_mask is not None:
            value_mask = proj_mask
            value_frac_tmp = float((cv2.bitwise_and(text_mask, value_mask)).mean() / 255.0)

    if value_frac_tmp < 0.08:
        value_mask = np.ones_like(text_mask, dtype=np.uint8) * 255


    # --- Mosaic/blur penalty on value side (detect low local entropy and block grid) ---
    val = text_norm.copy()
    vm = (value_mask > 0)
    k = max(5, int(0.02*val.shape[1]))
    blur = cv2.blur(val.astype(np.float32), (k, k))
    sq = cv2.blur((val.astype(np.float32))**2, (k, k))
    std_map = np.sqrt(np.maximum(0, sq - blur**2))
    low_entropy = (std_map < 6.0) & vm
    mosaic_frac = float(low_entropy.mean())

    grid_e = _grid_energy(val)

    # Intersection: use only text strokes that are also on the value side
    score_mask = cv2.bitwise_and(text_mask, value_mask)
    text_frac  = float(text_mask.mean()/255.0)
    value_frac = float(score_mask.mean()/255.0)

    # Metrics computed on score_mask
    edges_img = cv2.Canny(text_norm,60,180)
    edges = float((edges_img>0)[score_mask>0].mean()) if np.any(score_mask>0) else 0.0
    sharp = _variance_of_laplacian(text_norm) * (value_frac + 1e-6)
    contr = float(np.std(text_norm[score_mask>0].astype(np.float32)/255.0)) if np.any(score_mask>0) else 0.0
    ocr_c = _ocr_confidence(text_norm)
    # If OCR confidence is very low, ignore OCR entirely (treat as None)
    try:
        OCR_IGNORE_BELOW
    except NameError:
        OCR_IGNORE_BELOW = 30.0
    if ocr_c is not None and ocr_c < OCR_IGNORE_BELOW:
        ocr_c = None
    coverage = _text_coverage(text_norm)

    s_sharp=_norm_0_100(sharp,*SHARP_LO_HI)
    s_edges=_norm_0_100(edges,*EDGE_LO_HI)
    s_contr=_norm_0_100(contr,*CONTR_LO_HI)
    parts=[s_sharp*W_SHARP,s_edges*W_EDGE,s_contr*W_CONTR]; denom=W_SHARP+W_EDGE+W_CONTR
    if ocr_c is not None: parts.append(ocr_c*W_OCR); denom+=W_OCR
    score=float(np.clip(sum(parts)/denom,0,100))

    # Structural penalties
    censor_frac=_find_censor_fraction(text_norm)
    blockiness=_blockiness_score(text_norm)
    mser_density=_mser_text_density(text_norm)

    # Heuristic lift: if multiple cues indicate readable fields (even with low sharpness)
    try:
        HEURISTIC_MIN_SCORE
    except NameError:
        HEURISTIC_MIN_SCORE = 60.0
    try:
        HEURISTIC_BONUS
    except NameError:
        HEURISTIC_BONUS = 10.0


    if mser_density<MSER_MIN_COUNT_PER_MP:
        miss_ratio=np.clip((MSER_MIN_COUNT_PER_MP-mser_density)/MSER_MIN_COUNT_PER_MP,0,1)
        score-=miss_ratio*MSER_PENALTY_W
    if mser_density>MSER_MAX_COUNT_PER_MP:
        over=(mser_density-MSER_MAX_COUNT_PER_MP)/MSER_MAX_COUNT_PER_MP
        score-=np.clip(over,0,1)*MSER_OVER_PENALTY_W
    score-=np.clip(censor_frac,0,1)*CENSOR_PENALTY_W
    score-=np.clip(blockiness,0,1)*BLOCKINESS_PENALTY_W
    if ocr_c is not None and ocr_c<OCR_LOW_CONF_THR:
        ocr_pen=(OCR_LOW_CONF_THR-ocr_c)/OCR_LOW_CONF_THR
        score-=np.clip(ocr_pen,0,1)*OCR_LOW_CONF_PENALTY_W
    # Mosaic penalty
    try:
        MOSAIC_PENALTY_W
    except NameError:
        MOSAIC_PENALTY_W = 18.0
    try:
        GRID_E_MIN
    except NameError:
        GRID_E_MIN = 0.12
    score -= np.clip(mosaic_frac*3.0, 0, 1) * MOSAIC_PENALTY_W
    if grid_e > GRID_E_MIN and mosaic_frac > 0.08:
        score -= (grid_e - GRID_E_MIN) * 60.0

        ocr_pen=(OCR_LOW_CONF_THR-ocr_c)/OCR_LOW_CONF_THR
        score-=np.clip(ocr_pen,0,1)*OCR_LOW_CONF_PENALTY_W
    cov_lo,_=TEXT_COVERAGE_RANGE
    if coverage<cov_lo:
        miss=(cov_lo-coverage)/max(cov_lo,1e-6)
        score-=np.clip(miss,0,1)*TEXT_COVERAGE_PENALTY_W

    
    # Apply lift if edges/contrast/coverage/MSER are in healthy ranges
    if score < HEURISTIC_MIN_SCORE:
        
        if (edges >= 0.20 and contr >= 0.16 and 0.18 <= coverage <= 0.42 and 800 <= mser_density <= 9000):
            score = min(100.0, max(score, HEURISTIC_MIN_SCORE - 5))
            score = min(100.0, score + HEURISTIC_BONUS * 0.5)

    
    # Heuristic floor for structurally readable KTPs
    try:
        STRUCT_FLOOR_SCORE
    except NameError:
        STRUCT_FLOOR_SCORE = 60.0  # target floor when structure OK
    if score < STRUCT_FLOOR_SCORE and (edges >= 0.20 and coverage >= 0.28 and mser_density >= 3000):
        score = max(score, STRUCT_FLOOR_SCORE)
    score=float(np.clip(score,0,100))
    label="excellent" if score>=85 else "good" if score>=70 else "fair" if score>=55 else "poor"

    if save_debug_dir:
        os.makedirs(save_debug_dir,exist_ok=True)
        dbg=os.path.splitext(os.path.basename(base_name))[0]
        cv2.imwrite(os.path.join(save_debug_dir,f"{dbg}_rectified.png"),top)
        cv2.imwrite(os.path.join(save_debug_dir,f"{dbg}_textmask.png"),text_mask)
        cv2.imwrite(os.path.join(save_debug_dir,f"{dbg}_valuemask.png"),value_mask)

    return dict(
        score=round(score,1),
        label=label,
        sharpness_vlap=round(sharp,1),
        edge_density=round(edges,4),
        rms_contrast=round(contr,4),
        ocr_conf=(round(ocr_c,1) if ocr_c is not None else None),
        warped_size=[W,H],
        censor_area_frac=round(float(censor_frac),3),
        blockiness=round(float(blockiness),3),
        mser_per_megapixel=round(float(mser_density),1),
        text_coverage=round(float(coverage),3),
        text_pixels_frac=round(float(text_frac),3),
        value_pixels_frac=round(float(value_frac),3)
    )

# -------------------- CLI + glue ----------------------

def analyze_file(path, verbose=False, want_json=True, save_debug=None):
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to read image")

    doc_label, poly, poly_in, doc_stats, bg_stats = classify_document(bgr, verbose=verbose)

    if poly_in is None:
        h, w = bgr.shape[:2]
        my, mx = int(h*0.12), int(w*0.10)
        poly_in = np.array([[mx,my],[w-mx,my],[w-mx,h-my],[mx,h-my]], dtype=np.float32)

    text = score_text_legibility(bgr, poly_in, save_debug, os.path.basename(path))

    # ----- Final Score (down-weights grayscale/photocopy + unclear text) -----
    doc_mult = DOC_MULTIPLIERS.get(doc_label, 1.0)
    leg_label = text["label"]
    leg_mult = LEGIBILITY_MULTIPLIERS.get(leg_label, 1.0)
    combined_mult = float(doc_mult * leg_mult)

    
    # Cap color/document-based bonus so color KTP doesn't inflate mediocre text too much
    try:
        FINAL_COLOR_BONUS_CAP
    except NameError:
        FINAL_COLOR_BONUS_CAP = 8.0  # max extra points from doc type
    raw_mult_score = float(np.clip(text["score"] * combined_mult, 0, 100))
    add_cap_score = float(min(text["score"] + FINAL_COLOR_BONUS_CAP, 100.0))
    final_score = min(raw_mult_score, add_cap_score)

    final_label = ("excellent" if final_score>=85 else
                   "good" if final_score>=70 else
                   "fair" if final_score>=55 else "poor")

    out = {
        "file": os.path.basename(path),
        "label": doc_label,
        "stats": {
            "doc_meanS": None if doc_stats is None else round(doc_stats["mean"],1),
            "doc_p90S":  None if doc_stats is None else round(doc_stats["p90"],1),
            "doc_colored_frac": None if doc_stats is None else round(doc_stats["colored_frac"],3),
            "bg_meanS":  None if bg_stats  is None else round(bg_stats["mean"],1)
        },
        "text_legibility": text,
        "final": {
            "score": round(final_score,1),
            "label": final_label,
            "multipliers": {
                "doc": round(float(doc_mult), 3),
                "text": round(float(leg_mult), 3),
                "combined": round(float(combined_mult), 3)
            },
            "reason": f"{doc_label}; text={leg_label}"
        },
        "version": __version__
    }

    if want_json:
        return json.dumps(out, ensure_ascii=False)
    else:
        lines = [f"{out['file']}: {out['label']}"]
        if verbose and doc_stats is not None:
            lines.append(f"  [doc] meanS={out['stats']['doc_meanS']} p90S={out['stats']['doc_p90S']} colored_frac={out['stats']['doc_colored_frac']}")
            lines.append(f"  [bg ] meanS={out['stats']['bg_meanS']}")
        t = out["text_legibility"]; f = out["final"]
        lines.append(
            f"  text_score={t['score']} ({t['label']}); FINAL={f['score']} ({f['label']}) "
            f"[x{f['multipliers']['doc']}*{f['multipliers']['text']}={f['multipliers']['combined']} due to {f['reason']}]"
        )
        return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="KTP validator v1.0 — color/photocopy detection + value-text legibility + Final Score (0..100)")
    ap.add_argument("paths", nargs="+", help="image file(s) or folder(s)")
    ap.add_argument("--plain", action="store_true", help="print human-readable instead of JSON")
    ap.add_argument("--verbose", action="store_true", help="print debug stats")
    ap.add_argument("--save-debug", default=None, help="directory to save rectified/text debug images")
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
            print(analyze_file(f, verbose=args.verbose, want_json=want_json, save_debug=args.save_debug))
        except Exception as e:
            err = {"file": os.path.basename(f), "error": str(e)}
            print(json.dumps(err) if want_json else f"{err['file']}: ERROR -> {err['error']}")

if __name__ == "__main__":
    main()
