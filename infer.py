import sys
import os
import argparse
import cv2
import numpy as np

def extract_dl_features(img, net):
    """Passes image through ONNX Deep Feature Extractor network if present."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blob = cv2.dnn.blobFromImage(gray.astype(np.float32), 1.0 / 255.0, (gray.shape[1], gray.shape[0]))
    net.setInput(blob)
    feature_map = net.forward()
    out = np.squeeze(feature_map)
    return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def compute_correlation_map(search_img, template):
    """Computes normalized cross-correlation for 1-channel or 3-channel images."""
    if len(search_img.shape) == 2:
        return cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
    else:
        corr_b = cv2.matchTemplate(search_img[:, :, 0], template[:, :, 0], cv2.TM_CCOEFF_NORMED)
        corr_g = cv2.matchTemplate(search_img[:, :, 1], template[:, :, 1], cv2.TM_CCOEFF_NORMED)
        corr_r = cv2.matchTemplate(search_img[:, :, 2], template[:, :, 2], cv2.TM_CCOEFF_NORMED)
        return (corr_b + corr_g + corr_r) / 3.0

def locate_pattern(ref_img_path, search_img_path, weights_path="model.onnx"):
    ref_img = cv2.imread(ref_img_path, cv2.IMREAD_UNCHANGED)
    search_img = cv2.imread(search_img_path, cv2.IMREAD_UNCHANGED)

    if ref_img is None or search_img is None:
        raise FileNotFoundError(f"Could not load input images: '{ref_img_path}' or '{search_img_path}'")

    if os.path.exists(weights_path):
        try:
            net = cv2.dnn.readNetFromONNX(weights_path)
            ref_proc = extract_dl_features(ref_img, net)
            search_proc = extract_dl_features(search_img, net)
        except Exception:
            ref_proc, search_proc = ref_img, search_img
    else:
        ref_proc, search_proc = ref_img, search_img

    h_ref, w_ref = ref_proc.shape[:2]
    search_center = np.array([search_proc.shape[1] / 2.0, search_proc.shape[0] / 2.0])

    best_val = -1.0
    best_candidate_center = (500, 500)
    scale_range = np.linspace(0.08, 0.12, 5)

    for scale in scale_range:
        tw, th = int(w_ref * scale), int(h_ref * scale)
        if tw <= 0 or th <= 0:
            continue
        
        template = cv2.resize(ref_proc, (tw, th), interpolation=cv2.INTER_AREA)
        res = compute_correlation_map(search_proc, template)
        
        threshold = 0.45
        loc = np.where(res >= threshold)
        candidates = list(zip(*loc[::-1]))

        if candidates:
            for pt in candidates:
                val = res[pt[1], pt[0]]
                cx = pt[0] + tw / 2.0
                cy = pt[1] + th / 2.0
                dist_to_center = np.linalg.norm(np.array([cx, cy]) - search_center)
                score = val - (dist_to_center * 0.0001)

                if score > best_val:
                    best_val = score
                    best_candidate_center = (int(round(cx)), int(round(cy)))
        else:
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            cx = max_loc[0] + tw / 2.0
            cy = max_loc[1] + th / 2.0
            score = max_val
            if score > best_val:
                best_val = score
                best_candidate_center = (int(round(cx)), int(round(cy)))

    return best_candidate_center

def main():
    parser = argparse.ArgumentParser(description="SEM/RGB Target Localization Script.")
    parser.add_argument("ref_pos", nargs="?", default=None)
    parser.add_argument("search_pos", nargs="?", default=None)
    parser.add_argument("--ref", type=str, default=None)
    parser.add_argument("--search", type=str, default=None)
    parser.add_argument("--weights", type=str, default="model.onnx")
    
    args = parser.parse_args()
    ref_path = args.ref_pos or args.ref
    search_path = args.search_pos or args.search

    if not ref_path or not search_path:
        print("Usage: python infer.py <path_to_ref_image> <path_to_search_image>")
        sys.exit(1)

    predicted_x, predicted_y = locate_pattern(ref_path, search_path, weights_path=args.weights)
    print(f"{predicted_x}, {predicted_y}")

if __name__ == "__main__":
    main()
