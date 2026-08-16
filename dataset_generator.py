---

## 🐍 Item 2: `dataset_generator.py`

```python
import argparse
import csv
import os
import numpy as np
import cv2

def generate_dram_base(size=4000, grid_pitch=80, feature_radius=14):
    """Generates a continuous high-res DRAM memory array canvas."""
    layout = np.full((size, size), 40, dtype=np.float32)
    for pos in range(0, size, grid_pitch):
        layout[max(0, pos-2):min(size, pos+3), :] = 160
        layout[:, max(0, pos-2):min(size, pos+3)] = 160
    for y in range(0, size, grid_pitch):
        for x in range(0, size, grid_pitch):
            cv2.circle(layout, (x, y), feature_radius, (230), -1)
    return layout

def generate_finfet_base(size=4000, fin_pitch=30, gate_pitch=300):
    """Generates a continuous high-res FinFET logic structure canvas."""
    layout = np.full((size, size), 40, dtype=np.float32)
    for x in range(0, size, fin_pitch):
        layout[:, max(0, x-2):min(size, x+3)] = 170
    for y in range(0, size, gate_pitch):
        layout[max(0, y-10):min(size, y+11), :] = 220
    return layout

def apply_sem_noise(img, rot_deg=0.0, scale_var=1.0):
    """Applies SEM physics noise: edge brightening, surface charging, and detector noise."""
    work_img = img.astype(np.float32).copy()
    
    gx = cv2.Sobel(work_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(work_img, cv2.CV_32F, 0, 1, ksize=3)
    work_img += cv2.magnitude(gx, gy) * 0.35

    glow = cv2.GaussianBlur(work_img, (121, 121), 35)
    work_img = 0.82 * work_img + 0.18 * glow

    noise = np.random.normal(0, 14, work_img.shape).astype(np.float32)
    work_img = np.clip(work_img + noise, 0, 255).astype(np.uint8)

    if rot_deg != 0 or scale_var != 1.0:
        h, w = work_img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rot_deg, scale_var)
        work_img = cv2.warpAffine(work_img, M, (w, h), borderValue=40)

    return work_img

def convert_to_rgb_optical(gray_img):
    """Converts 1-channel layout to a 3-channel RGB optical capture."""
    h, w = gray_img.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_img[:, :] = [45, 35, 30]

    line_mask = gray_img >= 150
    rgb_img[line_mask] = [200, 180, 70]

    dot_mask = gray_img >= 220
    rgb_img[dot_mask] = [40, 140, 220]

    for c in range(3):
        noise = np.random.normal(0, 10, (h, w))
        rgb_img[:, :, c] = np.clip(rgb_img[:, :, c].astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return rgb_img

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic wafer image pairs.")
    parser.add_argument("--style", type=str, choices=["DRAM", "FinFET"], default="DRAM")
    parser.add_argument("--num_pairs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--is_rgb", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    gt_file_path = os.path.join(args.output_dir, "ground_truth.csv")

    with open(gt_file_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["pair_id", "ref_path", "search_path", "gt_center_x", "gt_center_y", "mode"])

        for i in range(1, args.num_pairs + 1):
            base_canvas = generate_dram_base(4000) if args.style.upper() == "DRAM" else generate_finfet_base(4000)

            ref_x = np.random.randint(200, 2800)
            ref_y = np.random.randint(200, 2800)

            reference_crop = base_canvas[ref_y:ref_y+1000, ref_x:ref_x+1000].copy()
            wide_search_raw = cv2.resize(base_canvas, (1000, 1000), interpolation=cv2.INTER_AREA)

            gt_center_x = float((ref_x + 500) / 4000.0 * 1000)
            gt_center_y = float((ref_y + 500) / 4000.0 * 1000)

            ref_img = apply_sem_noise(reference_crop, rot_deg=0.0)
            search_img = apply_sem_noise(wide_search_raw, rot_deg=np.random.uniform(-1.5, 1.5))

            mode_label = "SEM_GRAY"
            if args.is_rgb:
                ref_img = convert_to_rgb_optical(ref_img)
                search_img = convert_to_rgb_optical(search_img)
                mode_label = "OPTICAL_RGB"

            ref_full_path = os.path.join(args.output_dir, f"ref_{i:03d}.png")
            search_full_path = os.path.join(args.output_dir, f"search_{i:03d}.png")

            cv2.imwrite(ref_full_path, ref_img)
            cv2.imwrite(search_full_path, search_img)

            writer.writerow([i, ref_full_path, search_full_path, round(gt_center_x, 2), round(gt_center_y, 2), mode_label])
            print(f"Generated [{mode_label}] Pair #{i:03d} -> GT Center: ({gt_center_x:.2f}, {gt_center_y:.2f})")

if __name__ == "__main__":
    main()
