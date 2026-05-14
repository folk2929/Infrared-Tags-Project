import os
import re
import gc
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
import pandas as pd
import itertools
import time
import statistics

cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

IMAGE_SOURCE = r"C:\KMITL\Project ROG\testPhotos\Tag\Real photos90"
OUTPUT_DIR = r"C:\KMITL\Project ROG\testPhotos\Tag\Real photos90"

TARGET_TAG_ID = 0
WARMUP_RUNS = 5
MEASURE_RUNS = 10
TOP_N = 50

REUSE_EXISTING_BENCHMARK_CSV = False
EXISTING_BENCHMARK_CSV = "top_apriltag_pipelines_multi_distance.csv"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

PREP_METHODS = ["Gray", "CLAHE"]
BLUR_METHODS = ["None", "Median", "Gaussian", "Bilateral", "Avg"]
BLUR_KS = [3, 5, 7, 9, 11]
THRESH_METHODS = [
    "Otsu", "AdaptMean", "AdaptGauss",
    "Binary_80", "Binary_100", "Binary_127", "Binary_150", "Binary_180"
]
MORPH_METHODS = ["None", "Open", "Close", "Close_Open", "Open_Close", "Dilate", "Erode"]
MORPH_KS = [3, 5, 7, 9]

FONT = cv2.FONT_HERSHEY_DUPLEX
TEXT_COLOR = (0, 0, 0)
BORDER_COLOR = (0, 0, 0)
BG_COLOR = 245
PANEL_BG_COLOR = 255


def get_next_filename(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    counter = 1
    while True:
        candidate = parent / f"{stem}_no{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def safe_imwrite(path: Path, img: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        print(f"⚠️ บันทึกภาพไม่สำเร็จ: {path}")
    return ok


def create_clahe_image(gray_img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)


def load_image_paths(image_source: str) -> List[Path]:
    p = Path(image_source)

    if not p.exists():
        raise FileNotFoundError(f"ไม่พบ path นี้: {p}")

    if p.is_file():
        if p.suffix.lower() not in VALID_EXTS:
            raise ValueError(f"ไฟล์นี้ไม่ใช่ไฟล์รูปที่รองรับ: {p}")
        return [p]

    if p.is_dir():
        files = [x for x in sorted(p.iterdir()) if x.is_file() and x.suffix.lower() in VALID_EXTS]
        return files

    raise ValueError(f"ใช้ path นี้ไม่ได้: {p}")


def ensure_aruco_available() -> None:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV ตัวนี้ไม่มี cv2.aruco\n"
            "ให้ติดตั้ง opencv-contrib-python เช่น:\n"
            "pip install opencv-contrib-python"
        )


def create_detector_params():
    ensure_aruco_available()

    if hasattr(cv2.aruco, "DetectorParameters_create"):
        params = cv2.aruco.DetectorParameters_create()
    else:
        params = cv2.aruco.DetectorParameters()

    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.polygonalApproxAccuracyRate = 0.05
    return params


def create_detector(aruco_dict, aruco_params):
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    return None


def detect_markers(img, aruco_dict, aruco_params, detector):
    if detector is not None:
        return detector.detectMarkers(img)
    return cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)


def parse_name_and_k(text: str, label: str) -> Tuple[str, int]:
    text = str(text).strip()
    m = re.match(r"^(.*?)\((\d+)\)$", text)
    if not m:
        raise ValueError(f"รูปแบบ {label} ไม่ถูกต้อง: {text}")
    name = m.group(1).strip()
    k = int(m.group(2))
    return name, k


def parse_blur(text: str) -> Tuple[str, int]:
    return parse_name_and_k(text, "Blur")


def parse_morph(text: str) -> Tuple[str, int]:
    return parse_name_and_k(text, "Morphology")


def pipeline_key(prep: str, blur_name: str, blur_k: int, thresh_name: str, morph_name: str, morph_k: int) -> str:
    return f"{prep}|{blur_name}({blur_k})|{thresh_name}|{morph_name}({morph_k})"


def normalize_kernel(k: int) -> int:
    k = int(k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    return k


def build_pipeline_image(
    img_gray: np.ndarray,
    img_clahe: np.ndarray,
    prep: str,
    blur_name: str,
    blur_k: int,
    thresh_name: str,
    morph_name: str,
    morph_k: int,
) -> np.ndarray:
    img_current = img_clahe.copy() if prep == "CLAHE" else img_gray.copy()

    blur_k = normalize_kernel(blur_k)
    morph_k = max(1, int(morph_k))

    if blur_name == "Median":
        img_b = cv2.medianBlur(img_current, blur_k)
    elif blur_name == "Gaussian":
        img_b = cv2.GaussianBlur(img_current, (blur_k, blur_k), 0)
    elif blur_name == "Bilateral":
        img_b = cv2.bilateralFilter(img_current, blur_k, 75, 75)
    elif blur_name == "Avg":
        img_b = cv2.blur(img_current, (blur_k, blur_k))
    elif blur_name == "None":
        img_b = img_current.copy()
    else:
        raise ValueError(f"Unknown blur: {blur_name}")

    if thresh_name == "Otsu":
        _, img_t = cv2.threshold(img_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresh_name == "AdaptMean":
        img_t = cv2.adaptiveThreshold(img_b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
    elif thresh_name == "AdaptGauss":
        img_t = cv2.adaptiveThreshold(img_b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    elif thresh_name.startswith("Binary_"):
        val = int(thresh_name.split("_")[1])
        _, img_t = cv2.threshold(img_b, val, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown threshold: {thresh_name}")

    kernel = np.ones((morph_k, morph_k), np.uint8)

    if morph_name == "Open":
        img_final = cv2.morphologyEx(img_t, cv2.MORPH_OPEN, kernel)
    elif morph_name == "Close":
        img_final = cv2.morphologyEx(img_t, cv2.MORPH_CLOSE, kernel)
    elif morph_name == "Close_Open":
        img_final = cv2.morphologyEx(
            cv2.morphologyEx(img_t, cv2.MORPH_CLOSE, kernel),
            cv2.MORPH_OPEN,
            kernel,
        )
    elif morph_name == "Open_Close":
        img_final = cv2.morphologyEx(
            cv2.morphologyEx(img_t, cv2.MORPH_OPEN, kernel),
            cv2.MORPH_CLOSE,
            kernel,
        )
    elif morph_name == "Dilate":
        img_final = cv2.dilate(img_t, kernel, iterations=1)
    elif morph_name == "Erode":
        img_final = cv2.erode(img_t, kernel, iterations=1)
    elif morph_name == "None":
        img_final = img_t.copy()
    else:
        raise ValueError(f"Unknown morphology: {morph_name}")

    return img_final


def fit_to_cell(img: np.ndarray, cell_w: int, cell_h: int, pad: int = 10) -> np.ndarray:
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    avail_w = max(1, cell_w - 2 * pad)
    avail_h = max(1, cell_h - 2 * pad)

    h, w = img.shape[:2]
    scale = min(avail_w / w, avail_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((cell_h, cell_w, 3), PANEL_BG_COLOR, dtype=np.uint8)
    x = (cell_w - new_w) // 2
    y = (cell_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def make_rank_image(
    img_gray: np.ndarray,
    rank: int,
    pipeline_text: Optional[str] = None,
    detect_text: Optional[str] = None,
    canvas_w: int = 1600,
    canvas_h: int = 900,
    left_panel_w: int = 360,
    margin: int = 30,
) -> np.ndarray:
    if len(img_gray.shape) == 2:
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img_gray.copy()

    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)

    left_x1 = margin
    left_y1 = margin
    left_x2 = left_panel_w
    left_y2 = canvas_h - margin

    cv2.rectangle(canvas, (left_x1, left_y1), (left_x2, left_y2), (PANEL_BG_COLOR, PANEL_BG_COLOR, PANEL_BG_COLOR), -1)
    cv2.rectangle(canvas, (left_x1, left_y1), (left_x2, left_y2), BORDER_COLOR, 1)

    y = left_y1 + 55
    cv2.putText(canvas, f"Rank #{rank}", (left_x1 + 20, y), FONT, 1.0, TEXT_COLOR, 2, cv2.LINE_AA)
    y += 50

    info_lines = []
    if pipeline_text:
        info_lines.extend(pipeline_text.split("\n"))
    if detect_text:
        info_lines.append("")
        info_lines.extend(detect_text.split("\n"))

    for line in info_lines:
        if line == "":
            y += 18
            continue
        cv2.putText(canvas, line[:34], (left_x1 + 20, y), FONT, 0.65, TEXT_COLOR, 1, cv2.LINE_AA)
        y += 30

    img_x1 = left_panel_w + margin
    img_y1 = margin
    img_x2 = canvas_w - margin
    img_y2 = canvas_h - margin

    cv2.rectangle(canvas, (img_x1, img_y1), (img_x2, img_y2), (PANEL_BG_COLOR, PANEL_BG_COLOR, PANEL_BG_COLOR), -1)
    cv2.rectangle(canvas, (img_x1, img_y1), (img_x2, img_y2), BORDER_COLOR, 1)

    avail_w = (img_x2 - img_x1) - 40
    avail_h = (img_y2 - img_y1) - 40

    h, w = img_bgr.shape[:2]
    scale = min(avail_w / w, avail_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    paste_x = img_x1 + ((img_x2 - img_x1) - new_w) // 2
    paste_y = img_y1 + ((img_y2 - img_y1) - new_h) // 2

    canvas[paste_y:paste_y + new_h, paste_x:paste_x + new_w] = resized
    return canvas


def make_contact_sheet(rank_images, sheet_title: str = "ALL TOP 50", cols: int = 5, cell_w: int = 420, cell_h: int = 320, margin: int = 20, title_h: int = 80) -> np.ndarray:
    n = len(rank_images)
    rows = int(np.ceil(n / cols)) if n > 0 else 1

    sheet_w = margin + cols * (cell_w + margin)
    sheet_h = title_h + margin + rows * (cell_h + margin)

    sheet = np.full((sheet_h, sheet_w, 3), BG_COLOR, dtype=np.uint8)

    cv2.putText(sheet, sheet_title, (margin, 50), FONT, 1.0, TEXT_COLOR, 2, cv2.LINE_AA)

    for i, (rank, img) in enumerate(rank_images):
        row = i // cols
        col = i % cols

        x = margin + col * (cell_w + margin)
        y = title_h + row * (cell_h + margin)

        cell = fit_to_cell(img, cell_w, cell_h, pad=10)
        sheet[y:y + cell_h, x:x + cell_w] = cell
        cv2.rectangle(sheet, (x, y), (x + cell_w, y + cell_h), BORDER_COLOR, 1)
        cv2.putText(sheet, f"#{rank}", (x + 10, y + 30), FONT, 0.9, TEXT_COLOR, 2, cv2.LINE_AA)

    return sheet


def read_pipeline_fields(row: pd.Series) -> Tuple[str, str, int, str, str, int]:
    prep = str(row["Prep"])
    thresh_name = str(row["Threshold"])

    if "Blur Name" in row.index and "Blur K" in row.index:
        blur_name = str(row["Blur Name"])
        blur_k = int(row["Blur K"])
    else:
        blur_name, blur_k = parse_blur(row["Blur"])

    if "Morph Name" in row.index and "Morph K" in row.index:
        morph_name = str(row["Morph Name"])
        morph_k = int(row["Morph K"])
    else:
        morph_name, morph_k = parse_morph(row["Morphology"])

    return prep, blur_name, blur_k, thresh_name, morph_name, morph_k


def summarize_top_console(df: pd.DataFrame, limit: int = 10) -> None:
    if df.empty:
        print("⚠️ ไม่มีข้อมูลให้แสดง")
        return

    show_df = df.head(limit).copy()
    for col in ["Images Hit Ratio", "Target Hit Ratio", "Median Total (ms)", "Mean Total (ms)", "Std Total (ms)"]:
        if col in show_df.columns:
            show_df[col] = show_df[col].round(4)

    print("\n🏆 TOP MULTI-DISTANCE PIPELINES")
    print(show_df)


def benchmark_all_images(image_paths: List[Path]) -> pd.DataFrame:
    ensure_aruco_available()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    aruco_params = create_detector_params()
    detector = create_detector(aruco_dict, aruco_params)

    all_combinations = list(itertools.product(
        PREP_METHODS, BLUR_METHODS, BLUR_KS,
        THRESH_METHODS, MORPH_METHODS, MORPH_KS
    ))

    image_cache = []
    for image_path in image_paths:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print(f"❌ เปิดรูปไม่ได้: {image_path}")
            continue

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_clahe = create_clahe_image(img_gray)
        image_cache.append((image_path, img_gray, img_clahe))

    if not image_cache:
        raise RuntimeError("ไม่มีรูปที่เปิดได้เลย")

    results = []

    print("🚀 เริ่ม benchmark หลายระยะ")
    print(f"📦 จำนวนรูป: {len(image_cache)}")
    print(f"📦 จำนวน pipeline ดิบ: {len(all_combinations):,}")

    gc.disable()
    try:
        for idx, (prep, b, bk, t, m, mk) in enumerate(all_combinations):
            if (idx + 1) % 500 == 0:
                print(f"กำลังประมวลผล pipeline... [{idx + 1:,}/{len(all_combinations):,}]")

            if b == "None" and bk != 3:
                continue
            if m == "None" and mk != 3:
                continue

            total_times = []
            hit_any_images = 0
            hit_target_images = 0

            for image_path, img_gray, img_clahe in image_cache:
                for _ in range(WARMUP_RUNS):
                    img_final = build_pipeline_image(img_gray, img_clahe, prep, b, bk, t, m, mk)
                    detect_markers(img_final, aruco_dict, aruco_params, detector)

                run_times = []
                target_hit_this_image = False
                any_hit_this_image = False

                for _ in range(MEASURE_RUNS):
                    t0 = time.perf_counter_ns()
                    img_final = build_pipeline_image(img_gray, img_clahe, prep, b, bk, t, m, mk)
                    corners, ids, _ = detect_markers(img_final, aruco_dict, aruco_params, detector)
                    t1 = time.perf_counter_ns()

                    run_times.append((t1 - t0) / 1_000_000.0)

                    if ids is not None and len(ids) > 0:
                        any_hit_this_image = True
                        ids_list = ids.flatten().tolist()
                        if TARGET_TAG_ID is None or TARGET_TAG_ID in ids_list:
                            target_hit_this_image = True

                total_times.extend(run_times)

                if any_hit_this_image:
                    hit_any_images += 1
                if target_hit_this_image:
                    hit_target_images += 1

            results.append({
                "Pipeline Key": pipeline_key(prep, b, bk, t, m, mk),
                "Prep": prep,
                "Blur": f"{b}({bk})",
                "Blur Name": b,
                "Blur K": bk,
                "Threshold": t,
                "Morphology": f"{m}({mk})",
                "Morph Name": m,
                "Morph K": mk,
                "Images Hit Count": hit_any_images,
                "Target Hit Count": hit_target_images,
                "Images Hit Ratio": round(hit_any_images / len(image_cache), 4),
                "Target Hit Ratio": round(hit_target_images / len(image_cache), 4),
                "Median Total (ms)": statistics.median(total_times) if total_times else 999999.0,
                "Mean Total (ms)": statistics.mean(total_times) if total_times else 999999.0,
                "Std Total (ms)": statistics.pstdev(total_times) if len(total_times) > 1 else 0.0,
            })
    finally:
        gc.enable()

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(
        by=[
            "Target Hit Count",
            "Images Hit Count",
            "Target Hit Ratio",
            "Images Hit Ratio",
            "Median Total (ms)",
            "Std Total (ms)",
            "Prep",
            "Blur",
            "Threshold",
            "Morphology",
        ],
        ascending=[False, False, False, False, True, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)

    return df_sorted


def apply_top_pipelines_to_all_images(image_paths: List[Path], top_df: pd.DataFrame):
    ensure_aruco_available()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    aruco_params = create_detector_params()
    detector = create_detector(aruco_dict, aruco_params)

    summary_rows = []
    image_ranking_rows = []

    for image_path in image_paths:
        print(f"\nกำลังสร้างผลลัพธ์สวยงาม: {image_path.name}")

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print(f"❌ เปิดรูปไม่ได้: {image_path}")
            continue

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_clahe = create_clahe_image(img_gray)

        image_output_dir = Path(OUTPUT_DIR) / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        rank_images = []
        detected_count = 0
        detected_target_count = 0
        detected_rank_list = []

        for idx, row in top_df.iterrows():
            rank = idx + 1

            prep, blur_name, blur_k, thresh_name, morph_name, morph_k = read_pipeline_fields(row)

            img_final = build_pipeline_image(
                img_gray,
                img_clahe,
                prep,
                blur_name,
                blur_k,
                thresh_name,
                morph_name,
                morph_k,
            )

            _, ids, _ = detect_markers(img_final, aruco_dict, aruco_params, detector)

            detected = ids is not None and len(ids) > 0
            detected_ids_text = ""
            target_hit = False

            if detected:
                ids_list = ids.flatten().tolist()
                detected_ids_text = ",".join(map(str, ids_list))
                detected_count += 1
                detected_rank_list.append(rank)

                if TARGET_TAG_ID is None or TARGET_TAG_ID in ids_list:
                    detected_target_count += 1
                    target_hit = True

            pipeline_text = (
                f"{prep}\n"
                f"{blur_name}({blur_k})\n"
                f"{thresh_name}\n"
                f"{morph_name}({morph_k})"
            )
            detect_text = (
                f"Detected: {'Yes' if detected else 'No'}\n"
                f"Target: {'Yes' if target_hit else 'No'}\n"
                f"IDs: {detected_ids_text if detected_ids_text else '-'}"
            )

            rank_canvas = make_rank_image(
                img_final,
                rank,
                pipeline_text=pipeline_text,
                detect_text=detect_text,
            )

            out_name = f"{image_path.stem}_rank{rank:02d}.png"
            out_path = image_output_dir / out_name
            safe_imwrite(out_path, rank_canvas)

            rank_images.append((rank, img_final.copy()))

            summary_rows.append({
                "Image": image_path.name,
                "Rank": rank,
                "Prep": prep,
                "Blur": f"{blur_name}({blur_k})",
                "Blur Name": blur_name,
                "Blur K": blur_k,
                "Threshold": thresh_name,
                "Morphology": f"{morph_name}({morph_k})",
                "Morph Name": morph_name,
                "Morph K": morph_k,
                "Detected": detected,
                "Target Hit": target_hit,
                "Detected IDs": detected_ids_text,
                "Output File": str(out_path),
            })

        all_sheet = make_contact_sheet(
            rank_images,
            sheet_title=f"{image_path.stem} - ALL TOP {len(top_df)}",
            cols=5,
            cell_w=420,
            cell_h=320,
            margin=20,
            title_h=80,
        )

        all_sheet_path = image_output_dir / f"{image_path.stem}_ALL_TOP{len(top_df)}.png"
        safe_imwrite(all_sheet_path, all_sheet)

        image_ranking_rows.append({
            "Image": image_path.name,
            "Detected Count": detected_count,
            "Detected Target Count": detected_target_count,
            "Detected Ratio": round(detected_count / len(top_df), 4) if len(top_df) else 0.0,
            "Detected Target Ratio": round(detected_target_count / len(top_df), 4) if len(top_df) else 0.0,
            "Detected Ranks": ",".join(map(str, detected_rank_list)),
            "ALL Sheet": str(all_sheet_path),
        })

        print(f"✅ {image_path.name}: ติดรวม {detected_count}/{len(top_df)} | target {detected_target_count}/{len(top_df)}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = get_next_filename(Path(OUTPUT_DIR) / "top50_output_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    ranking_df = pd.DataFrame(image_ranking_rows)
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values(
            by=["Detected Target Count", "Detected Count", "Image"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)

    ranking_csv = get_next_filename(Path(OUTPUT_DIR) / "image_detect_ranking.csv")
    ranking_df.to_csv(ranking_csv, index=False, encoding="utf-8-sig")

    return summary_csv, ranking_csv


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = load_image_paths(IMAGE_SOURCE)
    if not image_paths:
        print("❌ ไม่พบรูปในโฟลเดอร์")
        return

    print(f"📁 ใช้รูปทั้งหมด {len(image_paths)} ไฟล์")

    benchmark_csv_path = output_dir / EXISTING_BENCHMARK_CSV

    if REUSE_EXISTING_BENCHMARK_CSV and benchmark_csv_path.exists():
        print(f"📄 ใช้ benchmark เดิมจากไฟล์: {benchmark_csv_path}")
        benchmark_df = pd.read_csv(benchmark_csv_path)
    else:
        benchmark_df = benchmark_all_images(image_paths)
        benchmark_csv_path = get_next_filename(benchmark_csv_path)
        benchmark_df.to_csv(benchmark_csv_path, index=False, encoding="utf-8-sig")
        print(f"💾 บันทึก benchmark ไว้ที่: {benchmark_csv_path}")

    if benchmark_df.empty:
        print("❌ ไม่มีผล benchmark")
        return

    top_df = benchmark_df.head(TOP_N).copy()
    summarize_top_console(top_df, limit=10)

    summary_csv, ranking_csv = apply_top_pipelines_to_all_images(image_paths, top_df)

    print("\nเสร็จแล้วทั้งหมด")
    print(f"CSV pipeline ที่คัดจากหลายระยะ: {benchmark_csv_path}")
    print(f"CSV สรุปราย rank: {summary_csv}")
    print(f"CSV จัดอันดับรูปที่สแกนติดมากสุด: {ranking_csv}")
    print(f"โฟลเดอร์ผลลัพธ์: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ โปรแกรมหยุดเพราะ error: {e}")
        raise
