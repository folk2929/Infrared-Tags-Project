Vscode
import time
import cv2
import numpy as np
from collections import deque
from picamera2 import Picamera2

TARGET_RES = (1280, 720)

MANUAL_FOCUS = True
LENS_POS = 6.5

TARGET_ID = 0

TEST_FRAMES = 1005
TEST_ROUNDS = 3
WARMUP_FRAMES = 5

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
BINARY_THRESH = 100
ERODE_KERNEL_SIZE = 7

WARP_SIZE = 240
INNER_CROP_RATIO = 0.0

MIN_NOISE_BLOB_AREA = 1
MAX_NOISE_BLOB_AREA = 999999
MIN_CODE_BLOB_AREA = 500
CODE_BORDER_MARGIN = 12
CODE_DILATE_K = 2

MIN_TAG_BBOX_AREA = 8000
MIN_TAG_SIDE = 60
MAX_ASPECT_DEV = 0.45

STABLE_HIT_WINDOW = 5
STABLE_HIT_MIN = 4

ROI_MODE_AUTO = "AUTO"
ROI_MODE_MANUAL = "MANUAL"

roi_mode = ROI_MODE_AUTO

manual_roi_corners = None
manual_roi_points = []
manual_collecting = False
active_measure_source = "AUTO-LIVE"

RAW_MANUAL_WINDOW = "Raw Manual ROI"


def create_clahe_image(gray_img):
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )
    return clahe.apply(gray_img)


def build_processed_image(gray_img):
    clahe_img = create_clahe_image(gray_img)

    _, binary_img = cv2.threshold(
        clahe_img,
        BINARY_THRESH,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((ERODE_KERNEL_SIZE, ERODE_KERNEL_SIZE), np.uint8)
    processed_img = cv2.erode(binary_img, kernel, iterations=1)

    return processed_img


def clamp_roi(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return x1, y1, x2, y2


def map_corners_back_to_original(corners, rot_name, orig_w, orig_h):
    mapped_corners = []

    for c in corners:
        pts = c.reshape((4, 2)).astype(np.float32)
        mapped = np.zeros_like(pts)

        for i, (x, y) in enumerate(pts):
            if rot_name == "0":
                ox, oy = x, y
            elif rot_name == "90":
                ox = y
                oy = orig_h - 1 - x
            elif rot_name == "180":
                ox = orig_w - 1 - x
                oy = orig_h - 1 - y
            elif rot_name == "270":
                ox = orig_w - 1 - y
                oy = x
            else:
                ox, oy = x, y

            mapped[i] = [ox, oy]

        mapped_corners.append(mapped.reshape((1, 4, 2)))

    return mapped_corners


def order_points_visual(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape((4, 2))

    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]

    top_left, top_right = top_two[np.argsort(top_two[:, 0])]
    bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

    return np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)


def warp_from_corners(img, corners, warp_size=WARP_SIZE):
    if corners is None or len(corners) == 0:
        return None, None, None

    raw_pts = corners[0].reshape((4, 2)).astype(np.float32)
    pts = order_points_visual(raw_pts)

    dst = np.array([
        [0, 0],
        [warp_size - 1, 0],
        [warp_size - 1, warp_size - 1],
        [0, warp_size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (warp_size, warp_size))

    return warped, M, pts


def validate_tag_geometry(corners, img_shape):
    if corners is None or len(corners) == 0:
        return False, "no corners"

    pts = corners[0].reshape((4, 2)).astype(np.float32)

    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))

    bw = x2 - x1
    bh = y2 - y1
    bbox_area = bw * bh

    if bw < MIN_TAG_SIDE or bh < MIN_TAG_SIDE:
        return False, "bbox too small"

    if bbox_area < MIN_TAG_BBOX_AREA:
        return False, "area too small"

    aspect = bw / bh if bh > 0 else 999.0
    if abs(aspect - 1.0) > MAX_ASPECT_DEV:
        return False, "aspect off"

    h, w = img_shape[:2]
    if x1 < 1 or y1 < 1 or x2 > (w - 2) or y2 > (h - 2):
        return False, "touch border"

    return True, "ok"


def build_code_mask_from_detect_inner(detect_inner_bin):
    empty = {
        "code_mask": None,
        "code_mask_dilated": None,
        "component_view": None,
        "code_pixels": 0,
        "code_blob_count": 0,
        "found": False
    }

    if detect_inner_bin is None or detect_inner_bin.size == 0:
        return empty

    _, detect_inner_bin = cv2.threshold(
        detect_inner_bin,
        127,
        255,
        cv2.THRESH_BINARY
    )

    h, w = detect_inner_bin.shape[:2]

    num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
        detect_inner_bin,
        connectivity=8
    )

    code_mask = np.zeros_like(detect_inner_bin)
    component_view = np.zeros_like(detect_inner_bin)

    code_pixels = 0
    code_blob_count = 0

    MIN_CODE_AREA_REAL = MIN_CODE_BLOB_AREA
    BORDER_MARGIN = CODE_BORDER_MARGIN
    DILATE_K = CODE_DILATE_K

    for label_id in range(1, num_labels):
        x = int(stats_cc[label_id, cv2.CC_STAT_LEFT])
        y = int(stats_cc[label_id, cv2.CC_STAT_TOP])
        bw = int(stats_cc[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats_cc[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats_cc[label_id, cv2.CC_STAT_AREA])

        x2 = x + bw
        y2 = y + bh

        component_mask = (labels == label_id)

        if area < MIN_CODE_AREA_REAL:
            continue

        touches_border = (
            x <= BORDER_MARGIN or
            y <= BORDER_MARGIN or
            x2 >= (w - BORDER_MARGIN) or
            y2 >= (h - BORDER_MARGIN)
        )
        if touches_border:
            continue

        code_mask[component_mask] = 255
        component_view[component_mask] = 255
        code_pixels += area
        code_blob_count += 1

    kernel = np.ones((DILATE_K, DILATE_K), np.uint8)
    code_mask_dilated = cv2.dilate(code_mask, kernel, iterations=1)

    return {
        "code_mask": code_mask,
        "code_mask_dilated": code_mask_dilated,
        "component_view": component_view,
        "code_pixels": code_pixels,
        "code_blob_count": code_blob_count,
        "found": code_pixels > 0
    }


def paste_inner_to_warp(mask_inner, crop_box, warp_size):
    full = np.zeros((warp_size, warp_size), dtype=np.uint8)

    if mask_inner is None or crop_box is None:
        return full

    x1, y1, x2, y2 = crop_box
    h = y2 - y1
    w = x2 - x1

    if mask_inner.shape[0] != h or mask_inner.shape[1] != w:
        mask_inner = cv2.resize(mask_inner, (w, h), interpolation=cv2.INTER_NEAREST)

    full[y1:y2, x1:x2] = mask_inner
    return full


def project_mask_to_original(mask_inner, crop_box, M, output_shape, warp_size=WARP_SIZE):
    if mask_inner is None or crop_box is None or M is None:
        return None

    warped_full = paste_inner_to_warp(mask_inner, crop_box, warp_size)

    Minv = np.linalg.inv(M)
    projected = cv2.warpPerspective(
        warped_full,
        Minv,
        (output_shape[1], output_shape[0]),
        flags=cv2.INTER_NEAREST
    )

    _, projected = cv2.threshold(projected, 127, 255, cv2.THRESH_BINARY)
    return projected


def measure_noise_in_warped_tag(processed_img, corners):
    empty_result = {
        "found": False,
        "code_white_ratio": 0.0,
        "noise_ratio": 0.0,
        "total_white_ratio": 0.0,
        "code_pixels": 0,
        "noise_pixels": 0,
        "total_white_pixels": 0,
        "total_pixels": 0,
        "background_pixels": 0,
        "noise_blob_count": 0,
        "roi_box": None,
        "warped_detect": None,
        "warped_noise": None,
        "inner_detect": None,
        "inner_noise": None,
        "code_mask": None,
        "noise_mask": None,
        "background_mask": None,
        "component_view": None,
        "M": None,
        "inner_crop_box": None,
        "noise_mask_projected": None,
        "code_mask_projected": None,
        "background_mask_projected": None
    }

    if corners is None or len(corners) == 0:
        return empty_result

    warped_detect, M, raw_pts = warp_from_corners(processed_img, corners, WARP_SIZE)
    warped_noise, _, _ = warp_from_corners(processed_img, corners, WARP_SIZE)

    if warped_detect is None or warped_noise is None or M is None:
        return empty_result

    inner_detect_bin = warped_detect.copy()
    inner_noise_bin = warped_noise.copy()
    inner_crop_box = (0, 0, WARP_SIZE, WARP_SIZE)

    _, inner_detect_bin = cv2.threshold(inner_detect_bin, 127, 255, cv2.THRESH_BINARY)
    _, inner_noise_bin = cv2.threshold(inner_noise_bin, 127, 255, cv2.THRESH_BINARY)

    total_pixels = inner_detect_bin.shape[0] * inner_detect_bin.shape[1]
    if total_pixels <= 0:
        return empty_result

    code_result = build_code_mask_from_detect_inner(inner_detect_bin)
    code_mask = code_result["code_mask"]
    code_mask_dilated = code_result["code_mask_dilated"]

    if code_mask is None or code_mask_dilated is None:
        return empty_result

    valid_mask = np.ones_like(inner_noise_bin, dtype=np.uint8) * 255
    background_mask = cv2.bitwise_and(valid_mask, cv2.bitwise_not(code_mask_dilated))
    noise_candidate = cv2.bitwise_and(inner_noise_bin, background_mask)

    num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
        noise_candidate,
        connectivity=8
    )

    noise_mask = np.zeros_like(inner_noise_bin)
    noise_pixels = 0
    noise_blob_count = 0

    for label_id in range(1, num_labels):
        area = int(stats_cc[label_id, cv2.CC_STAT_AREA])
        if MIN_NOISE_BLOB_AREA <= area <= MAX_NOISE_BLOB_AREA:
            component_mask = (labels == label_id)
            noise_mask[component_mask] = 255
            noise_pixels += area
            noise_blob_count += 1

    code_pixels = int(np.count_nonzero(code_mask))
    background_pixels = int(np.count_nonzero(background_mask))
    total_white_pixels = code_pixels + noise_pixels

    code_white_ratio = (code_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    total_white_ratio = (total_white_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    noise_ratio = (noise_pixels / background_pixels) * 100.0 if background_pixels > 0 else 0.0

    x1 = int(np.min(raw_pts[:, 0]))
    y1 = int(np.min(raw_pts[:, 1]))
    x2 = int(np.max(raw_pts[:, 0]))
    y2 = int(np.max(raw_pts[:, 1]))

    h, w = processed_img.shape[:2]
    x1, y1, x2, y2 = clamp_roi(x1, y1, x2, y2, w, h)

    debug_component_view = np.zeros_like(code_mask)
    debug_component_view[code_mask > 0] = 255
    debug_component_view[noise_mask > 0] = 120

    noise_mask_projected = project_mask_to_original(
        noise_mask, inner_crop_box, M, processed_img.shape, WARP_SIZE
    )
    code_mask_projected = project_mask_to_original(
        code_mask, inner_crop_box, M, processed_img.shape, WARP_SIZE
    )
    background_mask_projected = project_mask_to_original(
        background_mask, inner_crop_box, M, processed_img.shape, WARP_SIZE
    )

    return {
        "found": True,
        "code_white_ratio": code_white_ratio,
        "noise_ratio": noise_ratio,
        "total_white_ratio": total_white_ratio,
        "code_pixels": code_pixels,
        "noise_pixels": noise_pixels,
        "total_white_pixels": total_white_pixels,
        "total_pixels": total_pixels,
        "background_pixels": background_pixels,
        "noise_blob_count": noise_blob_count,
        "roi_box": (x1, y1, x2, y2),
        "warped_detect": warped_detect,
        "warped_noise": warped_noise,
        "inner_detect": inner_detect_bin,
        "inner_noise": inner_noise_bin,
        "code_mask": code_mask,
        "noise_mask": noise_mask,
        "background_mask": background_mask,
        "component_view": debug_component_view,
        "M": M,
        "inner_crop_box": inner_crop_box,
        "noise_mask_projected": noise_mask_projected,
        "code_mask_projected": code_mask_projected,
        "background_mask_projected": background_mask_projected
    }


def reset_stats():
    return {
        "current_round": 1,
        "total_frames": 0,
        "counted_frames": 0,
        "target_detect_frames": 0,
        "raw_detect_frames": 0,
        "total_times": [],
        "code_white_ratio_list": [],
        "noise_ratio_list": [],
        "noise_pixels_list": [],
        "noise_blob_count_list": [],
        "total_white_ratio_list": [],
        "finished": False,
        "round_detect_rates": [],
        "round_raw_rates": [],
        "round_fps_list": [],
        "round_median_total_list": [],
        "round_code_white_list": [],
        "round_noise_list": [],
        "round_noise_pixels_list": [],
        "round_noise_blob_count_list": [],
        "round_total_white_list": [],
        "avg_detect_rate": 0.0,
        "avg_raw_rate": 0.0,
        "avg_fps": 0.0,
        "avg_median_total": 0.0,
        "avg_code_white": 0.0,
        "avg_noise": 0.0,
        "avg_noise_pixels": 0.0,
        "avg_noise_blob_count": 0.0,
        "avg_total_white": 0.0,
    }


def detect_target_multi_rotation(img, aruco_dict, aruco_params, target_id):
    orig_h, orig_w = img.shape[:2]

    rotations = [
        ("0", img),
        ("90", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        ("180", cv2.rotate(img, cv2.ROTATE_180)),
        ("270", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    fallback = None

    for rot_name, test_img in rotations:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            test_img,
            aruco_dict,
            parameters=aruco_params
        )

        if ids is None or len(ids) == 0:
            continue

        ids_flat = ids.flatten()
        mapped_corners = map_corners_back_to_original(
            corners,
            rot_name,
            orig_w,
            orig_h
        )

        for c, tag_id in zip(mapped_corners, ids_flat):
            tag_id = int(tag_id)
            is_valid, reason = validate_tag_geometry([c], img.shape)

            if tag_id == target_id and is_valid:
                return {
                    "found": True,
                    "rot_name": rot_name,
                    "used_img": img,
                    "corners": [c],
                    "ids": np.array([[tag_id]], dtype=np.int32),
                    "rejected": rejected if rejected is not None else [],
                    "reason": "target valid"
                }

            if tag_id == target_id and fallback is None:
                fallback = {
                    "found": False,
                    "rot_name": rot_name,
                    "used_img": img,
                    "corners": [c],
                    "ids": np.array([[tag_id]], dtype=np.int32),
                    "rejected": rejected if rejected is not None else [],
                    "reason": reason
                }

    if fallback is not None:
        return fallback

    return {
        "found": False,
        "rot_name": "0",
        "used_img": img,
        "corners": [],
        "ids": None,
        "rejected": [],
        "reason": "not found"
    }


def draw_text_shadow(img, text, org, color, font_scale=0.72, thickness=2, shadow_thickness=5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org, font, font_scale, (0, 0, 0), shadow_thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def draw_header_box(img, text, x, y, fg=(255, 255, 0), bg=(0, 0, 0), border=(255, 255, 0),
                    font_scale=0.68, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    pad_x = 10
    pad_y = 8

    x1 = x - pad_x
    y1 = y - text_h - pad_y
    x2 = x + text_w + pad_x
    y2 = y + baseline + pad_y

    cv2.rectangle(img, (x1, y1), (x2, y2), bg, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 2)

    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale, fg, thickness, cv2.LINE_AA)


def on_mouse_processed(event, x, y, flags, param):
    global manual_roi_points, manual_roi_corners, manual_collecting, roi_mode

    if roi_mode != ROI_MODE_MANUAL or not manual_collecting:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        manual_roi_points.append([x, y])
        print(f"[ROI] point {len(manual_roi_points)} = ({x}, {y})")

        if len(manual_roi_points) == 4:
            pts = np.array(manual_roi_points, dtype=np.float32)
            pts = order_points_visual(pts)
            manual_roi_corners = [pts.reshape((1, 4, 2))]
            manual_collecting = False
            manual_roi_points = []
            print("[ROI] Manual ROI saved (4 points complete)")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(manual_roi_points) > 0:
            manual_roi_points.pop()
            print(f"[ROI] Undo point -> remain {len(manual_roi_points)}")


picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": TARGET_RES,
        "format": "RGB888"
    },
    buffer_count=4
)

picam2.configure(config)
picam2.start()
time.sleep(1.0)

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_16h5
)

try:
    aruco_params = cv2.aruco.DetectorParameters_create()
except AttributeError:
    aruco_params = cv2.aruco.DetectorParameters()

aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.adaptiveThreshWinSizeStep = 10
aruco_params.minMarkerPerimeterRate = 0.01
aruco_params.maxMarkerPerimeterRate = 6.0
aruco_params.polygonalApproxAccuracyRate = 0.08
aruco_params.minCornerDistanceRate = 0.02
aruco_params.minDistanceToBorder = 2
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementWinSize = 7
aruco_params.cornerRefinementMaxIterations = 80
aruco_params.cornerRefinementMinAccuracy = 0.01

stats = reset_stats()
test_started = False
recent_hits = deque(maxlen=STABLE_HIT_WINDOW)
last_rotation = "0"

last_good_corners = None
last_good_measure_result = None
using_cached_measure = False

cv2.namedWindow("Processed", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Processed", on_mouse_processed)

cv2.namedWindow(RAW_MANUAL_WINDOW, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(RAW_MANUAL_WINDOW, on_mouse_processed)

try:
    if MANUAL_FOCUS:
        picam2.set_controls({
            "AfMode": 0,
            "LensPosition": float(LENS_POS)
        })
        print(f"[INFO] Focus locked at LensPosition={LENS_POS}")

    while True:
        t0 = time.perf_counter()

        frame = picam2.capture_array()
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        raw_manual_view = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_img = build_processed_image(gray)

        detect_result = detect_target_multi_rotation(
            processed_img,
            aruco_dict,
            aruco_params,
            TARGET_ID
        )

        corners = detect_result["corners"]
        ids = detect_result["ids"]
        target_detected_this_frame = detect_result["found"]
        last_rotation = detect_result["rot_name"]
        detect_reason = detect_result.get("reason", "")

        if target_detected_this_frame and corners is not None and len(corners) > 0:
            last_good_corners = corners

        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

        using_cached_measure = False
        active_measure_source = "AUTO-LIVE"
        measure_corners = None

        if roi_mode == ROI_MODE_MANUAL:
            if manual_roi_corners is not None:
                measure_corners = manual_roi_corners
                active_measure_source = "MANUAL"
            elif last_good_corners is not None:
                measure_corners = last_good_corners
                active_measure_source = "MANUAL-FALLBACK-AUTO"
            elif corners is not None and len(corners) > 0:
                measure_corners = corners
                active_measure_source = "MANUAL-LIVE-AUTO"
            else:
                measure_corners = None
                active_measure_source = "MANUAL-NO-ROI"
        else:
            measure_corners = corners
            if measure_corners is None or len(measure_corners) == 0:
                if last_good_corners is not None:
                    measure_corners = last_good_corners
                    active_measure_source = "AUTO-LAST"
                else:
                    active_measure_source = "AUTO-NO-ROI"

        measure_result = measure_noise_in_warped_tag(
            processed_img,
            measure_corners
        )

        if measure_result["found"]:
            last_good_measure_result = measure_result
        else:
            if last_good_measure_result is not None:
                measure_result = last_good_measure_result
                using_cached_measure = True
                active_measure_source += "+CACHED"

        current_code_white_ratio = measure_result["code_white_ratio"]
        current_total_white_ratio = measure_result["total_white_ratio"]
        current_noise_ratio = measure_result["noise_ratio"]
        current_noise_pixels = measure_result["noise_pixels"]
        current_noise_blob_count = measure_result["noise_blob_count"]
        current_background_pixels = measure_result["background_pixels"]

        tag_roi_box = measure_result["roi_box"]

        warped_detect = measure_result["warped_detect"]
        inner_detect = measure_result["inner_detect"]
        code_mask = measure_result["code_mask"]

        noise_mask = measure_result["noise_mask"]
        component_view = measure_result["component_view"]

        noise_mask_projected = measure_result["noise_mask_projected"]
        code_mask_projected = measure_result["code_mask_projected"]
        background_mask_projected = measure_result["background_mask_projected"]

        display = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        processed_no_overlay = cv2.cvtColor(processed_img.copy(), cv2.COLOR_GRAY2BGR)
        clean_view = cv2.cvtColor(processed_img.copy(), cv2.COLOR_GRAY2BGR)

        if background_mask_projected is not None:
            bg_overlay = np.zeros_like(display)
            bg_overlay[background_mask_projected > 0] = (40, 40, 40)
            display = cv2.addWeighted(display, 1.0, bg_overlay, 0.55, 0)

        if code_mask_projected is not None:
            display[code_mask_projected > 0] = (255, 255, 255)

        if noise_mask_projected is not None:
            display[noise_mask_projected > 0] = (0, 0, 255)

        recent_hits.append(1 if target_detected_this_frame else 0)
        stable_target_detected = (
            len(recent_hits) == STABLE_HIT_WINDOW and
            sum(recent_hits) >= STABLE_HIT_MIN and
            recent_hits[-1] == 1
        )

        if ids is not None and roi_mode == ROI_MODE_AUTO:
            ids_flat = ids.flatten()

            for tag_corners, tag_id in zip(corners, ids_flat):
                pts = tag_corners.reshape((4, 2)).astype(int)
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)

                x, y = pts[0]
                draw_text_shadow(display, f"ID: {tag_id}", (x, y - 10), (0, 255, 0), 0.7, 2, 5)

        if roi_mode == ROI_MODE_MANUAL and manual_roi_corners is not None:
            manual_pts = manual_roi_corners[0].reshape((4, 2)).astype(int)
            cv2.polylines(display, [manual_pts], True, (255, 0, 255), 2)

            mx, my = manual_pts[0]
            draw_text_shadow(display, "Tag ROI", (mx, max(30, my - 10)), (255, 0, 255), 0.6, 2, 5)

        elif tag_roi_box is not None:
            rx1, ry1, rx2, ry2 = tag_roi_box
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)
            draw_text_shadow(display, "Tag ROI", (rx1, max(30, ry1 - 10)), (255, 0, 255), 0.6, 2, 5)

        if manual_roi_corners is not None:
            manual_pts_raw = manual_roi_corners[0].reshape((4, 2)).astype(int)
            cv2.polylines(raw_manual_view, [manual_pts_raw], True, (0, 165, 255), 2)

            mx, my = manual_pts_raw[0]
            draw_text_shadow(raw_manual_view,
                            "Manual ROI",
                            (mx, max(30, my - 10)),
                            (0, 165, 255), 0.6, 2, 5)

        if manual_collecting and len(manual_roi_points) > 0:
            pts_tmp_raw = np.array(manual_roi_points, dtype=np.int32)

            for i, (px, py) in enumerate(pts_tmp_raw):
                cv2.circle(raw_manual_view, (px, py), 5, (0, 255, 255), -1)
                draw_text_shadow(raw_manual_view,
                                str(i + 1),
                                (px + 6, py - 6),
                                (0, 255, 255), 0.55, 2, 5)

            if len(pts_tmp_raw) >= 2:
                cv2.polylines(raw_manual_view, [pts_tmp_raw], False, (0, 255, 255), 1)

        if test_started and not stats["finished"]:
            stats["total_frames"] += 1
            in_warmup = stats["total_frames"] <= WARMUP_FRAMES

            if not in_warmup:
                stats["counted_frames"] += 1
                stats["total_times"].append(total_ms)
                stats["code_white_ratio_list"].append(current_code_white_ratio)
                stats["noise_ratio_list"].append(current_noise_ratio)
                stats["noise_pixels_list"].append(current_noise_pixels)
                stats["noise_blob_count_list"].append(current_noise_blob_count)
                stats["total_white_ratio_list"].append(current_total_white_ratio)

                if target_detected_this_frame:
                    stats["raw_detect_frames"] += 1

                if stable_target_detected:
                    stats["target_detect_frames"] += 1

            if stats["total_frames"] >= TEST_FRAMES:
                counted = stats["counted_frames"]

                if counted > 0:
                    round_raw_rate = (stats["raw_detect_frames"] / counted) * 100.0
                    round_detect_rate = (stats["target_detect_frames"] / counted) * 100.0
                    round_median = float(np.median(stats["total_times"]))
                    round_fps = 1000.0 / round_median if round_median > 0 else 0.0
                    round_code_white = float(np.median(stats["code_white_ratio_list"]))
                    round_noise = float(np.median(stats["noise_ratio_list"]))
                    round_noise_pixels = float(np.median(stats["noise_pixels_list"]))
                    round_noise_blobs = float(np.median(stats["noise_blob_count_list"]))
                    round_total_white = float(np.median(stats["total_white_ratio_list"]))
                else:
                    round_raw_rate = 0.0
                    round_detect_rate = 0.0
                    round_median = 0.0
                    round_fps = 0.0
                    round_code_white = 0.0
                    round_noise = 0.0
                    round_noise_pixels = 0.0
                    round_noise_blobs = 0.0
                    round_total_white = 0.0

                stats["round_raw_rates"].append(round_raw_rate)
                stats["round_detect_rates"].append(round_detect_rate)
                stats["round_fps_list"].append(round_fps)
                stats["round_median_total_list"].append(round_median)
                stats["round_code_white_list"].append(round_code_white)
                stats["round_noise_list"].append(round_noise)
                stats["round_noise_pixels_list"].append(round_noise_pixels)
                stats["round_noise_blob_count_list"].append(round_noise_blobs)
                stats["round_total_white_list"].append(round_total_white)

                stats["avg_raw_rate"] = float(np.mean(stats["round_raw_rates"]))
                stats["avg_detect_rate"] = float(np.mean(stats["round_detect_rates"]))
                stats["avg_fps"] = float(np.mean(stats["round_fps_list"]))
                stats["avg_median_total"] = float(np.mean(stats["round_median_total_list"]))
                stats["avg_code_white"] = float(np.mean(stats["round_code_white_list"]))
                stats["avg_noise"] = float(np.mean(stats["round_noise_list"]))
                stats["avg_noise_pixels"] = float(np.mean(stats["round_noise_pixels_list"]))
                stats["avg_noise_blob_count"] = float(np.mean(stats["round_noise_blob_count_list"]))
                stats["avg_total_white"] = float(np.mean(stats["round_total_white_list"]))

                print(
                    f"[INFO] Round {stats['current_round']}/{TEST_ROUNDS} finished | "
                    f"Raw Rate = {round_raw_rate:.2f}% | "
                    f"Stable Rate = {round_detect_rate:.2f}% | "
                    f"Median Total = {round_median:.2f} ms | "
                    f"FPS = {round_fps:.1f} | "
                    f"Code White = {round_code_white:.2f}% | "
                    f"Noise Ratio = {round_noise:.2f}% | "
                    f"Noise Px = {round_noise_pixels:.0f} | "
                    f"Noise Blobs = {round_noise_blobs:.0f} | "
                    f"Total White = {round_total_white:.2f}%"
                )

                if stats["current_round"] >= TEST_ROUNDS:
                    stats["finished"] = True
                    test_started = False

                    print(f"[INFO] All {TEST_ROUNDS} rounds finished")
                    print(f"[INFO] Average Raw Rate       = {stats['avg_raw_rate']:.2f}%")
                    print(f"[INFO] Average Stable Rate    = {stats['avg_detect_rate']:.2f}%")
                    print(f"[INFO] Average FPS            = {stats['avg_fps']:.1f}")
                    print(f"[INFO] Average Median Total   = {stats['avg_median_total']:.2f} ms")
                    print(f"[INFO] Average Code White     = {stats['avg_code_white']:.2f}%")
                    print(f"[INFO] Average Noise Ratio    = {stats['avg_noise']:.2f}%")
                    print(f"[INFO] Average Noise Pixels   = {stats['avg_noise_pixels']:.0f} px")
                    print(f"[INFO] Average Noise Blobs    = {stats['avg_noise_blob_count']:.0f}")
                    print(f"[INFO] Average Total White    = {stats['avg_total_white']:.2f}%")
                else:
                    stats["current_round"] += 1
                    stats["total_frames"] = 0
                    stats["counted_frames"] = 0
                    stats["target_detect_frames"] = 0
                    stats["raw_detect_frames"] = 0
                    stats["total_times"] = []
                    stats["code_white_ratio_list"] = []
                    stats["noise_ratio_list"] = []
                    stats["noise_pixels_list"] = []
                    stats["noise_blob_count_list"] = []
                    stats["total_white_ratio_list"] = []
                    recent_hits.clear()

        counted = stats["counted_frames"]

        if counted > 0:
            raw_detect_rate = (stats["raw_detect_frames"] / counted) * 100.0
            stable_detect_rate = (stats["target_detect_frames"] / counted) * 100.0
            median_total = float(np.median(stats["total_times"]))
            fps = 1000.0 / median_total if median_total > 0 else 0.0
            median_code_white = float(np.median(stats["code_white_ratio_list"]))
            median_noise = float(np.median(stats["noise_ratio_list"]))
            median_noise_pixels = float(np.median(stats["noise_pixels_list"]))
            median_noise_blobs = float(np.median(stats["noise_blob_count_list"]))
        else:
            raw_detect_rate = 0.0
            stable_detect_rate = 0.0
            median_total = 0.0
            fps = 0.0
            median_code_white = current_code_white_ratio
            median_noise = current_noise_ratio
            median_noise_pixels = current_noise_pixels
            median_noise_blobs = current_noise_blob_count

        remaining_frames = TEST_FRAMES - stats["total_frames"]
        if remaining_frames < 0:
            remaining_frames = 0

        if test_started:
            status_text = "WARMUP" if stats["total_frames"] <= WARMUP_FRAMES else "RUNNING"
            status_color = (0, 165, 255) if stats["total_frames"] <= WARMUP_FRAMES else (0, 255, 0)
        elif stats["finished"]:
            status_text = "FINISHED"
            status_color = (255, 255, 0)
        else:
            status_text = "WAITING - Press S"
            status_color = (0, 255, 255)

        draw_text_shadow(display, status_text, (20, 40), status_color, 0.9, 2, 5)

        draw_text_shadow(display, f"Round: {stats['current_round']}/{TEST_ROUNDS}", (20, 90), (255, 255, 0), 0.8, 2, 5)
        draw_text_shadow(display, f"Frames Left: {remaining_frames}", (20, 135), (255, 255, 0), 0.8, 2, 5)

        AVG_X = 900
        AVG_Y1 = 110
        AVG_Y2 = 290
        AVG_W = 340

        draw_header_box(display, "=== AVERAGE ===", AVG_X, 140,
                        fg=(255, 255, 0), bg=(0, 0, 0), border=(255, 255, 0),
                        font_scale=0.68, thickness=2)

        draw_text_shadow(display, f"Avg Raw:    {stats['avg_raw_rate']:.2f}%", (AVG_X, 185), (0, 200, 255), 0.72, 2, 5)
        draw_text_shadow(display, f"Avg Median: {stats['avg_median_total']:.2f} ms", (AVG_X, 225), (0, 255, 0), 0.72, 2, 5)
        draw_text_shadow(display, f"Avg Ratio:  {stats['avg_noise']:.2f}%", (AVG_X, 265), (0, 255, 150), 0.72, 2, 5)

        draw_text_shadow(display, f"Raw Detect Rate: {raw_detect_rate:.2f}%", (20, 180), (0, 200, 255), 0.72, 2, 5)
        draw_text_shadow(display, f"Stable Detect Rate: {stable_detect_rate:.2f}%", (20, 220), (0, 255, 255), 0.72, 2, 5)
        draw_text_shadow(display, f"Code White: {median_code_white:.2f}%", (20, 260), (255, 120, 255), 0.72, 2, 5)
        draw_text_shadow(display, f"Noise Px: {median_noise_pixels:.0f}  Blobs: {median_noise_blobs:.0f}  Ratio: {median_noise:.2f}%", (20, 300), (0, 255, 150), 0.72, 2, 5)
        draw_text_shadow(display, f"Background Px: {current_background_pixels}", (20, 340), (180, 180, 255), 0.68, 2, 5)
        draw_text_shadow(display, f"Median Total Time: {median_total:.2f} ms", (20, 380), (0, 255, 0), 0.72, 2, 5)
        draw_text_shadow(display, f"FPS: {fps:.1f}", (20, 420), (0, 255, 0), 0.72, 2, 5)
        draw_text_shadow(display, "Pipeline: CLAHE2.0 + Binary100 + Erode7", (20, 460), (255, 255, 0), 0.58, 2, 5)
        draw_text_shadow(display, f"NoiseArea:{MIN_NOISE_BLOB_AREA}-{MAX_NOISE_BLOB_AREA}  CodeArea:{MIN_CODE_BLOB_AREA}", (20, 500), (255, 255, 0), 0.58, 2, 5)

        state_color = (0, 255, 0) if stable_target_detected else (0, 100, 255)
        draw_text_shadow(display,
                         f"Raw: {'YES' if target_detected_this_frame else 'NO '}  Stable: {'YES' if stable_target_detected else 'NO '}  Rot:{last_rotation} {detect_reason}",
                         (20, 540), state_color, 0.60, 2, 5)

        draw_text_shadow(display,
                         f"Measure: {'CACHED' if using_cached_measure else 'LIVE'}",
                         (20, 570),
                         (0, 255, 255) if using_cached_measure else (0, 255, 0),
                         0.55, 2, 5)

        draw_text_shadow(display,
                         f"ROI Mode: {roi_mode} | Source: {active_measure_source}",
                         (20, 600),
                         (255, 255, 0) if roi_mode == ROI_MODE_MANUAL else (255, 200, 0),
                         0.55, 2, 5)

        if manual_collecting:
            draw_text_shadow(display,
                            f"Manual ROI: click 4 points on Raw Manual ROI ({len(manual_roi_points)}/4) | Right click = undo",
                            (20, 630),
                            (0, 255, 255), 0.52, 2, 5)
        else:
            draw_text_shadow(display,
                             "RED = Noise | M=Manual ROI  C=Clear Manual ROI  S=Start  R=Reset  A/D=MaxNoise  ESC=Exit",
                             (20, 630),
                             (200, 200, 200), 0.52, 2, 5)

        cv2.imshow("Processed", display)
        cv2.imshow(RAW_MANUAL_WINDOW, raw_manual_view)

        if warped_detect is not None:
            cv2.imshow("01 Warped Tag", warped_detect)

        if inner_detect is not None:
            cv2.imshow("02 Tag Full Area", inner_detect)

        if code_mask is not None:
            cv2.imshow("03 Code White Blob", code_mask)

        if noise_mask_projected is not None:
            cv2.imshow("04 Noise Speckle", noise_mask_projected)

        if component_view is not None:
            cv2.imshow("05 Component View", component_view)

        if background_mask_projected is not None:
            cv2.imshow("06 Background Mask", background_mask_projected)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            stats = reset_stats()
            test_started = True
            recent_hits.clear()
            last_good_corners = None
            last_good_measure_result = None
            using_cached_measure = False
            print(f"[INFO] Test started | {TEST_ROUNDS} rounds x {TEST_FRAMES} frames")

        elif key == ord("r"):
            stats = reset_stats()
            test_started = False
            recent_hits.clear()
            last_good_corners = None
            last_good_measure_result = None
            using_cached_measure = False
            print("[INFO] Test reset")

        elif key == ord("m"):
            if roi_mode == ROI_MODE_AUTO:
                roi_mode = ROI_MODE_MANUAL
                manual_collecting = True
                manual_roi_points = []

                if last_good_corners is not None:
                    pts = last_good_corners[0].reshape((4, 2)).astype(np.float32).copy()
                    manual_roi_corners = [pts.reshape((1, 4, 2))]
                    print("[ROI] MANUAL mode ON | preloaded last detected ROI | click 4 points to override")
                else:
                    print("[ROI] MANUAL mode ON | click 4 points on Raw Manual ROI window")
            else:
                roi_mode = ROI_MODE_AUTO
                manual_collecting = False
                manual_roi_points = []
                print("[ROI] AUTO mode ON")

        elif key == ord("c"):
            manual_roi_corners = None
            manual_roi_points = []
            manual_collecting = False
            print("[ROI] Manual ROI cleared")

        elif key == ord("a"):
            MAX_NOISE_BLOB_AREA = max(10, MAX_NOISE_BLOB_AREA - 100)
            print(f"[TUNE] MAX_NOISE_BLOB_AREA = {MAX_NOISE_BLOB_AREA}")

        elif key == ord("d"):
            MAX_NOISE_BLOB_AREA = min(999999, MAX_NOISE_BLOB_AREA + 100)
            print(f"[TUNE] MAX_NOISE_BLOB_AREA = {MAX_NOISE_BLOB_AREA}")

        elif key == 27:
            break

finally:
    cv2.destroyAllWindows()
    picam2.close()