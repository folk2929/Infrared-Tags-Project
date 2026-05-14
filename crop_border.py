import cv2
import numpy as np
import os
import sys


def find_crop_box(img, threshold=30, padding=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    corners = [gray[0,0], gray[0,w-1], gray[h-1,0], gray[h-1,w-1]]
    bg_color = int(np.median(corners))

    diff = np.abs(gray.astype(int) - bg_color)
    mask = diff > threshold

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return 0, 0, w, h

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding + 1)
    y2 = min(h, y2 + padding + 1)

    return x1, y1, x2, y2


def crop_image(img_path, out_path, threshold=30, padding=2, preview=False):
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [SKIP] Cannot read: {img_path}")
        return False

    h, w = img.shape[:2]
    x1, y1, x2, y2 = find_crop_box(img, threshold, padding)
    cropped = img[y1:y2, x1:x2]
    ch, cw = cropped.shape[:2]

    cv2.imwrite(out_path, cropped)
    print(f"  [OK] {os.path.basename(img_path)}  {w}x{h} -> {cw}x{ch}  saved: {out_path}")

    if preview:
        sw, sh = 1280, 720
        scale = min(sw / (w * 2 + 20), sh / max(h, ch), 1.0)
        left  = cv2.resize(img, (int(w*scale), int(h*scale)))
        right = cv2.resize(cropped, (int(cw*scale), int(ch*scale)))
        max_h = max(left.shape[0], right.shape[0])
        def pad_h(im, target):
            diff = target - im.shape[0]
            return cv2.copyMakeBorder(im, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=(50,50,50))
        left  = pad_h(left, max_h)
        right = pad_h(right, max_h)
        divider = np.full((max_h, 10, 3), 80, dtype=np.uint8)
        side_by_side = np.hstack([left, divider, right])
        cv2.imshow("Before (left)  |  After (right)  -- any key = next", side_by_side)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True


def process_folder(folder, threshold=30, padding=2, preview=False):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    files = [f for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in exts
             and '_cropped' not in f]

    if not files:
        print("No image files found.")
        return

    out_dir = os.path.join(folder, "cropped")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nFound {len(files)} images -> saving to: {out_dir}\n")

    ok = 0
    for fname in sorted(files):
        in_path  = os.path.join(folder, fname)
        base, ext = os.path.splitext(fname)
        out_path = os.path.join(out_dir, base + "_cropped" + ext)
        if crop_image(in_path, out_path, threshold, padding, preview):
            ok += 1

    print(f"\nDone! {ok}/{len(files)} images processed.")
    print(f"Output folder: {out_dir}")


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1].strip().strip('"').strip("'")
    else:
        target = input("Image file or folder path: ").strip().strip('"').strip("'")

    if not os.path.exists(target):
        print(f"Not found: {target}")
        return

    try:
        thr_input = input("Border threshold (default 30, higher = stricter): ").strip()
        threshold = int(thr_input) if thr_input else 30
    except:
        threshold = 30

    preview_input = input("Show preview for each image? (y/n, default n): ").strip().lower()
    preview = preview_input == 'y'

    print()
    if os.path.isdir(target):
        process_folder(target, threshold=threshold, padding=3, preview=preview)
    else:
        base, ext = os.path.splitext(target)
        out_path = base + "_cropped" + ext
        crop_image(target, out_path, threshold=threshold, padding=3, preview=preview)


if __name__ == "__main__":
    main()
