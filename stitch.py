import cv2
import numpy as np
import os

print("Running Harris-Laplace + ORB + RANSAC + Blending pipeline...")

# -----------------------------
# Setup
# -----------------------------
image_dir = "images"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

files = sorted(os.listdir(image_dir))
images = [cv2.imread(os.path.join(image_dir, f)) for f in files]
assert all(img is not None for img in images)

# -----------------------------
# Stage 1: Harris-Laplace
# -----------------------------
def harris_laplace(img, max_pts=1200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    harris = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    harris = cv2.dilate(harris, None)

    coords = np.argwhere(harris > 0.01 * harris.max())
    responses = harris[coords[:,0], coords[:,1]]

    idx = np.argsort(responses)[-max_pts:]
    keypoints = [cv2.KeyPoint(float(coords[i][1]), float(coords[i][0]), 16) for i in idx]
    return gray, keypoints

# -----------------------------
# Stage 2: ORB Descriptor
# -----------------------------
orb = cv2.ORB_create()
keypoints = []
descriptors = []

for i, img in enumerate(images):
    gray, kp = harris_laplace(img)
    kp, des = orb.compute(gray, kp)
    if des is None:
        kp, des = [], None
    keypoints.append(kp)
    descriptors.append(des)
    vis = cv2.drawKeypoints(img, kp[:300], None, (0,255,0))
    cv2.imwrite(f"{output_dir}/stage2_orb_{i}.jpg", vis)

# -----------------------------
# Stage 3: KNN + Ratio Test
# -----------------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
good_matches_all = []

for i in range(len(images)-1):
    if descriptors[i] is None or descriptors[i+1] is None:
        good_matches_all.append([])
        continue

    knn = bf.knnMatch(descriptors[i], descriptors[i+1], k=2)
    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    good_matches_all.append(good)

    vis = cv2.drawMatches(images[i], keypoints[i], images[i+1], keypoints[i+1], good[:40], None)
    cv2.imwrite(f"{output_dir}/stage3_matches_{i}.jpg", vis)

# -----------------------------
# Stage 4: RANSAC + Blending
# -----------------------------
panorama = images[0]
H_total = np.eye(3)

for i in range(len(good_matches_all)):
    matches = good_matches_all[i]
    if len(matches) < 8:
        print(f"Skipping pair {i}-{i+1}")
        continue

    pts_prev = np.float32([keypoints[i][m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts_next = np.float32([keypoints[i+1][m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, _ = cv2.findHomography(pts_next, pts_prev, cv2.RANSAC, 4.0)
    if H is None:
        continue

    H_total = H_total @ H

    h_p, w_p = panorama.shape[:2]
    h_i, w_i = images[i+1].shape[:2]

    warped = cv2.warpPerspective(images[i+1], H_total, (w_p + w_i, max(h_p, h_i)))

    # Simple blending
    mask_panorama = (panorama > 0).astype(np.float32)
    mask_warped = (warped > 0).astype(np.float32)

    blended = warped.copy()
    overlap = np.logical_and(mask_panorama.any(axis=2), mask_warped.any(axis=2))
    for c in range(3):
        blended[:,:,c] = np.where(
            overlap,
            (warped[:,:,c].astype(np.float32) + panorama[:,:,c].astype(np.float32)) / 2,
            blended[:,:,c]
        )
    blended[0:h_p, 0:w_p] = np.where(mask_panorama.astype(bool), panorama, blended[0:h_p, 0:w_p])

    panorama = blended
    cv2.imwrite(f"{output_dir}/stage4_stitch_{i}.jpg", panorama)

# -----------------------------
# Auto-crop black borders
# -----------------------------
gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
coords = cv2.findNonZero(thresh)
x, y, w, h = cv2.boundingRect(coords)
panorama = panorama[y:y+h, x:x+w]

cv2.imwrite(f"{output_dir}/FINAL_PANORAMA.jpg", panorama)
print("✅ Final panorama saved")
