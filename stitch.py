import cv2
import numpy as np
import os


def get_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Harris-Laplace for scale-invariant corner detection
    detector = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
    kp = detector.detect(gray)
    # FREAK for fast, accurate binary description
    descriptor = cv2.xfeatures2d.FREAK_create()
    kp, des = descriptor.compute(img, kp)
    return kp, des


def get_homography(img_src, img_dst):
    kp1, des1 = get_features(img_src)
    kp2, des2 = get_features(img_dst)
    if des1 is None or des2 is None: return None
   
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    # Lowe's Ratio Test to filter out ambiguous matches
    good = [m for m, n in raw_matches if m.distance < 0.7 * n.distance]
   
    if len(good) < 10: return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # RANSAC for robust alignment
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H


def stitch_3_ultra_seamless(im1_path, im2_path, im3_path):
    # 1. LOAD IMAGES
    img1 = cv2.imread(im1_path)
    img2 = cv2.imread(im2_path)
    img3 = cv2.imread(im3_path)


    if any(x is None for x in [img1, img2, img3]):
        return print("Error: Check image paths in your 'images' folder.")


    # 2. ALIGNMENT
    print("Aligning images using Harris-Laplace + FREAK...")
    H12 = get_homography(img1, img2)
    H32 = get_homography(img3, img2)
    if H12 is None or H32 is None: return print("Error: Alignment failed.")


    # 3. WARPING TO LARGE CANVAS
    h, w = img2.shape[:2]
    T = np.array([[1, 0, w], [0, 1, h//2], [0, 0, 1]], dtype=np.float32)
    canvas_size = (w * 3, h * 2)


    warp1 = cv2.warpPerspective(img1, T @ H12, canvas_size)
    warp2 = cv2.warpPerspective(img2, T, canvas_size)
    warp3 = cv2.warpPerspective(img3, T @ H32, canvas_size)


    # 4. EXPOSURE COMPENSATION (The Secret to No Seams)
    print("Compensating for exposure differences...")
    warped_images = [warp1, warp2, warp3]
   
    def get_binary_mask(iw):
        gray = cv2.cvtColor(iw, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]


    masks = [get_binary_mask(w) for w in warped_images]
   
    # Gain compensator balances brightness between images
    compensator = cv2.detail_GainCompensator()
    # Feed corner positions (0,0) and images/masks
    corners = [(0, 0), (0, 0), (0, 0)]
    compensator.feed(corners, warped_images, masks)
   
    # Apply compensation
    for i in range(len(warped_images)):
        compensator.apply(i, (0,0), warped_images[i], masks[i])


    # 5. MULTI-BAND BLENDING (LAPLACIAN PYRAMID)
    print("Final Multi-band blending...")
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(10) # More bands for smoother color transitions
    blender.prepare((0, 0, canvas_size[0], canvas_size[1]))
   
    for i in range(len(warped_images)):
        # Erode masks slightly to hide warping artifacts at edges
        eroded_mask = cv2.erode(masks[i], np.ones((10, 10), np.uint8))
        blender.feed(warped_images[i].astype(np.int16), eroded_mask, (0, 0))
   
    result, result_mask = blender.blend(None, None)
    final = np.clip(result, 0, 255).astype(np.uint8)


    # 6. AUTO-CROP & SAVE
    cnts, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, cw, ch = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        final = final[y:y+ch, x:x+cw]


    output_path = "final_ultra_seamless.jpg"
    cv2.imwrite(output_path, final)
    print(f"Success! Output saved in root directory: {output_path}")
   
    cv2.imshow("Ultra Seamless Result", cv2.resize(final, (1200, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run the final script
stitch_3_ultra_seamless('images/img1.JPG', 'images/img2.JPG', 'images/img3.JPG')

