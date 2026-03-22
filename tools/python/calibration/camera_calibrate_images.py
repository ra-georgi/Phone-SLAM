import cv2
import numpy as np
import glob

# ===== USER SETTINGS =====
image_paths = glob.glob("recordings/Calibration/*.jpg")
pattern_size = (7, 9)      # number of INNER corners per row and column
square_size = 20.0         # mm, or any consistent unit
# =========================

# 3D points in checkerboard coordinates
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []   # 3D points in world coordinates
imgpoints = []   # 2D points in image coordinates

image_size = None

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read {path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if found:
        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        objpoints.append(objp)
        imgpoints.append(corners_refined)

        # Optional: visualize detections
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern_size, corners_refined, found)
        cv2.imshow("Corners", vis)
        cv2.waitKey(100)
    else:
        print(f"Checkerboard not found in {path}")

cv2.destroyAllWindows()

if len(objpoints) < 10:
    raise RuntimeError("Too few valid images. Capture more views.")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

print("RMS reprojection error:", ret)
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs.ravel())

# Compute mean reprojection error
total_error = 0
for i in range(len(objpoints)):
    projected, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
    )
    error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
    total_error += error

print("Mean reprojection error:", total_error / len(objpoints))