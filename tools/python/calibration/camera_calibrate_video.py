import cv2
import numpy as np
import os

# ===== USER SETTINGS =====
video_path = "recordings/Calibration/Camera/1774175875889.mp4"
pattern_size = (7, 9)      # number of INNER corners per row and column
square_size = 20.0         # mm, or any consistent unit

frame_step = 10            # try every Nth frame
min_valid_views = 10
save_debug = True
debug_dir = "recordings/Calibration/debug_detections"
display = True
# =========================

if save_debug:
    os.makedirs(debug_dir, exist_ok=True)

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

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

frame_idx = 0
accepted = 0

while True:
    ret_frame, frame = cap.read()
    if not ret_frame:
        break

    if frame_idx % frame_step != 0:
        frame_idx += 1
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    # More robust than plain findChessboardCorners on many videos
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:
        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        objpoints.append(objp.copy())
        imgpoints.append(corners_refined)
        accepted += 1

        vis = frame.copy()
        cv2.drawChessboardCorners(vis, pattern_size, corners_refined, found)

        if save_debug:
            out_path = os.path.join(debug_dir, f"detected_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, vis)

        if display:
            cv2.imshow("Corners", vis)
            key = cv2.waitKey(50)
            if key == 27:  # ESC to stop early
                break
    else:
        print(f"Checkerboard not found in frame {frame_idx}")

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

print(f"Accepted views: {len(objpoints)}")

if len(objpoints) < min_valid_views:
    raise RuntimeError(
        f"Too few valid views ({len(objpoints)}). Capture more diverse checkerboard poses."
    )

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