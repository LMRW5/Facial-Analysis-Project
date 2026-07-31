import cv2
import sys
import os

# Create output folder
output_dir = "tools/processing/stored-faces"
os.makedirs(output_dir, exist_ok=True)

# Load face detector
alg = "models/haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

if haar_cascade.empty():
    print("Could not load Haar Cascade")
    sys.exit()

# Read image (keep color version)
file_name = "tools/processing/blankface.jpg"
img = cv2.imread(file_name)

if img is None:
    print("Image not Found")
    sys.exit()

# Convert to grayscale for detection
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = haar_cascade.detectMultiScale(
    gray_img,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(100, 100)
)

if len(faces) == 0:
    print("No face detected")
    sys.exit()

# Pick the largest face (main face)
x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

# Crop the face from both versions
face_color = img[y:y+h, x:x+w]
face_gray = gray_img[y:y+h, x:x+w]

# Save color version
cv2.imwrite(
    f"{output_dir}/face_color.jpg",
    cv2.resize(face_color,(112,112))
)

# Save grayscale version
cv2.imwrite(
    f"{output_dir}/face_gray.jpg",
    cv2.resize(face_gray,(112,112))
)

print("Saved main face:")
print(f"{output_dir}/face_color.jpg")
print(f"{output_dir}/face_gray.jpg")