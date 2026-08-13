import cv2
import sys
import os
import numpy as np

from insightface.app import FaceAnalysis

def main():
    crop_face("tools/processing/pictures/the rock.png")


def crop_face(path):
# ============================================================
# SETTINGS
# ============================================================

    OUTPUT_DIR = "tools/processing/stored-faces"
    FILE_NAME = path

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )
    


    # ============================================================
    # READ IMAGE
    # ============================================================

    img = cv2.imread(
        FILE_NAME
    )

    if img is None:
        print("Image not Found")
        sys.exit()


    # ============================================================
    # LOAD SCRFD FACE DETECTOR
    # ============================================================

    app = FaceAnalysis(
        allowed_modules=["detection"]
    )

    app.prepare(
        ctx_id=-1,
        det_size=(640, 640)
    )


    # ============================================================
    # DETECT FACES
    # ============================================================

    faces = app.get(img)

    if len(faces) == 0:
        print("No face detected")
        sys.exit()


    # ============================================================
    # SELECT LARGEST FACE
    # ============================================================

    face = max(
        faces,
        key=lambda f: (
            f.bbox[2] - f.bbox[0]
        ) * (
            f.bbox[3] - f.bbox[1]
        )
    )


    # ============================================================
    # GET FACIAL LANDMARKS
    # ============================================================

    # SCRFD gives five landmarks:
    #
    # 0 = left eye
    # 1 = right eye
    # 2 = nose
    # 3 = left mouth
    # 4 = right mouth

    kps = np.asarray(
        face.kps,
        dtype=np.float32
    )


    left_eye = kps[0]
    right_eye = kps[1]

    nose = kps[2]

    left_mouth = kps[3]
    right_mouth = kps[4]


    # ============================================================
    # CALCULATE FACE SCALE
    # ============================================================

    # Distance between the eyes is used as the scale
    # for the rest of the crop.

    eye_width = np.linalg.norm(
        right_eye - left_eye
    )


    eye_center_y = (
        left_eye[1] +
        right_eye[1]
    ) / 2


    mouth_center_y = (
        left_mouth[1] +
        right_mouth[1]
    ) / 2


    # ============================================================
    # TOP EDGE
    # ============================================================

    # SCRFD doesn't provide eyebrow landmarks.
    #
    # Estimate eyebrow level slightly above the eyes.

    eyebrow_y = (
        eye_center_y -
        eye_width * 0.28
    )


    # Start above eyebrow level. Increase padding to include more forehead.

    # How much extra forehead to include, relative to eye width.
    forehead_pad = eye_width * 0.40

    top = int(
        eyebrow_y -
        forehead_pad
    )


    # ============================================================
    # BOTTOM EDGE
    # ============================================================

    # End around the lower lip / upper chin area.

    bottom = int(
        mouth_center_y +
        eye_width * 0.38
    )


    # ============================================================
    # LEFT EDGE
    # ============================================================

    # Half an eye-width outside the left eye.

    left = int(
        left_eye[0] -
        eye_width * 0.50
    )


    # ============================================================
    # RIGHT EDGE
    # ============================================================

    # Half an eye-width outside the right eye.

    right = int(
        right_eye[0] +
        eye_width * 0.50
    )


    # ============================================================
    # KEEP CROP INSIDE IMAGE
    # ============================================================

    height, width = img.shape[:2]


    left = max(
        0,
        left
    )

    top = max(
        0,
        top
    )

    right = min(
        width,
        right
    )

    bottom = min(
        height,
        bottom
    )


    # ============================================================
    # VALIDATE CROP
    # ============================================================

    if right <= left or bottom <= top:

        print("Invalid crop")
        sys.exit()


    # ============================================================
    # CROP
    # ============================================================

    # IMPORTANT:
    #
    # The database embedding pipeline reads a face image and
    # immediately resizes it to 112x112 before calling the model.
    # To keep the saved crop consistent with that observable
    # interface, we extract the face region first and then resize
    # both outputs to the same 112x112 target.

    face_color = img[
        top:bottom,
        left:right
    ]


    if face_color.size == 0:

        print("Empty crop")
        sys.exit()


    face_color = cv2.resize(
        face_color,
        (112, 112),
        interpolation=cv2.INTER_AREA
    )


    # ============================================================
    # GRAYSCALE VERSION
    # ============================================================

    gray_img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )


    face_gray = gray_img[
        top:bottom,
        left:right
    ]


    face_gray = cv2.resize(
        face_gray,
        (112, 112),
        interpolation=cv2.INTER_AREA
    )


    # ============================================================
    # SAVE CROPS
    # ============================================================

    color_path = os.path.join(
        OUTPUT_DIR,
        "face_color.jpg"
    )

    gray_path = os.path.join(
        OUTPUT_DIR,
        "face_gray.jpg"
    )


    cv2.imwrite(
        color_path,
        face_color
    )

    cv2.imwrite(
        gray_path,
        face_gray
    )


    # ============================================================
    # OUTPUT INFORMATION
    # ============================================================

    print()
    print("Face successfully cropped!")
    print()

    print(
        f"Crop coordinates: "
        f"({left}, {top}) -> ({right}, {bottom})"
    )

    print(
        f"Crop dimensions: "
        f"{right - left} x {bottom - top}"
    )

    print()

    print("Color crop:")
    print(color_path)

    print()

    print("Grayscale crop:")
    print(gray_path)


if __name__ == "__main__":
    main()