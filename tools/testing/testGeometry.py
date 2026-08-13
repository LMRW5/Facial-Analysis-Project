import cv2
import mediapipe as mp

from tools.objects.faceGeometry import FaceGeometry


MODEL_PATH = "models/face_landmarker.task"


def get_landmarks(image_path):

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=(
            mp.tasks.vision.RunningMode.IMAGE
        ),
        num_faces=1
    )

    image = cv2.imread(
        image_path
    )

    if image is None:
        return None

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        result = landmarker.detect(
            mp_image
        )

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]


# ---------------------------------------------
# INPUT
# ---------------------------------------------

input_landmarks = get_landmarks(
    "tools/processing/pictures/Ryan.png"
)


match_landmarks = get_landmarks(
    "tools/processing/pictures/Jackie-Chan.png"
)


if input_landmarks is None:
    print("No face found in input")
    exit()


if match_landmarks is None:
    print("No face found in comparison")
    exit()


# ---------------------------------------------
# GEOMETRY
# ---------------------------------------------

input_geometry = FaceGeometry(
    input_landmarks
)

match_geometry = FaceGeometry(
    match_landmarks
)


report = input_geometry.comparison_report(
    match_geometry
)


# ---------------------------------------------
# OUTPUT
# ---------------------------------------------

print()
print("FACIAL GEOMETRY COMPARISON")
print("==========================")

print(
    f"Overall similarity: "
    f"{report['overall'] * 100:.1f}%"
)

print()

for group, score in sorted(
    report["groups"].items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{group:15s}: "
        f"{score * 100:.1f}%"
    )

print()

print(
    "Most similar feature group:",
    report["most_similar"]
)