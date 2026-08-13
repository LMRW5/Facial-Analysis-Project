import cv2
import mediapipe as mp

from pathlib import Path

from server.ML.tools.objects.faceSearcher import FaceSearcher
from server.ML.tools.objects.faceGeometry import FaceGeometry

# ============================================================
# SETTINGS
# ============================================================

INPUT_IMAGE = (
    "tools/processing/stored-faces/face_color.jpg"
)

DATABASE_DIR = Path("lfw")

LANDMARK_MODEL = (
    "models/face_landmarker.task"
)

FAISS_RESULTS = 200
TOP_PEOPLE = 5


# ============================================================
# MEDIAPIPE LANDMARK DETECTOR
# ============================================================

def create_landmarker():

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=LANDMARK_MODEL
        ),

        running_mode=(
            mp.tasks.vision.RunningMode.IMAGE
        ),

        num_faces=1
    )

    return (
        mp.tasks.vision.FaceLandmarker
        .create_from_options(options)
    )


# ============================================================
# GET LANDMARKS FROM IMAGE
# ============================================================

def get_landmarks(
    image_path,
    landmarker
):

    image = cv2.imread(
        str(image_path)
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


    result = landmarker.detect(
        mp_image
    )


    if not result.face_landmarks:
        return None


    return result.face_landmarks[0]


# ============================================================
# SEARCH
# ============================================================

searcher = FaceSearcher()
print("Total faces in database: " + str(searcher.index.ntotal))


results = searcher.search(
    INPUT_IMAGE,
    top_k=FAISS_RESULTS
)


if not results:

    print("No search results.")
    raise SystemExit


# ============================================================
# GROUP BY PERSON
# ============================================================

best_per_person = {}


for result in results:

    person = result["person"]


    if person not in best_per_person:

        best_per_person[person] = result


    elif (
        result["score"]
        >
        best_per_person[person]["score"]
    ):

        best_per_person[person] = result


# Sort people by their best ArcFace similarity

people = sorted(
    best_per_person.values(),
    key=lambda x: x["score"],
    reverse=True
)


people = people[
    :TOP_PEOPLE
]


# ============================================================
# INPUT LANDMARKS
# ============================================================

print()
print("Detecting input facial geometry...")


with create_landmarker() as landmarker:

    input_landmarks = get_landmarks(
        INPUT_IMAGE,
        landmarker
    )


    if input_landmarks is None:

        print(
            "Could not detect landmarks "
            "on input image."
        )

        raise SystemExit


    input_geometry = FaceGeometry(
        input_landmarks
    )


    # ========================================================
    # COMPARE AGAINST TOP PEOPLE
    # ========================================================

    final_results = []


    for result in people:

        person = result["person"]
        file = result["file"]


        reference_path = (
            DATABASE_DIR
            / person
            / file
        )


        print(
            f"Analyzing geometry: {person}"
        )


        reference_landmarks = (
            get_landmarks(
                reference_path,
                landmarker
            )
        )


        if reference_landmarks is None:

            print(
                f"  Could not detect "
                f"landmarks: {reference_path}"
            )

            continue


        reference_geometry = (
            FaceGeometry(
                reference_landmarks
            )
        )


        report = (
            input_geometry
            .comparison_report(
                reference_geometry
            )
        )


        final_results.append(
            {
                "person": person,

                # ArcFace / FAISS
                "arcface_score": (
                    result["score"]
                ),

                # Geometry
                "geometry_score": (
                    report["overall"]
                ),

                "geometry_groups": (
                    report["groups"]
                ),

                "most_similar_feature": (
                    report["most_similar"]
                ),

                "file": file
            }
        )


# ============================================================
# SORT BY GEOMETRY + ARCFACE
# ============================================================

# For now we DO NOT combine the scores.
#
# We keep both independently visible so you can see
# whether ArcFace and geometry agree.
#
# Later we can tune a combined score based on testing.

final_results.sort(
    key=lambda x: x["arcface_score"],
    reverse=True
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("FACE SIMILARITY RESULTS")
print("=" * 60)


for rank, result in enumerate(
    final_results,
    start=1
):

    print()
    print(
        f"{rank}. {result['person']}"
    )

    print(
        f"   ArcFace similarity: "
        f"{result['arcface_score']:.4f}"
    )

    print(
        f"   Geometry similarity: "
        f"{result['geometry_score']:.4f}"
    )

    print(
        f"   Closest geometry: "
        f"{result['most_similar_feature']}"
    )

    print(
        f"   Reference: "
        f"{result['file']}"
    )

    print()
    print("   Feature breakdown:")


    # Highest geometric matches first

    sorted_features = sorted(
        result["geometry_groups"].items(),
        key=lambda x: x[1],
        reverse=True
    )


    for feature, score in sorted_features:

        print(
            f"      {feature:15s} "
            f"{score * 100:.1f}%"
        )