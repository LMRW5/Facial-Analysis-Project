import cv2
import mediapipe as mp
import numpy as np

from pathlib import Path

from tools.objects.faceSearcher import FaceSearcher
from tools.objects.faceGeometry import FaceGeometry


# ============================================================
# SETTINGS
# ============================================================

INPUT_IMAGE = "tools/processing/stored-faces/face_color.jpg"

DATABASE_DIR = Path("Open Famous People Faces")

LANDMARK_MODEL = "models/face_landmarker.task"

FAISS_TOP_K = 10
TOP_PEOPLE = 5


# ============================================================
# CREATE MEDIAPIPE LANDMARKER
# ============================================================

def create_landmarker():

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=LANDMARK_MODEL
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5
    )

    return mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    )


# ============================================================
# GET LANDMARKS
# ============================================================

def get_landmarks(image_path, landmarker):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not load: {image_path}")
        return None

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]


# ============================================================
# GET ALL REFERENCE IMAGES
# ============================================================

def get_person_images(person):

    person_dir = DATABASE_DIR / person

    if not person_dir.exists():
        print(f"Person folder not found: {person_dir}")
        return []

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    return [
        path
        for path in person_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in valid_extensions
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # FAISS SEARCH
    # --------------------------------------------------------

    searcher = FaceSearcher()

    results = searcher.search(
        INPUT_IMAGE,
        top_k=FAISS_TOP_K
    )

    if not results:
        print("No FAISS results found.")
        return

    print()
    print("FAISS candidates:")
    print()

    for i, result in enumerate(results, start=1):

        print(
            f"{i}. "
            f"{result['person']} "
            f"-> {result['score']:.4f}"
        )

    candidate_people = results[:TOP_PEOPLE]

    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    with create_landmarker() as landmarker:

        # ----------------------------------------------------
        # INPUT GEOMETRY
        # ----------------------------------------------------

        input_landmarks = get_landmarks(
            INPUT_IMAGE,
            landmarker
        )

        if input_landmarks is None:
            print("Could not detect geometry on input.")
            return

        input_geometry = FaceGeometry(
            input_landmarks
        )

        final_results = []

        # ----------------------------------------------------
        # ANALYZE EACH PERSON
        # ----------------------------------------------------

        for person_result in candidate_people:

            person = person_result["person"]

            arcface_score = float(
                person_result["score"]
            )

            print()
            print(f"Analyzing {person}...")

            image_paths = get_person_images(
                person
            )

            print(
                f"Reference images found: "
                f"{len(image_paths)}"
            )

            if not image_paths:
                continue

            geometry_reports = []

            # ------------------------------------------------
            # ANALYZE EVERY REFERENCE IMAGE
            # ------------------------------------------------

            for image_path in image_paths:

                landmarks = get_landmarks(
                    image_path,
                    landmarker
                )

                if landmarks is None:
                    continue

                reference_geometry = FaceGeometry(
                    landmarks
                )

                report = (
                    input_geometry.comparison_report(
                        reference_geometry
                    )
                )

                geometry_reports.append(
                    {
                        "file": image_path.name,
                        "path": image_path,
                        "report": report
                    }
                )

            if not geometry_reports:
                print("No usable geometry.")
                continue

            # ------------------------------------------------
            # MEDIAN GEOMETRY SCORE
            # ------------------------------------------------

            overall_scores = [
                item["report"]["overall"]
                for item in geometry_reports
            ]

            geometry_median = float(
                np.median(
                    overall_scores
                )
            )

            # ------------------------------------------------
            # MEDIAN FEATURE-GROUP SCORES
            # ------------------------------------------------

            group_scores = {}

            for group in [
                "Eyes",
                "Nose",
                "Mouth",
                "Face shape",
                "Lower face"
            ]:

                scores = [
                    item["report"]["groups"][group]
                    for item in geometry_reports
                    if group in item["report"]["groups"]
                ]

                if scores:

                    group_scores[group] = float(
                        np.median(scores)
                    )

            # ------------------------------------------------
            # BEST INDIVIDUAL GEOMETRY MATCH
            # ------------------------------------------------

            best_reference = max(
                geometry_reports,
                key=lambda item:
                item["report"]["overall"]
            )

            best_geometry_score = float(
                best_reference["report"]["overall"]
            )

            # ------------------------------------------------
            # MOST SIMILAR FEATURE
            # ------------------------------------------------

            if group_scores:

                most_similar_feature = max(
                    group_scores,
                    key=group_scores.get
                )

            else:

                most_similar_feature = None

            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

            final_results.append(
                {
                    "person": person,

                    "arcface_score":
                        arcface_score,

                    "geometry_median":
                        geometry_median,

                    "geometry_best":
                        best_geometry_score,

                    "geometry_groups":
                        group_scores,

                    "most_similar_feature":
                        most_similar_feature,

                    "best_reference":
                        best_reference["file"],

                    "images_analyzed":
                        len(geometry_reports)
                }
            )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("FACIAL SIMILARITY ANALYSIS")
    print("=" * 70)

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
            f"   Geometry median: "
            f"{result['geometry_median'] * 100:.1f}%"
        )

        print(
            f"   Best geometry match: "
            f"{result['geometry_best'] * 100:.1f}%"
        )

        print(
            f"   Images analyzed: "
            f"{result['images_analyzed']}"
        )

        print(
            f"   Best reference: "
            f"{result['best_reference']}"
        )

        print(
            f"   Most similar feature: "
            f"{result['most_similar_feature']}"
        )

        print()
        print("   Feature similarity:")

        sorted_groups = sorted(
            result["geometry_groups"].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for feature, score in sorted_groups:

            print(
                f"      {feature:<15} "
                f"{score * 100:.1f}%"
            )


if __name__ == "__main__":
    main()