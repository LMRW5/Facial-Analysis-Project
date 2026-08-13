import numpy as np


class FaceGeometry:

    # =========================================================
    # MEDIAPIPE LANDMARK INDICES
    # =========================================================

    LEFT_EYE = [
        33, 133, 160, 159,
        158, 157, 173, 144,
        145, 153
    ]

    RIGHT_EYE = [
        362, 263, 387, 386,
        385, 384, 398, 373,
        374, 380
    ]

    MOUTH_OUTER = [
        61, 146, 91, 181,
        84, 17, 314, 405,
        321, 375, 291
    ]

    MOUTH_INNER = [
        78, 95, 88, 178,
        87, 14, 317, 402,
        318, 324
    ]

    NOSE = [
        1, 2, 98, 327,
        129, 358, 94, 19
    ]

    FACE_OVAL = [
        10, 338, 297, 332,
        284, 251, 389, 356,
        454, 323, 361, 288,
        397, 365, 379, 378,
        400, 377, 152, 148,
        176, 149, 150, 136,
        172, 58, 132, 93,
        234, 127, 162, 21,
        54, 103, 67, 109
    ]


    # =========================================================
    # FEATURE TOLERANCES
    #
    # A relative difference equal to the tolerance gives
    # approximately 0% similarity for that feature.
    #
    # These are starting heuristics, not scientifically
    # calibrated thresholds.
    # =========================================================

    FEATURE_TOLERANCES = {

        "eye_spacing": 0.12,
        "eye_width": 0.15,
        "eye_height": 0.20,
        "eye_aspect_ratio": 0.20,

        "nose_width": 0.18,
        "nose_height": 0.18,
        "nose_to_eye": 0.15,
        "nose_to_mouth": 0.15,

        "mouth_width": 0.15,
        "mouth_height": 0.20,
        "mouth_aspect_ratio": 0.20,
        "inner_mouth_width": 0.18,
        "inner_mouth_height": 0.22,

        "face_aspect_ratio": 0.12,
        "cheek_width_ratio": 0.12,

        "eye_to_chin_ratio": 0.15,
        "mouth_to_chin_ratio": 0.15
    }


    # =========================================================
    # INIT
    # =========================================================

    def __init__(self, landmarks):

        self.landmarks = landmarks

        self.points = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in landmarks
            ],
            dtype=np.float32
        )


    # =========================================================
    # BASIC HELPERS
    # =========================================================

    def get_points(self, indices):

        return self.points[indices]


    def centroid(self, indices):

        return np.mean(
            self.get_points(indices),
            axis=0
        )


    def distance(self, p1, p2):

        return float(
            np.linalg.norm(
                p1 - p2
            )
        )


    # =========================================================
    # FACE BOUNDS
    # =========================================================

    def face_bounds(self):

        points = self.get_points(
            self.FACE_OVAL
        )

        min_x = np.min(
            points[:, 0]
        )

        max_x = np.max(
            points[:, 0]
        )

        min_y = np.min(
            points[:, 1]
        )

        max_y = np.max(
            points[:, 1]
        )

        return (
            float(min_x),
            float(max_x),
            float(min_y),
            float(max_y)
        )


    def face_width(self):

        min_x, max_x, _, _ = (
            self.face_bounds()
        )

        return max_x - min_x


    def face_height(self):

        _, _, min_y, max_y = (
            self.face_bounds()
        )

        return max_y - min_y


    # =========================================================
    # EYE FEATURES
    # =========================================================

    def eye_features(self):

        left = self.get_points(
            self.LEFT_EYE
        )

        right = self.get_points(
            self.RIGHT_EYE
        )

        left_center = np.mean(
            left,
            axis=0
        )

        right_center = np.mean(
            right,
            axis=0
        )

        face_width = self.face_width()
        face_height = self.face_height()

        left_width = (
            np.max(left[:, 0])
            -
            np.min(left[:, 0])
        )

        right_width = (
            np.max(right[:, 0])
            -
            np.min(right[:, 0])
        )

        left_height = (
            np.max(left[:, 1])
            -
            np.min(left[:, 1])
        )

        right_height = (
            np.max(right[:, 1])
            -
            np.min(right[:, 1])
        )

        eye_spacing = self.distance(
            left_center,
            right_center
        )

        average_eye_width = (
            left_width + right_width
        ) / 2

        average_eye_height = (
            left_height + right_height
        ) / 2

        return {

            "eye_spacing":
                eye_spacing / face_width,

            "eye_width":
                average_eye_width / face_width,

            "eye_height":
                average_eye_height / face_height,

            "eye_aspect_ratio":
                average_eye_width /
                max(average_eye_height, 1e-6)
        }


    # =========================================================
    # NOSE FEATURES
    # =========================================================

    def nose_features(self):

        nose = self.get_points(
            self.NOSE
        )

        face_width = self.face_width()
        face_height = self.face_height()

        nose_width = (
            np.max(nose[:, 0])
            -
            np.min(nose[:, 0])
        )

        nose_height = (
            np.max(nose[:, 1])
            -
            np.min(nose[:, 1])
        )

        nose_tip = self.points[1]

        eye_center = (
            self.centroid(
                self.LEFT_EYE
            )
            +
            self.centroid(
                self.RIGHT_EYE
            )
        ) / 2

        mouth_center = self.centroid(
            self.MOUTH_OUTER
        )

        return {

            "nose_width":
                nose_width / face_width,

            "nose_height":
                nose_height / face_height,

            "nose_to_eye":
                self.distance(
                    nose_tip,
                    eye_center
                ) / face_height,

            "nose_to_mouth":
                self.distance(
                    nose_tip,
                    mouth_center
                ) / face_height
        }


    # =========================================================
    # MOUTH FEATURES
    # =========================================================

    def mouth_features(self):

        outer = self.get_points(
            self.MOUTH_OUTER
        )

        inner = self.get_points(
            self.MOUTH_INNER
        )

        face_width = self.face_width()
        face_height = self.face_height()

        outer_width = (
            np.max(outer[:, 0])
            -
            np.min(outer[:, 0])
        )

        outer_height = (
            np.max(outer[:, 1])
            -
            np.min(outer[:, 1])
        )

        inner_width = (
            np.max(inner[:, 0])
            -
            np.min(inner[:, 0])
        )

        inner_height = (
            np.max(inner[:, 1])
            -
            np.min(inner[:, 1])
        )

        return {

            "mouth_width":
                outer_width / face_width,

            "mouth_height":
                outer_height / face_height,

            "mouth_aspect_ratio":
                outer_width /
                max(outer_height, 1e-6),

            "inner_mouth_width":
                inner_width / face_width,

            "inner_mouth_height":
                inner_height / face_height
        }


    # =========================================================
    # FACE SHAPE FEATURES
    # =========================================================

    def face_features(self):

        face_width = self.face_width()
        face_height = self.face_height()

        left_cheek = self.points[234]
        right_cheek = self.points[454]

        chin = self.points[152]

        eye_center = (
            self.centroid(
                self.LEFT_EYE
            )
            +
            self.centroid(
                self.RIGHT_EYE
            )
        ) / 2

        mouth_center = self.centroid(
            self.MOUTH_OUTER
        )

        cheek_width = self.distance(
            left_cheek,
            right_cheek
        )

        eye_to_chin = self.distance(
            eye_center,
            chin
        )

        mouth_to_chin = self.distance(
            mouth_center,
            chin
        )

        return {

            "face_aspect_ratio":
                face_width / face_height,

            "cheek_width_ratio":
                cheek_width / face_width,

            "eye_to_chin_ratio":
                eye_to_chin / face_height,

            "mouth_to_chin_ratio":
                mouth_to_chin / face_height
        }


    # =========================================================
    # ALL FEATURES
    # =========================================================

    def get_features(self):

        features = {}

        features.update(
            self.eye_features()
        )

        features.update(
            self.nose_features()
        )

        features.update(
            self.mouth_features()
        )

        features.update(
            self.face_features()
        )

        return features


    # =========================================================
    # FEATURE GROUPS
    # =========================================================

    FEATURE_GROUPS = {

        "Eyes": [
            "eye_spacing",
            "eye_width",
            "eye_height",
            "eye_aspect_ratio"
        ],

        "Nose": [
            "nose_width",
            "nose_height",
            "nose_to_eye",
            "nose_to_mouth"
        ],

        "Mouth": [
            "mouth_width",
            "mouth_height",
            "mouth_aspect_ratio",
            "inner_mouth_width",
            "inner_mouth_height"
        ],

        "Face shape": [
            "face_aspect_ratio",
            "cheek_width_ratio"
        ],

        "Lower face": [
            "eye_to_chin_ratio",
            "mouth_to_chin_ratio"
        ]
    }


    # =========================================================
    # FEATURE SIMILARITY
    # =========================================================

    @staticmethod
    def feature_similarity(
        value_a,
        value_b,
        tolerance
    ):
        """
        Converts relative difference into a 0-1 score.

        Same value:
            1.0

        Difference >= tolerance:
            0.0
        """

        denominator = max(
            abs(value_a),
            abs(value_b),
            1e-8
        )

        relative_difference = (
            abs(value_a - value_b)
            / denominator
        )

        similarity = (
            1.0
            -
            relative_difference / tolerance
        )

        return float(
            np.clip(
                similarity,
                0.0,
                1.0
            )
        )


    # =========================================================
    # COMPARE FEATURES
    # =========================================================

    def compare_features(self, other):

        features_a = (
            self.get_features()
        )

        features_b = (
            other.get_features()
        )

        similarities = {}

        for feature in features_a:

            tolerance = (
                self.FEATURE_TOLERANCES.get(
                    feature,
                    0.15
                )
            )

            similarities[feature] = (
                self.feature_similarity(
                    features_a[feature],
                    features_b[feature],
                    tolerance
                )
            )

        return similarities


    # =========================================================
    # GROUP SIMILARITY
    # =========================================================

    def compare_groups(self, other):

        feature_similarities = (
            self.compare_features(other)
        )

        group_scores = {}

        for group, features in (
            self.FEATURE_GROUPS.items()
        ):

            scores = [
                feature_similarities[feature]
                for feature in features
                if feature in feature_similarities
            ]

            if scores:

                group_scores[group] = (
                    float(
                        np.mean(scores)
                    )
                )

        return group_scores


    # =========================================================
    # OVERALL GEOMETRY SCORE
    # =========================================================

    def overall_similarity(self, other):

        group_scores = (
            self.compare_groups(other)
        )

        if not group_scores:
            return 0.0

        return float(
            np.mean(
                list(
                    group_scores.values()
                )
            )
        )


    # =========================================================
    # MOST SIMILAR GROUP
    # =========================================================

    def most_similar_group(self, other):

        group_scores = (
            self.compare_groups(other)
        )

        if not group_scores:
            return None

        return max(
            group_scores,
            key=group_scores.get
        )


    # =========================================================
    # FULL REPORT
    # =========================================================

    def comparison_report(self, other):

        feature_scores = (
            self.compare_features(other)
        )

        group_scores = (
            self.compare_groups(other)
        )

        overall = (
            self.overall_similarity(other)
        )

        most_similar = (
            self.most_similar_group(other)
        )

        return {
            "overall": overall,
            "groups": group_scores,
            "features": feature_scores,
            "most_similar": most_similar
        }