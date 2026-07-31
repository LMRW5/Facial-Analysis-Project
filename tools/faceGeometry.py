import numpy as np


class FaceGeometry:


    def get_ratios(self, kps, bbox):

        left_eye = kps[0]
        right_eye = kps[1]
        nose = kps[2]
        mouth_left = kps[3]
        mouth_right = kps[4]


        x1, y1, x2, y2 = bbox


        face_width = x2 - x1
        face_height = y2 - y1


        if face_width == 0 or face_height == 0:
            return None


        eye_distance = np.linalg.norm(
            right_eye - left_eye
        )


        mouth_width = np.linalg.norm(
            mouth_right - mouth_left
        )


        nose_eye_distance = np.linalg.norm(
            nose - ((left_eye + right_eye) / 2)
        )


        return np.array([

            face_width / face_height,

            eye_distance / face_width,

            mouth_width / face_width,

            nose_eye_distance / face_height

        ], dtype=np.float32)