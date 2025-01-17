import cv2
import numpy as np
from PIL import Image, ImageDraw
import mediapipe as mp
from typing import Dict, List, Tuple, Any
import os

class ImageProcessor:
    def __init__(self):
        """画像処理クラスの初期化"""
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        
    def get_landmarks(self, image: np.ndarray) -> Dict[str, Any]:
        """Mediapipeを使用してポーズと顔のランドマークを取得"""
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
             self.mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:

            pose_results = pose.process(image_rgb)
            face_results = face_mesh.process(image_rgb)

            if pose_results.pose_landmarks and face_results.multi_face_landmarks:
                pose_landmarks = pose_results.pose_landmarks.landmark
                face_landmarks = face_results.multi_face_landmarks[0].landmark

                def get_face_point(index):
                    landmark = face_landmarks[index]
                    return int(landmark.x * w), int(landmark.y * h)

                return {
                    'pose': pose_landmarks,
                    'face': {
                        'right_forehead': get_face_point(103),
                        'left_forehead': get_face_point(332),
                        'right_cheek': get_face_point(93),
                        'left_cheek': get_face_point(352),
                        'right_jaw': get_face_point(172),
                        'left_jaw': get_face_point(397)
                    }
                }
        return None

    def get_cheek_landmarks(self, face_landmarks: Any, h: int, w: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """頬のランドマークポイントを取得"""
        left_cheek = [
            (int(face_landmarks.landmark[234].x * w), int(face_landmarks.landmark[234].y * h)),
            (int(face_landmarks.landmark[138].x * w), int(face_landmarks.landmark[138].y * h)),
            (int(face_landmarks.landmark[185].x * w), int(face_landmarks.landmark[185].y * h)),
            (int(face_landmarks.landmark[232].x * w), int(face_landmarks.landmark[232].y * h))
        ]

        right_cheek = [
            (int(face_landmarks.landmark[454].x * w), int(face_landmarks.landmark[454].y * h)),
            (int(face_landmarks.landmark[367].x * w), int(face_landmarks.landmark[367].y * h)),
            (int(face_landmarks.landmark[409].x * w), int(face_landmarks.landmark[409].y * h)),
            (int(face_landmarks.landmark[452].x * w), int(face_landmarks.landmark[452].y * h))
        ]

        return left_cheek, right_cheek

    def extract_cheek_region(self, image: np.ndarray, cheek_coords: List[Tuple[int, int]]) -> np.ndarray:
        """頬領域のマスクを作成し、該当部分のRGB値を抽出"""
        pil_image = Image.fromarray(image)
        mask = Image.new('L', pil_image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(cheek_coords, outline=1, fill=1)
        mask = np.array(mask)
        cheek_pixels = np.array(pil_image)[:, :, :3]
        cheek_rgb = cheek_pixels[mask == 1]
        return cheek_rgb

    def extract_body_region(self, image: np.ndarray, pose_landmarks: Dict[str, Any]) -> np.ndarray:
        """体の領域を抽出"""
        h, w, _ = image.shape
        pose_points = pose_landmarks['pose']
        face_points = pose_landmarks['face']

        body_points = [
            face_points['right_forehead'],
            face_points['right_cheek'],
            face_points['right_jaw'],
            (int(pose_points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
             int(pose_points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)),
            (int(pose_points[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w),
             int(pose_points[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h)),
            (int(pose_points[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w),
             int(pose_points[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h)),
            (int(pose_points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
             int(pose_points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)),
            face_points['left_jaw'],
            face_points['left_cheek'],
            face_points['left_forehead']
        ]

        return np.array(body_points, np.int32)

    def extract_body_features(self, image: np.ndarray, body_coords: np.ndarray) -> np.ndarray:
        """体領域の特性を抽出"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        body_coords = [(int(x), int(y)) for x, y in body_coords]

        mask = Image.new('L', pil_image.size, 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.polygon(body_coords, outline=1, fill=1)

        mask = np.array(mask)
        body_pixels = np.array(pil_image)[:, :, :3]
        body_rgb = body_pixels[mask == 1]
        return body_rgb

    def calculate_features(self, rgb_values: np.ndarray) -> Dict[str, float]:
        """RGB値から特徴量を計算"""
        # 明るさ
        brightness = rgb_values.mean()

        # コントラスト
        luminance = 0.2126 * rgb_values[:, 0] + 0.7152 * rgb_values[:, 1] + 0.0722 * rgb_values[:, 2]
        contrast = luminance.max() - luminance.min()

        # 彩度
        max_rgb = rgb_values.max(axis=1)
        min_rgb = rgb_values.min(axis=1)
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-5)
        saturation = saturation.mean()

        return {
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation
        }

    def apply_adjustment(self, image: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        """画像に補正係数を適用"""
        return np.clip(image * coefficients, 0, 255).astype(np.uint8) 