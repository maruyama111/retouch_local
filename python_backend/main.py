from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import os
import cv2
import numpy as np
from core.image_processor import ImageProcessor
from core.model_manager import ModelManager
import tempfile
import shutil
import mediapipe as mp
from datetime import datetime

app = FastAPI(title="Retouch API")

# モデルディレクトリの設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 画像処理とモデル管理のインスタンス化
image_processor = ImageProcessor()
model_manager = ModelManager(models_dir=MODELS_DIR)

def save_upload_file_temp(upload_file: UploadFile) -> str:
    """アップロードされたファイルを一時ディレクトリに保存"""
    try:
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(upload_file.file, temp_file)
            return temp_file.name
    finally:
        upload_file.file.close()

@app.post("/api/retouch/single")
async def retouch_single_image(
    file: UploadFile = File(...),
    model_name: str = "trained_4factor_model3"
):
    """単一画像のレタッチ処理"""
    try:
        # 画像の一時保存
        temp_path = save_upload_file_temp(file)
        image = cv2.imread(temp_path)
        
        if image is None:
            raise ValueError("Failed to load image")
            
        # ランドマークの検出
        landmarks = image_processor.get_landmarks(image)
        if not landmarks:
            raise ValueError("No landmarks detected")
            
        # 体の領域を抽出
        body_coords = image_processor.extract_body_region(image, landmarks)
        body_rgb = image_processor.extract_body_features(image, body_coords)
        
        # 頬の領域を抽出
        h, w, _ = image.shape
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                left_cheek, right_cheek = image_processor.get_cheek_landmarks(
                    results.multi_face_landmarks[0], h, w
                )
                
        left_rgb = image_processor.extract_cheek_region(image, left_cheek)
        right_rgb = image_processor.extract_cheek_region(image, right_cheek)
        cheek_rgb = np.vstack([left_rgb, right_rgb])
        
        # 特徴量の計算
        body_features = image_processor.calculate_features(body_rgb)
        cheek_features = image_processor.calculate_features(cheek_rgb)
        image_features = image_processor.calculate_features(image.reshape(-1, 3))
        
        # 特徴量をモデル入力用に整形
        features = np.array([[
            image_features['brightness'],
            cheek_features['brightness'],
            body_features['saturation'],
            body_features['contrast']
        ]])
        
        # モデルによる予測
        coefficients = model_manager.predict(model_name, features)
        
        # 画像の補正
        adjusted_image = image_processor.apply_adjustment(image, coefficients[0])
        
        # 結果の保存
        output_path = temp_path.replace('.', '_retouched.')
        cv2.imwrite(output_path, adjusted_image)
        
        return {"status": "success", "result_path": output_path}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        # 一時ファイルの削除
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/retouch/batch")
async def retouch_batch_images(
    files: List[UploadFile] = File(...),
    model_name: str = "trained_4factor_model3"
):
    """複数画像の一括レタッチ処理"""
    results = []
    for file in files:
        result = await retouch_single_image(file, model_name)
        if result["status"] == "success":
            results.append({
                "filename": file.filename,
                "result_path": result["result_path"]
            })
    return {"status": "success", "results": results}

@app.get("/api/models")
async def list_models():
    """利用可能なモデル一覧の取得"""
    try:
        models = model_manager.list_models()
        return {"status": "success", "models": models}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/train")
async def train_model(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
    base_model: str = Form("trained_4factor_model3"),
    new_model_name: str = Form(None)
):
    """モデルの追加学習"""
    try:
        print(f"Received new_model_name: {new_model_name}")  # デバッグ用ログ
        
        # 画像の一時保存
        before_path = save_upload_file_temp(before_image)
        after_path = save_upload_file_temp(after_image)

        # 画像の読み込み
        before_img = cv2.imread(before_path)
        after_img = cv2.imread(after_path)

        if before_img is None or after_img is None:
            raise ValueError("Failed to load images")

        # 特徴量の抽出（before画像）
        before_landmarks = image_processor.get_landmarks(before_img)
        if not before_landmarks:
            raise ValueError("No landmarks detected in before image")

        before_body_coords = image_processor.extract_body_region(before_img, before_landmarks)
        before_body_rgb = image_processor.extract_body_features(before_img, before_body_coords)

        h, w, _ = before_img.shape
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                before_left_cheek, before_right_cheek = image_processor.get_cheek_landmarks(
                    results.multi_face_landmarks[0], h, w
                )

        before_left_rgb = image_processor.extract_cheek_region(before_img, before_left_cheek)
        before_right_rgb = image_processor.extract_cheek_region(before_img, before_right_cheek)
        before_cheek_rgb = np.vstack([before_left_rgb, before_right_rgb])

        # 特徴量の計算（before画像）
        before_body_features = image_processor.calculate_features(before_body_rgb)
        before_cheek_features = image_processor.calculate_features(before_cheek_rgb)
        before_image_features = image_processor.calculate_features(before_img.reshape(-1, 3))

        # 入力特徴量の作成
        features = np.array([[
            before_image_features['brightness'],
            before_cheek_features['brightness'],
            before_body_features['saturation'],
            before_body_features['contrast']
        ]])

        # 目標値の計算（after画像とbefore画像の比率）
        ratio = after_img.astype(np.float32) / (before_img.astype(np.float32) + 1e-8)
        targets = np.array([
            [ratio[:, :, 0].mean(), ratio[:, :, 1].mean(), ratio[:, :, 2].mean()]
        ])

        # モデルの学習
        if new_model_name and new_model_name.strip():
            model_name = new_model_name.strip()
            print(f"Using custom model name: {model_name}")  # デバッグ用ログ
        else:
            model_name = f"{base_model}_extended_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"Using default model name: {model_name}")  # デバッグ用ログ

        model_manager.train_model(features, targets, model_name)

        return {
            "status": "success",
            "message": "Model training completed",
            "new_model": model_name
        }

    except Exception as e:
        print(f"Error in train_model: {str(e)}")  # デバッグ用ログ
        return {"status": "error", "message": str(e)}
    finally:
        # 一時ファイルの削除
        if 'before_path' in locals() and os.path.exists(before_path):
            os.remove(before_path)
        if 'after_path' in locals() and os.path.exists(after_path):
            os.remove(after_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 