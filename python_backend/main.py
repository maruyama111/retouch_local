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
import sys
import traceback
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
import socket
import time

app = FastAPI(title="Retouch API")

# アプリケーションのベースパスを取得
if getattr(sys, 'frozen', False):
    # PyInstallerでビルドされた場合
    base_path = sys._MEIPASS
else:
    # 通常の実行の場合
    base_path = os.path.dirname(os.path.abspath(__file__))

# MediaPipeの設定を更新
os.environ["MEDIAPIPE_RESOURCE_DIR"] = os.path.join(base_path, "mediapipe")

# モデルディレクトリの設定
if getattr(sys, 'frozen', False):
    # 実行ファイルとして実行されている場合
    BASE_DIR = os.path.dirname(sys.executable)
    MODELS_DIR = os.path.join(BASE_DIR, "models")
else:
    # 開発環境での実行の場合
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")

print(f"Using models directory: {MODELS_DIR}")  # デバッグ用ログ

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

def process_image_with_mediapipe(image_path):
    """MediaPipeを使用して画像を処理"""
    try:
        # MediaPipeの設定を確認
        resource_dir = os.environ.get("MEDIAPIPE_RESOURCE_DIR", "")
        print(f"MediaPipe resource directory: {resource_dir}")
        print(f"Looking for model file: {os.path.join(resource_dir, 'modules/pose_landmark/pose_landmark_heavy.tflite')}")
        
        # 画像を読み込み
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # MediaPipeの処理を実行
        with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        ) as pose:
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        if not results.pose_landmarks:
            raise ValueError("No pose landmarks detected in the image")
            
        return results.pose_landmarks
        
    except Exception as e:
        print(f"Error in process_image_with_mediapipe: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        raise Exception(f"Failed to process image with mediapipe: {str(e)}")

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
    before_image: UploadFile = File(..., description="レタッチ前の画像"),
    after_image: UploadFile = File(..., description="レタッチ後の画像"),
    base_model: str = Form(default="default", description="ベースモデル名"),
    new_model_name: str = Form(None, description="新しいモデル名")
):
    """Train the model with a new image pair"""
    before_path = None
    after_path = None
    
    try:
        print("Received training request:")
        print(f"- Base model: {base_model}")
        print(f"- New model name: {new_model_name}")
        print(f"- Before image: {before_image.filename}")
        print(f"- After image: {after_image.filename}")
        print(f"- Models directory: {MODELS_DIR}")

        # 一時ファイルを作成
        before_path = tempfile.mktemp(suffix='.jpg')
        after_path = tempfile.mktemp(suffix='.jpg')
        
        # ファイルを保存
        with open(before_path, 'wb') as f:
            content = await before_image.read()
            f.write(content)
        with open(after_path, 'wb') as f:
            content = await after_image.read()
            f.write(content)
            
        print("Temporary files created:")
        print(f"- Before image path: {before_path}")
        print(f"- After image path: {after_path}")
        print("- Files exist check:")
        print(f"  - Before image exists: {os.path.exists(before_path)}")
        print(f"  - After image exists: {os.path.exists(after_path)}")

        # 画像を処理
        before_landmarks = process_image_with_mediapipe(before_path)
        after_landmarks = process_image_with_mediapipe(after_path)

        # ランドマークを特徴量に変換
        before_features = np.array([[landmark.x, landmark.y, landmark.z] for landmark in before_landmarks.landmark]).flatten()
        after_features = np.array([[landmark.x, landmark.y, landmark.z] for landmark in after_landmarks.landmark]).flatten()

        # 変換比率を計算
        ratio = after_features / (before_features + 1e-8)
        targets = np.array([[ratio.mean()]])

        # モデルを保存
        if new_model_name:
            model_name = new_model_name
        else:
            model_name = f"{base_model}_extended_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        
        # モデルデータを保存
        model_data = {
            'ratio': ratio
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        return {
            "status": "success",
            "message": "Model training completed",
            "new_model": model_name
        }

    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "message": f"画像ペアの学習に失敗しました: {str(e)}"
        }

    finally:
        # 一時ファイルを削除
        for temp_path in [before_path, after_path]:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print(f"Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    print(f"Error cleaning up temporary file {temp_path}: {str(e)}")

def find_available_port(start_port, max_attempts=100):
    """利用可能なポートを見つける"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                s.close()  # 明示的にソケットを閉じる
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts - 1}")

if __name__ == "__main__":
    # 環境変数からポート番号を取得（デフォルトは8000）
    initial_port = int(os.environ.get('PORT', 8000))
    
    # 利用可能なポートを見つける
    try:
        port = find_available_port(initial_port)
        print(f"Selected port: {port}")
        
        # FastAPIアプリケーションの起動
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=port,
            log_level="info"
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1) 