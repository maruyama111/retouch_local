from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import threading
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
import asyncio
from typing import Optional

server = None  # uvicorn.Server のインスタンスを格納する変数
app = FastAPI(title="Retouch API")

# CORS設定などは省略可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# アプリケーションのベースパスを取得
if getattr(sys, 'frozen', False):
    # PyInstallerでビルドされた場合
    base_path = sys._MEIPASS
else:
    # 通常の実行の場合
    base_path = os.path.dirname(os.path.abspath(__file__))

# デバッグ追加
print("Debug: sys._MEIPASS ->", getattr(sys, '_MEIPASS', 'NOT FROZEN'))
print("Debug: base_path ->", base_path)

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

# デバッグ追加
if getattr(sys, 'frozen', False):
    print("[Debug] Running in frozen mode (PyInstaller)")
    print("[Debug] sys._MEIPASS =", sys._MEIPASS)
else:
    print("[Debug] Running in normal mode")
print("[Debug] base_path =", base_path)

resource_dir = os.environ.get("MEDIAPIPE_RESOURCE_DIR", "")
print(f"[Debug] resource_dir = {resource_dir}")

tflite_path = os.path.join(resource_dir, 'modules', 'pose_landmark', 'pose_landmark_heavy.tflite')
print(f"[Debug] Checking if TFLite file exists: {tflite_path} -> {os.path.exists(tflite_path)}")


def process_image_with_mediapipe(image_path):
    """MediaPipeを使用して画像を処理"""
    try:
        # MediaPipeの設定を確認
        resource_dir = os.environ.get("MEDIAPIPE_RESOURCE_DIR", "")
        print(f"MediaPipe resource directory: {resource_dir}")
        print(f"Looking for model file: {os.path.join(resource_dir, 'modules/pose_landmark/pose_landmark_heavy.tflite')}")
        
        # デバッグ追加
        tflite_path = os.path.join(resource_dir, "modules", "pose_landmark", "pose_landmark_heavy.tflite")
        print(f"[Debug] TFLite path: {tflite_path}, exists: {os.path.exists(tflite_path)}")
        
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
    # model_nameをユーザー指定で受け取る。指定がなければ trained_4factor_model3 をデフォルトとする
    model_name: str = Form("trained_4factor_model3")
):
    """
    単一画像のレタッチ処理。呼び出し時に FormData で model_name を指定すると、	
    そのモデルを使って画像を補正する
    """
    temp_path = None
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
        left_rgb, right_rgb = [], []
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                left_cheek, right_cheek = image_processor.get_cheek_landmarks(
                    results.multi_face_landmarks[0], h, w
                )
                
        left_rgb = image_processor.extract_cheek_region(image, left_cheek)
        right_rgb = image_processor.extract_cheek_region(image, right_cheek)
        cheek_rgb = np.vstack([left_rgb, right_rgb])if len(left_rgb) > 0 else np.array([])
        
        # 特徴量の計算
        body_features = image_processor.calculate_features(body_rgb)
        image_features = image_processor.calculate_features(image.reshape(-1, 3))

        if cheek_rgb.size > 0:
            cheek_features = image_processor.calculate_features(cheek_rgb)
        else:
            cheek_features = {"brightness": 0, "saturation": 0, "contrast": 0}
        
        # 特徴量をモデル入力用に整形
        features = np.array([[
            image_features['brightness'],
            cheek_features['brightness'],
            body_features['saturation'],
            body_features['contrast']
        ]])
        
        # モデルによる予測
        coefficients = model_manager.predict(model_name, features)
        
        # デバッグ出力
        print(f"Use model: {model_name}, coefficients: {coefficients}")
        
        # 画像の補正
        adjusted_image = image_processor.apply_adjustment(image, coefficients[0])
        
        # 結果の保存
        output_path = temp_path.replace('.', '_retouched.')
        cv2.imwrite(output_path, adjusted_image)
        
        return {"status": "success", "result_path": output_path, "model_name": model_name}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        # 一時ファイルの削除
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/retouch/batch")
async def retouch_batch_images(
    files: List[UploadFile] = File(...),
    # まとめて指定する場合も同様に model_name をフォームで受け取る
    model_name: str = Form("trained_4factor_model3")
):
    """
    複数画像の一括レタッチ処理。model_name をForm で渡す。 
    指定がなければ "trained_4factor_model3" が使われる。
    """
    results = []
    for file in files:
        # retouch_single_image に model_name を渡す
        result = await retouch_single_image(file, model_name=model_name)
        if result["status"] == "success":
            results.append({
                "filename": file.filename,
                "result_path": result["result_path"],
                "used_model": model_name
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
    new_model_name: str = Form("custom")
):
    
    """
    1) Before画像から特徴量(4因子)を抽出
    2) After画像とBefore画像のR/G/B平均色の差分を targets として作成
    3) model_manager.train_model(features, targets, model_name) を呼び出し、R/G/B用の回帰モデルを学習
    """
    
    temp_before = None
    temp_after = None
    
    try:
        # 1) 一時ファイルへ保存
        temp_before = save_upload_file_temp(before_image)
        temp_after = save_upload_file_temp(after_image)
        
        before_img = cv2.imread(temp_before)
        after_img = cv2.imread(temp_after)
        
        if before_img is None:
            raise ValueError("Failed to load before_image")
        if after_img is None:
            raise ValueError("Failed to load after_image")
        
        # 2) Before画像とAfter画像、それぞれで「ボディ領域」「頬領域」を検出
        #    -- before画像で 4-factor (features) を作成
        #    -- after画像で R/G/B の平均色を求め、beforeとの比率を学習ターゲットに
        # ----------------------------------------------------------
        
        # (a) before画像のランドマーク
        before_landmarks = image_processor.get_landmarks(before_img)
        if not before_landmarks:
            raise ValueError("No landmarks detected in before_image")
        
        before_body_coords = image_processor.extract_body_region(before_img, before_landmarks)
        before_body_rgb = image_processor.extract_body_features(before_img, before_body_coords)
        
        # 頬領域
        h_b, w_b, _ = before_img.shape
        before_cheek_rgb = np.array([])
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            results_b = face_mesh.process(cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB))
            if results_b.multi_face_landmarks:
                left_cheek_b, right_cheek_b = image_processor.get_cheek_landmarks(
                    results_b.multi_face_landmarks[0], h_b, w_b
                )
                left_rgb_b = image_processor.extract_cheek_region(before_img, left_cheek_b)
                right_rgb_b = image_processor.extract_cheek_region(before_img, right_cheek_b)
                if len(left_rgb_b) > 0:
                    before_cheek_rgb = np.vstack([left_rgb_b, right_rgb_b])

        before_body_features = image_processor.calculate_features(before_body_rgb)
        before_image_features = image_processor.calculate_features(before_img.reshape(-1, 3))
        if before_cheek_rgb.size > 0:
            before_cheek_features = image_processor.calculate_features(before_cheek_rgb)
        else:
            before_cheek_features = {"brightness": 0, "saturation": 0, "contrast": 0}
        
        # --- Beforeの4-factor特徴量 (Brightness, CheekBrightness, Saturation, Contrast)
        before_features_4 = np.array([[
            before_image_features['brightness'],
            before_cheek_features['brightness'],
            before_body_features['saturation'],
            before_body_features['contrast']
        ]], dtype=np.float32)
        
        # --- (B) 画像全体のRGB平均値の変化「率」を計算 ---
        # OpenCV では画素順が (B, G, R) なので注意
        # ここでは R, G, B の「全画素平均」の変化率 = after / before
        before_mean_bgr = before_img.reshape(-1, 3).mean(axis=0)  # [B, G, R]
        after_mean_bgr = after_img.reshape(-1, 3).mean(axis=0)    # [B, G, R]

        # 各チャネルごとに ratio = after / (before + 1e-8)
        # model_manager.train_model では R/G/B の順を想定しているので順番を合わせる
        #  -> i=0 => R, i=1 => G, i=2 => B
        ratio_r = after_mean_bgr[2] / (before_mean_bgr[2] + 1e-8)
        ratio_g = after_mean_bgr[1] / (before_mean_bgr[1] + 1e-8)
        ratio_b = after_mean_bgr[0] / (before_mean_bgr[0] + 1e-8)

        targets = np.array([[ratio_r, ratio_g, ratio_b]], dtype=np.float32)

        # --- (C) model_manager.train_model(features, targets, model_name) で学習 ---
        model_manager.train_model(before_features_4, targets, model_name=new_model_name)
        
        return {
            "status": "success",
            "message": f"Model '{new_model_name}' trained successfully (ratio-based)."
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        for p in [temp_before, temp_after]:
            if p and os.path.exists(p):
                os.remove(p)

@app.get("/shutdown")
def shutdown_server():
    """
    サーバーを終了するためのエンドポイント。
    Electron 側からこのエンドポイントを呼び出してからアプリを閉じるようにすると、
    Uvicorn サーバーが正常終了してポートを解放する。
    """
    def stopper():
        # 少し待ってからサーバーを止める
        time.sleep(0.5)
        if server is not None:
            server.should_exit = True

    threading.Thread(target=stopper).start()
    return {"message": "Shutting down the server..."}


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

def main():
    global server
    import uvicorn
    
    port = int(os.environ.get("PORT", 8080))
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    main()
