import os
import pickle
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from pathlib import Path

class ModelManager:
    def __init__(self, models_dir: str = "models"):
        """モデル管理クラスの初期化"""
        self.models_dir = models_dir
        self.models_cache: Dict[str, Dict[str, Any]] = {}
        self._ensure_models_directory()
        
    def _ensure_models_directory(self):
        """モデルディレクトリの存在確認"""
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        
    def get_model(self, model_name: str = "default") -> Dict[str, Any]:
        """モデルの取得"""
        if model_name in self.models_cache:
            return self.models_cache[model_name]
            
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise ValueError(f"Model {model_name} not found")
            
        # モデルのロード
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        if not all(key in model_data for key in ['model_r', 'model_g', 'model_b', 'scaler_X', 'scaler_Y']):
            raise ValueError("Invalid model format")
            
        self.models_cache[model_name] = model_data
        return model_data
        
    def list_models(self) -> List[Dict[str, str]]:
        """利用可能なモデル一覧の取得"""
        models = []
        for file in os.listdir(self.models_dir):
            if file.endswith(".pkl"):
                model_name = file[:-4]
                model_path = os.path.join(self.models_dir, file)
                models.append({
                    "name": model_name,
                    "path": model_path,
                    "created_at": datetime.fromtimestamp(
                        os.path.getctime(model_path)
                    ).isoformat()
                })
        return models
        
    def train_model(self, features: np.ndarray, targets: np.ndarray, model_name: str = "custom"):
        """新しいモデルの学習"""
        # スケーラーの初期化と特徴量のスケーリング
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(features)
        Y_scaled = scaler_Y.fit_transform(targets)
        
        # RGB各チャンネルのモデルを学習
        models = []
        for i in range(3):  # R, G, B
            model = SGDRegressor(max_iter=1000, tol=1e-3)
            model.fit(X_scaled, Y_scaled[:, i])
            models.append(model)
        
        # モデルの保存
        model_data = {
            'model_r': models[0],
            'model_g': models[1],
            'model_b': models[2],
            'scaler_X': scaler_X,
            'scaler_Y': scaler_Y
        }
        
        self._save_model(model_data, model_name)
        self.models_cache[model_name] = model_data
        
    def predict(self, model_name: str, features: np.ndarray) -> np.ndarray:
        """モデルを使用して予測"""
        model_data = self.get_model(model_name)
        
        # 特徴量のスケーリング
        X_scaled = model_data['scaler_X'].transform(features)
        
        # RGB各チャンネルの予測
        predictions_scaled = np.column_stack([
            model_data['model_r'].predict(X_scaled),
            model_data['model_g'].predict(X_scaled),
            model_data['model_b'].predict(X_scaled)
        ])
        
        # スケールを元に戻す
        predictions = model_data['scaler_Y'].inverse_transform(predictions_scaled)
        
        return predictions
        
    def _save_model(self, model_data: Dict[str, Any], model_name: str):
        """モデルの保存"""
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f) 