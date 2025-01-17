import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Any
from .model_architecture import RetouchNet

class RetouchProcessor:
    def __init__(self):
        """レタッチ処理クラスの初期化"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def process_single(self, input_path: str, model: RetouchNet) -> str:
        """単一画像のレタッチ処理"""
        # 画像の読み込み
        image = self._load_image(input_path)
        
        # 画像の前処理
        processed_image = self._preprocess_image(image)
        
        # モデルをデバイスに移動
        model = model.to(self.device)
        processed_image = processed_image.to(self.device)
        
        # モデルによる処理
        with torch.no_grad():
            result = self._apply_model(processed_image, model)
            
        # 後処理
        final_image = self._postprocess_image(result.cpu())
        
        # 結果の保存
        output_path = self._generate_output_path(input_path)
        self._save_image(final_image, output_path)
        
        return output_path
        
    def _load_image(self, path: str) -> np.ndarray:
        """画像の読み込み"""
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
            
        # サイズの標準化（必要に応じて）
        image = cv2.resize(image, (256, 256))
        return image
        
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """画像の前処理"""
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # To tensor [B, C, H, W]
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
        
        return tensor
        
    def _apply_model(self, image: torch.Tensor, model: RetouchNet) -> torch.Tensor:
        """モデルの適用"""
        model.eval()  # 評価モード
        return model(image)
        
    def _postprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        """画像の後処理"""
        # Remove batch dimension and convert to numpy [H, W, C]
        image = tensor.squeeze(0).permute(1, 2, 0).numpy()
        
        # Clip to [0, 1] range
        image = np.clip(image, 0, 1)
        
        # Scale to [0, 255]
        image = (image * 255.0).astype(np.uint8)
        
        # RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
        
    def _generate_output_path(self, input_path: str) -> str:
        """出力パスの生成"""
        path = Path(input_path)
        return str(path.parent / f"{path.stem}_retouched{path.suffix}")
        
    def _save_image(self, image: np.ndarray, path: str):
        """画像の保存"""
        if not cv2.imwrite(path, image):
            raise ValueError(f"Failed to save image: {path}") 