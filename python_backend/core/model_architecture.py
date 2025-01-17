import torch
import torch.nn as nn

class RetouchNet(nn.Module):
    """画像レタッチ用のニューラルネットワーク"""
    def __init__(self):
        super(RetouchNet, self).__init__()
        
        # エンコーダー部分
        self.encoder = nn.Sequential(
            self._conv_block(3, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512),
        )
        
        # デコーダー部分
        self.decoder = nn.Sequential(
            self._upconv_block(512, 256),
            self._upconv_block(256, 128),
            self._upconv_block(128, 64),
            self._final_block(64, 3),
        )
        
        # 残差接続用の1x1畳み込み
        self.residual = nn.Conv2d(3, 3, kernel_size=1, padding=0)
        
    def forward(self, x):
        # 入力を保存（残差接続用）
        identity = self.residual(x)
        
        # エンコーダーを通す
        features = self.encoder(x)
        
        # デコーダーを通す
        out = self.decoder(features)
        
        # 残差接続を加算
        return out + identity
        
    def _conv_block(self, in_channels, out_channels):
        """畳み込みブロック"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
    def _upconv_block(self, in_channels, out_channels):
        """アップサンプリングブロック"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def _final_block(self, in_channels, out_channels):
        """最終出力ブロック"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # -1から1の範囲に正規化
        ) 