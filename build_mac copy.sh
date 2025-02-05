#!/bin/bash

# Mac用ビルドスクリプト

# エラーが発生したら停止
set -e

echo "Installing Python dependencies..."
cd python_backend
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
pip3 install pyinstaller

# バックエンドのビルド
echo "Building Python backend..."
pyinstaller --name retouch_backend --onefile \
    --hidden-import=mediapipe \
    --hidden-import=cv2 \
    --hidden-import=matplotlib \
    --hidden-import=matplotlib.backends.backend_tkagg \
    --collect-data mediapipe \
    --collect-all mediapipe \
    --add-data "models:models" \
    --noconfirm \
    --clean main.py

# Electronアプリのビルド
echo "Building Electron app..."
cd ../electron_app
npm install

# Electronアプリのビルドとバックエンドの統合
npm run build:mac
mkdir -p "dist/mac/Retouch App.app/Contents/Resources/backend"
cp ../python_backend/dist/retouch_backend "dist/mac/Retouch App.app/Contents/Resources/backend/"

echo "Build completed! Check the electron_app/dist directory for the output." 




# Mac用ビルドスクリプト例 (PowerShell)

Write-Host "Building Electron app for Mac..."
cd electron_app
npm install

# Mac用のビルドタスク（package.jsonのscriptsに 'build:mac' 等のコマンドを用意しておく）
npm run build:mac

# Macビルド出力先 (electron-builderの設定によって異なる場合あり)
# 例：dist/mac/xxxxx.app
# ここでは単純に「dist/mac」というフォルダに出力される想定

# バックエンド配置用のフォルダ作成
mkdir -Force dist/mac/backend

# PyInstallerで作成されたMac用実行ファイル（例：retouch_backend）をコピー
Copy-Item ../python_backend/dist/retouch_backend -Destination dist/mac/backend/

# モデル等の必要ファイルを再帰的にコピー
Copy-Item -Recurse ../python_backend/models -Destination dist/mac/backend/

# 実行権限付与が必要な場合は以下を追加
# chmod +x dist/mac/backend/retouch_backend

Write-Host "Build for Mac completed! Check the electron_app/dist directory for the output."
