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