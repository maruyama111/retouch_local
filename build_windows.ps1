# Windows用ビルドスクリプト

# Python環境のセットアップ
Write-Host "Installing Python dependencies..."
cd python_backend
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

# バックエンドのビルド
Write-Host "Building Python backend..."
#python -m pyinstaller retouch_backend.spec
pyinstaller --onefile main.py --name retouch_backend

# Electronアプリのビルド
Write-Host "Building Electron app..."
cd ../electron_app
npm install

# Electronアプリのビルドとバックエンドの統合
npm run build:win
mkdir -Force dist/win-unpacked/backend
Copy-Item ../python_backend/dist/retouch_backend.exe -Destination dist/win-unpacked/backend/
Copy-Item -Recurse ../python_backend/models -Destination dist/win-unpacked/backend/

Write-Host "Build completed! Check the electron_app/dist directory for the output." 