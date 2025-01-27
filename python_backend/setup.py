import sys
from cx_Freeze import setup, Executable
import os
from pathlib import Path
import site

# 依存パッケージを指定
packages = [
    "mediapipe",
    "mediapipe.python",
    "mediapipe.python.solutions",
    "mediapipe.python.solutions.pose",
    "mediapipe.python.solutions.face_mesh",
    "mediapipe.framework.formats",
    "mediapipe.tasks.python",
    "uvicorn",
    "uvicorn.config",
    "uvicorn.main",
    "uvicorn.server",
    "uvicorn.supervisors",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.websockets",
    "fastapi",
    "numpy",
    "cv2",
    "PIL",
    "starlette",
    "starlette.routing",
    "starlette.applications",
    "pydantic",
    "typing",
    "typing_extensions",
    "asyncio",
    "click",
    "h11",
    "websockets",
]

# 必要なモジュールを指定
includes = [
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
]

# MediaPipeのデータファイルを収集
site_packages = site.getsitepackages()[0]
mediapipe_path = os.path.join(site_packages, "mediapipe")
include_files = []

print(f"Looking for MediaPipe files in: {mediapipe_path}")

# MediaPipeのモデルファイルを含める
for root, dirs, files in os.walk(mediapipe_path):
    for file in files:
        if file.endswith(('.tflite', '.pbtxt')):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, mediapipe_path)
            target_path = os.path.join('mediapipe', rel_path)
            include_files.append((full_path, target_path))
            print(f"Including file: {full_path} -> {target_path}")

# ビルド設定
build_exe_options = {
    "packages": packages,
    "includes": includes,
    "include_files": include_files,
    "excludes": [],
    "include_msvcr": True,
    "zip_include_packages": "*",
    "zip_exclude_packages": [],
    "build_exe": "dist/retouch_backend",
}

# 実行ファイルの設定
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="retouch_backend",
    version="1.0",
    description="Retouch Backend Application",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            script="main.py",
            base=base,
            target_name="retouch_backend.exe"
        )
    ]
) 