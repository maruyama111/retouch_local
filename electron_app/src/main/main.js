const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');
const { spawn } = require('child_process');

// バックエンドプロセスの参照を保持
let backendProcess = null;

// バックエンドの実行ファイルパスを取得
function getBackendPath() {
    const isDev = process.env.NODE_ENV === 'development';
    if (isDev) {
        // 開発環境では直接Pythonスクリプトを実行
        return {
            command: 'python',
            args: [path.join(__dirname, '../../../python_backend/main.py')]
        };
    }

    // 本番環境ではビルドされた実行ファイルを使用
    if (process.platform === 'win32') {
        return {
            command: path.join(process.resourcesPath, 'backend', 'retouch_backend.exe'),
            args: []
        };
    } else {
        return {
            command: path.join(process.resourcesPath, 'backend', 'retouch_backend'),
            args: []
        };
    }
}

// バックエンドの起動
function startBackend() {
    const { command, args } = getBackendPath();
    
    console.log('Starting backend process:', command, args);
    
    backendProcess = spawn(command, args, {
        stdio: 'pipe'
    });

    backendProcess.stdout.on('data', (data) => {
        console.log('Backend stdout:', data.toString());
    });

    backendProcess.stderr.on('data', (data) => {
        console.error('Backend stderr:', data.toString());
    });

    backendProcess.on('error', (error) => {
        console.error('Failed to start backend:', error);
    });

    backendProcess.on('close', (code) => {
        console.log('Backend process exited with code:', code);
        backendProcess = null;
    });
}

// 開発モードの場合はホットリロードを有効化
if (process.env.NODE_ENV === 'development') {
    require('electron-reload')(__dirname, {
        electron: path.join(__dirname, '../../node_modules', '.bin', 'electron'),
        hardResetMethod: 'exit'
    });
}

// APIのベースURL
const API_BASE_URL = 'http://127.0.0.1:8000';

function createWindow() {
    const mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, '../preload/preload.js')
        }
    });

    // 開発モードの場合はDevToolsを開く
    if (process.env.NODE_ENV === 'development') {
        mainWindow.webContents.openDevTools();
    }

    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
}

// アプリの準備が整ったらウィンドウを作成とバックエンドを起動
app.whenReady().then(() => {
    startBackend();
    createWindow();

    app.on('activate', function () {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

// 全てのウィンドウが閉じられたらアプリを終了
app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') app.quit();
});

// アプリケーション終了時にバックエンドプロセスを終了
app.on('before-quit', () => {
    if (backendProcess) {
        backendProcess.kill();
    }
});

// ファイル選択ダイアログを開く
ipcMain.handle('select-files', async (event, options) => {
    const result = await dialog.showOpenDialog({
        properties: options.multiple ? ['openFile', 'multiSelections'] : ['openFile'],
        filters: [
            { name: 'Images', extensions: ['jpg', 'jpeg', 'png'] }
        ]
    });
    return result.filePaths;
});

// 単一画像のレタッチ処理
ipcMain.handle('retouch-single', async (event, filePath, modelName = 'default') => {
    try {
        const formData = new FormData();
        formData.append('file', fs.createReadStream(filePath));
        formData.append('model_name', modelName);

        const response = await axios.post(`${API_BASE_URL}/api/retouch/single`, formData, {
            headers: formData.getHeaders()
        });

        if (response.data.status === 'success') {
            // 元のファイル名を保持して新しいパスを生成
            const originalName = path.basename(filePath);
            const newPath = path.join(path.dirname(response.data.result_path), 
                                    `retouched_${originalName}`);
            await fs.promises.rename(response.data.result_path, newPath);
            response.data.result_path = newPath;
        }

        return response.data;
    } catch (error) {
        console.error('Error in retouch-single:', error);
        throw error;
    }
});

// 複数画像の一括レタッチ処理
ipcMain.handle('retouch-batch', async (event, filePaths, modelName = 'default') => {
    try {
        const formData = new FormData();
        filePaths.forEach(filePath => {
            formData.append('files', fs.createReadStream(filePath));
        });
        formData.append('model_name', modelName);

        const response = await axios.post(`${API_BASE_URL}/api/retouch/batch`, formData, {
            headers: formData.getHeaders()
        });

        if (response.data.status === 'success') {
            // 各ファイルの元の名前を保持
            response.data.results = await Promise.all(response.data.results.map(async (result, index) => {
                const originalName = path.basename(filePaths[index]);
                const newPath = path.join(path.dirname(result.result_path), 
                                        `retouched_${originalName}`);
                await fs.promises.rename(result.result_path, newPath);
                return {
                    ...result,
                    result_path: newPath,
                    original_name: originalName
                };
            }));
        }

        return response.data;
    } catch (error) {
        console.error('Error in retouch-batch:', error);
        throw error;
    }
});

// モデル一覧の取得
ipcMain.handle('get-models', async () => {
    try {
        const response = await axios.get(`${API_BASE_URL}/api/models`);
        return response.data;
    } catch (error) {
        console.error('Error in get-models:', error);
        throw error;
    }
});

// モデルの追加学習
ipcMain.handle('train-model', async (event, data) => {
    try {
        const formData = new FormData();
        formData.append('before_image', fs.createReadStream(data.beforeImage));
        formData.append('after_image', fs.createReadStream(data.afterImage));
        formData.append('base_model', data.baseModel);
        
        // 新しいモデル名を送信
        formData.append('new_model_name', data.newModelName || '');

        console.log('Sending training request with model name:', data.newModelName);  // デバッグ用ログ

        const response = await axios.post(`${API_BASE_URL}/api/train`, formData, {
            headers: {
                ...formData.getHeaders(),
                'Content-Type': 'multipart/form-data'
            }
        });

        console.log('Training response:', response.data);  // デバッグ用ログ
        return response.data;
    } catch (error) {
        console.error('Error in train-model:', error);
        throw error;
    }
});

// ファイル保存ハンドラ
ipcMain.handle('save-files', async (event, files) => {
    try {
        const result = await dialog.showOpenDialog({
            properties: ['openDirectory']
        });

        if (!result.canceled) {
            const saveDir = result.filePaths[0];
            const savedFiles = [];

            for (const file of files) {
                const fileName = path.basename(file);
                const targetPath = path.join(saveDir, fileName);
                await fs.promises.copyFile(file, targetPath);
                savedFiles.push(targetPath);
            }

            // 一時ファイルを削除
            for (const file of files) {
                try {
                    await fs.promises.unlink(file);
                } catch (error) {
                    console.error('Error deleting temporary file:', error);
                }
            }

            return {
                status: 'success',
                message: 'ファイルを保存しました',
                savedFiles
            };
        }
        return {
            status: 'canceled',
            message: '保存がキャンセルされました'
        };
    } catch (error) {
        return {
            status: 'error',
            message: `保存中にエラーが発生しました: ${error.message}`
        };
    }
}); 