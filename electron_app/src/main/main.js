const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');
const { spawn } = require('child_process');

// ログファイルの設定
const LOG_FILE = path.join(app.getPath('userData'), 'logs', 'main.log');

// ログディレクトリの作成
function setupLogging() {
    const logDir = path.dirname(LOG_FILE);
    if (!fs.existsSync(logDir)) {
        fs.mkdirSync(logDir, { recursive: true });
    }
}

// ログ出力関数
function log(level, message, ...args) {
    const timestamp = new Date().toISOString();
    const logMessage = `${timestamp} [${level}] ${message}`;
    
    // コンソールに出力
    console.log(logMessage, ...args);
    
    // ファイルに出力
    try {
        fs.appendFileSync(LOG_FILE, logMessage + (args.length > 0 ? ' ' + JSON.stringify(args) : '') + '\n');
    } catch (error) {
        console.error('Failed to write to log file:', error);
    }
}

// バックエンドプロセスの参照を保持
let backendProcess = null;

// バックエンドの設定
const BACKEND_PORT = 8080;  // ポートを8080に変更
const API_BASE_URL = `http://127.0.0.1:${BACKEND_PORT}`;

// バックエンドの実行ファイルパスを取得
function getBackendPath() {
    const isDev = process.env.NODE_ENV === 'development';
    if (isDev) {
        const command = path.join(__dirname, '../../../python_backend/venv/Scripts/python.exe');
        //const scriptPath = path.join(__dirname, '../../../python_backend/retouch_backend.py');
        const scriptPath = path.join(__dirname, '../../../python_backend/main.py');
        
        log('INFO', 'Development mode backend paths', {
            command,
            scriptPath,
            commandExists: fs.existsSync(command),
            scriptExists: fs.existsSync(scriptPath)
        });
        
        return {
            command,
            args: [scriptPath]
        };
    }

    // 本番環境ではビルドされた実行ファイルを使用
    const backendPath = path.join(process.resourcesPath, 'backend', 'retouch_backend.exe');
    log('INFO', 'Production mode backend path', {
        backendPath,
        exists: fs.existsSync(backendPath),
        resourcesPath: process.resourcesPath,
        stats: fs.existsSync(backendPath) ? fs.statSync(backendPath) : null
    });
    
    // バックエンドディレクトリの内容を確認
    const backendDir = path.dirname(backendPath);
    if (fs.existsSync(backendDir)) {
        try {
            const files = fs.readdirSync(backendDir);
            log('INFO', 'Backend directory contents:', {
                directory: backendDir,
                files
            });
        } catch (error) {
            log('ERROR', 'Failed to read backend directory:', error);
        }
    } else {
        log('ERROR', 'Backend directory does not exist:', backendDir);
    }
    
    return {
        command: backendPath,
        args: []
    };
}

// バックエンドの起動
function startBackend() {
    return new Promise((resolve, reject) => {
        const { command, args } = getBackendPath();
        const backendDir = path.dirname(command);
        
        log('INFO', 'Starting backend process', {
            command,
            backendDir,
            cwd: process.cwd(),
            resourcesPath: process.resourcesPath,
            port: BACKEND_PORT
        });
        
        // バックエンドの実行ファイルが存在するか確認
        if (!fs.existsSync(command)) {
            const error = new Error(`Backend executable not found at: ${command}`);
            log('ERROR', error.message);
            reject(error);
            return;
        }

        try {
            // modelsディレクトリを確認
            const modelsDir = path.join(backendDir, 'models');
            if (!fs.existsSync(modelsDir)) {
                log('INFO', `Creating models directory at: ${modelsDir}`);
                fs.mkdirSync(modelsDir, { recursive: true });
            }

            // 環境変数の設定
            const env = {
                ...process.env,
                PYTHONUNBUFFERED: '1',
                PATH: `${backendDir};${process.env.PATH}`,
                PYTHONPATH: backendDir,
                MODELS_DIR: modelsDir,
                PORT: BACKEND_PORT.toString()  // ポート番号を環境変数として渡す
            };

            log('INFO', 'Spawning backend process with environment:', {
                PATH: env.PATH,
                PYTHONPATH: env.PYTHONPATH,
                MODELS_DIR: env.MODELS_DIR,
                PORT: env.PORT
            });

            backendProcess = spawn(command, args, {
                stdio: ['pipe', 'pipe', 'pipe'],
                env: env,
                cwd: backendDir,
                windowsHide: false,
                shell: true
            });

            if (!backendProcess) {
                const error = new Error('Failed to create backend process');
                log('ERROR', error.message);
                reject(error);
                return;
            }

            log('INFO', `Backend process spawned with PID: ${backendProcess.pid}`);
        } catch (error) {
            log('ERROR', 'Failed to spawn backend process', error);
            reject(error);
            return;
        }

        backendProcess.stdout.on('data', (data) => {
            const output = data.toString();
            log('INFO', 'Backend stdout:', output);
            // サーバーの起動完了を検知（標準出力の場合）
            if (output.includes(`Uvicorn running on http://127.0.0.1:${BACKEND_PORT}`)) {
                serverStarted = true;
                resolve();
            }
        });

        let serverStarted = false;
        backendProcess.stderr.on('data', (data) => {
            const output = data.toString();
            log('ERROR', 'Backend stderr:', output);
            
            // サーバーの起動完了を検知（標準エラー出力の場合）
            if (output.includes(`Uvicorn running on http://127.0.0.1:${BACKEND_PORT}`)) {
                serverStarted = true;
                resolve();
            }
        });

        backendProcess.on('error', (error) => {
            log('ERROR', 'Backend process error:', error);
            reject(error);
        });

        backendProcess.on('close', (code, signal) => {
            log('WARN', 'Backend process closed', { code, signal });
            backendProcess = null;
            if (!serverStarted) {
                reject(new Error(`Backend process exited with code ${code}, signal: ${signal}`));
            }
        });

        backendProcess.on('exit', (code, signal) => {
            log('WARN', 'Backend process exited', { code, signal });
        });

        // タイムアウト設定（30秒に延長）
        setTimeout(() => {
            if (!serverStarted) {
                log('ERROR', 'Backend server startup timeout');
                if (backendProcess) {
                    try {
                        backendProcess.kill('SIGTERM');
                    } catch (error) {
                        log('ERROR', 'Error killing backend process:', error);
                    }
                }
                reject(new Error('Backend server startup timeout'));
            }
        }, 30000);
    });
}

// 開発モードの場合はホットリロードを有効化
if (process.env.NODE_ENV === 'development') {
    require('electron-reload')(__dirname, {
        electron: path.join(__dirname, '../../node_modules', '.bin', 'electron'),
        hardResetMethod: 'exit'
    });
}

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

// アプリケーション起動時にログ設定を初期化
app.whenReady().then(() => {
    setupLogging();
    log('INFO', 'Application starting...');
    startBackend()
        .then(() => {
            log('INFO', 'Backend started successfully');
            createWindow();
        })
        .catch((error) => {
            log('ERROR', 'Failed to start application:', error);
            dialog.showErrorBox('起動エラー',
                `アプリケーションの起動に失敗しました: ${error.message}\n\n` +
                `ログファイル: ${LOG_FILE}`);
            app.quit();
        });
});

// Pythonサーバーをシャットダウンする関数
function shutdownPythonServer() {
    return axios.get(`${API_BASE_URL}/shutdown`, { timeout: 5000 })
    .then((res) => {
        log('INFO', 'Successfully called /shutdown on Python backend:', res.data);
    })
    .catch((err) => {
        log('ERROR', 'Failed to call /shutdown on Python backend:', err);
    });
}

// 全てのウィンドウが閉じられたらアプリを終了
app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') 
        shutdownPythonServer()
            .finally(() => {
                if (backendProcess) {
                    backendProcess.kill();
                    backendProcess = null;
                }
                app.quit();
            });
});

// アプリケーション終了時にバックエンドプロセスを終了
//app.on('before-quit', () => {
//    if (backendProcess) {
//        backendProcess.kill();
//    }
//});

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
ipcMain.handle('retouch-single', async (event, filePath, modelName = 'trained_4factor_model3') => {
    try {
        const formData = new FormData();
        formData.append('file', fs.createReadStream(filePath));
        // Python 側の retouch_single_image で受け取る "model_name" というキーに合わせる
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
ipcMain.handle('retouch-batch', async (event, filePaths, modelName = 'trained_4factor_model3') => {
    try {
        const formData = new FormData();
        filePaths.forEach(filePath => {
            formData.append('files', fs.createReadStream(filePath));
        });
        // こちらも同様に "model_name" キーで送る
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
        // data が正しく渡っているかチェック
        if (!data || !data.beforeImage || !data.afterImage) {
            throw new Error('beforeImage/afterImage が正しく指定されていません。');
        }

        const formData = new FormData();
        
        // ファイルパスの正規化（file:// プロトコルの除去とデコード）
        const beforeImagePath = decodeURIComponent(data.beforeImage.replace(/^file:\/\/\/?/, ''));
        const afterImagePath = decodeURIComponent(data.afterImage.replace(/^file:\/\/\/?/, ''));
        
        console.log('Processing file paths:');
        console.log('Original before path:', data.beforeImage);
        console.log('Original after path:', data.afterImage);
        console.log('Normalized before path:', beforeImagePath);
        console.log('Normalized after path:', afterImagePath);
        
        // ファイルの存在確認
        if (!fs.existsSync(beforeImagePath)) {
            console.error('Before image not found:', beforeImagePath);
            throw new Error(`レタッチ前の画像が見つかりません: ${beforeImagePath}`);
        }
        if (!fs.existsSync(afterImagePath)) {
            console.error('After image not found:', afterImagePath);
            throw new Error(`レタッチ後の画像が見つかりません: ${afterImagePath}`);
        }
        
        // ファイルストリームの作成
        const beforeStream = fs.createReadStream(beforeImagePath);
        const afterStream = fs.createReadStream(afterImagePath);
        
        // ファイル名を取得
        const beforeFileName = path.basename(beforeImagePath);
        const afterFileName = path.basename(afterImagePath);
        
        formData.append('before_image', beforeStream, {
            filename: beforeFileName,
            contentType: 'image/jpeg'
        });
        formData.append('after_image', afterStream, {
            filename: afterFileName,
            contentType: 'image/jpeg'
        });
        formData.append('base_model', String(data.baseModel || ''));
        formData.append('new_model_name', String(data.newModelName || ''));

        console.log('typeof newModelName:', typeof data.newModelName)

        console.log('Sending training request:');
        console.log('Model name:', data.newModelName);
        console.log('Base model:', data.baseModel);
        console.log('Before image filename:', beforeFileName);
        console.log('After image filename:', afterFileName);

        const response = await axios.post(`${API_BASE_URL}/api/train`, formData, {
            headers: {
                ...formData.getHeaders(),
                'Content-Type': 'multipart/form-data'
            },
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });

        console.log('Training response:', response.data);
        return response.data;
    } catch (error) {
        console.error('Error in train-model:', error);
        if (error.response) {
            console.error('Response data:', error.response.data);
            console.error('Response status:', error.response.status);
        }
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