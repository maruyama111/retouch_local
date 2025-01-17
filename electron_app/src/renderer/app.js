// アプリケーションの状態管理
const state = {
    selectedFiles: [],
    processedFiles: [],
    models: [],
    selectedModel: 'default',
    isProcessing: false,
    error: null,
    success: null,
    // 追加学習用の状態
    beforeImages: [],  // レタッチ前の画像リスト
    afterImages: [],   // レタッチ後の画像リスト
    isTraining: false,
    currentTab: 'retouch',  // タブの状態を保持
    lastTrainedModel: null,  // 最後に学習したモデル名
    customModelName: ''  // ユーザー指定のモデル名
};

// UIの更新
function updateUI() {
    const app = document.getElementById('app');
    app.innerHTML = `
        <div class="container">
            <h1>レタッチアプリ</h1>
            
            ${state.error ? `<div class="error-message">${state.error}</div>` : ''}
            ${state.success ? `<div class="success-message">${state.success}</div>` : ''}
            
            <div class="tabs">
                <button id="retouchTab" class="tab-button ${state.currentTab === 'retouch' ? 'active' : ''}">レタッチ</button>
                <button id="trainingTab" class="tab-button ${state.currentTab === 'training' ? 'active' : ''}">追加学習</button>
            </div>

            <div id="retouchContent" class="tab-content" style="display: ${state.currentTab === 'retouch' ? 'block' : 'none'}">
                <div class="model-select">
                    <label for="model">使用するモデル:</label>
                    <select id="model" ${state.isProcessing ? 'disabled' : ''}>
                        ${state.models.map(model => `
                            <option value="${model.name}" ${model.name === state.selectedModel ? 'selected' : ''}>
                                ${model.name}
                            </option>
                        `).join('')}
                    </select>
                </div>
                
                <div class="button-group">
                    <button id="selectSingleFile" data-multiple="false" ${state.isProcessing ? 'disabled' : ''}>
                        画像を選択
                    </button>
                    <button id="selectMultipleFiles" data-multiple="true" ${state.isProcessing ? 'disabled' : ''}>
                        複数の画像を選択
                    </button>
                    ${state.selectedFiles.length > 0 ? `
                        <button id="processButton" ${state.isProcessing ? 'disabled' : ''}>
                            ${state.isProcessing ? '処理中...' : 'レタッチ実行'}
                        </button>
                    ` : ''}
                </div>
                
                ${state.selectedFiles.length > 0 ? `
                    <div class="image-grid">
                        ${state.selectedFiles.map((file, index) => `
                            <div class="image-card">
                                <img src="file://${file}" alt="選択された画像 ${index + 1}">
                                <div class="info">
                                    ${file.split('\\').pop()}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
                
                ${state.processedFiles.length > 0 ? `
                    <h2>処理結果</h2>
                    <div class="button-group">
                        <button id="saveAllProcessed" class="save-button">
                            処理結果をまとめて保存
                        </button>
                    </div>
                    <div class="image-grid">
                        ${state.processedFiles.map((file, index) => `
                            <div class="image-card">
                                <img src="file://${file}" alt="処理結果 ${index + 1}">
                                <div class="info">
                                    ${file.split('\\').pop()}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
            </div>

            <div id="trainingContent" class="tab-content" style="display: ${state.currentTab === 'training' ? 'block' : 'none'}">
                <h2>モデルの追加学習</h2>
                <div class="training-container">
                    <div class="training-section">
                        <h3>学習データの選択</h3>
                        
                        <div class="model-name-input">
                            <label for="customModelName">新しいモデル名:</label>
                            <input type="text" 
                                   id="customModelName" 
                                   value="${state.customModelName}"
                                   placeholder="新しいモデル名を入力してください"
                                   ${state.isTraining ? 'disabled' : ''}>
                        </div>

                        <div class="training-image-selection">
                            <div class="image-selection-group">
                                <h4>レタッチ前の画像（${state.beforeImages.length}枚選択中）</h4>
                                <button id="selectBeforeImages" ${state.isTraining ? 'disabled' : ''}>
                                    レタッチ前の画像を選択
                                </button>
                                ${state.beforeImages.length > 0 ? `
                                    <div class="image-grid">
                                        ${state.beforeImages.map((image, index) => `
                                            <div class="image-card">
                                                <img src="file://${image}" alt="レタッチ前 ${index + 1}">
                                                <div class="info">画像 ${index + 1}</div>
                                            </div>
                                        `).join('')}
                                    </div>
                                ` : ''}
                            </div>

                            <div class="image-selection-group">
                                <h4>レタッチ後の画像（${state.afterImages.length}枚選択中）</h4>
                                <button id="selectAfterImages" ${state.isTraining ? 'disabled' : ''}>
                                    レタッチ後の画像を選択
                                </button>
                                ${state.afterImages.length > 0 ? `
                                    <div class="image-grid">
                                        ${state.afterImages.map((image, index) => `
                                            <div class="image-card">
                                                <img src="file://${image}" alt="レタッチ後 ${index + 1}">
                                                <div class="info">画像 ${index + 1}</div>
                                            </div>
                                        `).join('')}
                                    </div>
                                ` : ''}
                            </div>
                        </div>

                        ${state.beforeImages.length > 0 && state.afterImages.length > 0 && 
                          state.beforeImages.length === state.afterImages.length ? `
                            <button id="startTraining" ${state.isTraining ? 'disabled' : ''}>
                                ${state.isTraining ? '学習中...' : '学習開始'}
                            </button>
                        ` : ''}
                        
                        ${state.lastTrainedModel ? `
                            <div class="success-message">
                                追加学習が完了しました。<br>
                                新しいモデル名: ${state.lastTrainedModel}
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        </div>
    `;

    // イベントリスナーの設定
    setupEventListeners();
}

// タブ切り替え
function switchTab(tabName) {
    state.currentTab = tabName;
    updateUI();
}

// 学習用画像ペア追加ハンドラ
async function handleAddTrainingPair() {
    try {
        // レタッチ前の画像を選択
        const beforeFiles = await window.api.selectFiles({ multiple: false });
        if (!beforeFiles || beforeFiles.length === 0) return;

        // レタッチ後の画像を選択
        const afterFiles = await window.api.selectFiles({ multiple: false });
        if (!afterFiles || afterFiles.length === 0) return;

        // 画像ペアを追加
        state.trainingData.push({
            before: beforeFiles[0],
            after: afterFiles[0]
        });

        state.error = null;
        updateUI();
    } catch (error) {
        state.error = '画像の選択中にエラーが発生しました';
        updateUI();
    }
}

// 画像ペア削除ハンドラ
function handleRemoveTrainingPair(index) {
    state.trainingData.splice(index, 1);
    updateUI();
}

// イベントリスナーの設定
function setupEventListeners() {
    const selectSingleFile = document.getElementById('selectSingleFile');
    const selectMultipleFiles = document.getElementById('selectMultipleFiles');
    const processButton = document.getElementById('processButton');
    const modelSelect = document.getElementById('model');
    const retouchTab = document.getElementById('retouchTab');
    const trainingTab = document.getElementById('trainingTab');
    const startTraining = document.getElementById('startTraining');
    const selectBeforeImages = document.getElementById('selectBeforeImages');
    const selectAfterImages = document.getElementById('selectAfterImages');
    const saveAllProcessed = document.getElementById('saveAllProcessed');
    const customModelNameInput = document.getElementById('customModelName');

    if (selectSingleFile) {
        selectSingleFile.addEventListener('click', () => handleSelectFiles(false));
    }
    if (selectMultipleFiles) {
        selectMultipleFiles.addEventListener('click', () => handleSelectFiles(true));
    }
    if (processButton) {
        processButton.addEventListener('click', handleProcess);
    }
    if (modelSelect) {
        modelSelect.addEventListener('change', handleModelChange);
    }
    if (retouchTab) {
        retouchTab.addEventListener('click', () => switchTab('retouch'));
    }
    if (trainingTab) {
        trainingTab.addEventListener('click', () => switchTab('training'));
    }
    if (startTraining) {
        startTraining.addEventListener('click', handleStartTraining);
    }
    if (selectBeforeImages) {
        selectBeforeImages.addEventListener('click', () => handleSelectTrainingImages('before'));
    }
    if (selectAfterImages) {
        selectAfterImages.addEventListener('click', () => handleSelectTrainingImages('after'));
    }
    if (saveAllProcessed) {
        saveAllProcessed.addEventListener('click', handleSaveAllProcessed);
    }
    if (customModelNameInput) {
        customModelNameInput.addEventListener('input', (e) => {
            state.customModelName = e.target.value;
        });
    }
}

// 学習用画像選択ハンドラ
async function handleSelectTrainingImage(type) {
    try {
        const files = await window.api.selectFiles({ multiple: false });
        if (files && files.length > 0) {
            state.trainingData[type] = files[0];
            state.error = null;
            updateUI();
        }
    } catch (error) {
        state.error = '画像の選択中にエラーが発生しました';
        updateUI();
    }
}

// 学習開始ハンドラ
async function handleStartTraining() {
    if (state.beforeImages.length !== state.afterImages.length) {
        state.error = 'レタッチ前後の画像の数が一致しません';
        updateUI();
        return;
    }

    if (!state.customModelName.trim()) {
        state.error = '新しいモデル名を入力してください';
        updateUI();
        return;
    }

    try {
        state.isTraining = true;
        state.error = null;
        state.success = null;
        updateUI();

        let currentModel = state.selectedModel;
        
        // 各画像ペアで学習を実行
        for (let i = 0; i < state.beforeImages.length; i++) {
            const result = await window.api.trainModel({
                beforeImage: state.beforeImages[i],
                afterImage: state.afterImages[i],
                baseModel: currentModel,
                newModelName: state.customModelName
            });

            if (result.status !== 'success') {
                throw new Error(`画像ペア ${i + 1} の学習に失敗しました: ${result.message}`);
            }
            
            // 次の学習のベースモデルを更新
            currentModel = result.new_model;
        }

        // 最後に学習したモデル名を保存
        state.lastTrainedModel = currentModel;
        state.success = 'すべての画像ペアの学習が完了しました';
        
        // モデル一覧を更新
        const modelsResult = await window.api.getModels();
        if (modelsResult.status === 'success') {
            state.models = modelsResult.models;
        }
    } catch (error) {
        state.error = `学習中にエラーが発生しました: ${error.message}`;
    } finally {
        state.isTraining = false;
        state.beforeImages = [];
        state.afterImages = [];
        state.customModelName = '';
        updateUI();
    }
}

// ファイル選択ハンドラ
async function handleSelectFiles(multiple) {
    try {
        const files = await window.api.selectFiles({ multiple });
        if (files && files.length > 0) {
            state.selectedFiles = files;
            state.processedFiles = [];
            state.error = null;
            state.success = null;
            updateUI();
        }
    } catch (error) {
        state.error = '画像の選択中にエラーが発生しました';
        updateUI();
    }
}

// 処理実行ハンドラ
async function handleProcess() {
    try {
        state.isProcessing = true;
        state.error = null;
        state.success = null;
        updateUI();

        if (state.selectedFiles.length === 1) {
            const result = await window.api.retouchSingle(state.selectedFiles[0], state.selectedModel);
            if (result.status === 'success') {
                state.processedFiles = [result.result_path];
                state.success = '画像の処理が完了しました';
            } else {
                throw new Error(result.message);
            }
        } else {
            const result = await window.api.retouchBatch(state.selectedFiles, state.selectedModel);
            if (result.status === 'success') {
                state.processedFiles = result.results.map(r => r.result_path);
                state.success = '全ての画像の処理が完了しました';
            } else {
                throw new Error(result.message);
            }
        }
    } catch (error) {
        state.error = `処理中にエラーが発生しました: ${error.message}`;
    } finally {
        state.isProcessing = false;
        updateUI();
    }
}

// モデル選択ハンドラ
async function handleModelChange(event) {
    state.selectedModel = event.target.value;
    updateUI();
}

// 初期化処理
async function initialize() {
    try {
        console.log('Fetching models...');
        // モデル一覧の取得
        const result = await window.api.getModels();
        console.log('Models result:', result);
        if (result.status === 'success') {
            state.models = result.models;
        } else {
            throw new Error(result.message || 'Failed to fetch models');
        }
    } catch (error) {
        console.error('Error in initialize:', error);
        state.error = 'モデル一覧の取得に失敗しました: ' + error.message;
    }
    
    updateUI();
}

// 学習用画像選択ハンドラ
async function handleSelectTrainingImages(type) {
    try {
        const files = await window.api.selectFiles({ multiple: true });
        if (files && files.length > 0) {
            if (type === 'before') {
                state.beforeImages = files;
            } else {
                state.afterImages = files;
            }
            state.error = null;
            updateUI();
        }
    } catch (error) {
        state.error = '画像の選択中にエラーが発生しました';
        updateUI();
    }
}

// 処理済み画像の一括保存
async function handleSaveAllProcessed() {
    try {
        const result = await window.api.saveFiles(state.processedFiles);
        if (result.status === 'success') {
            state.success = '処理結果を保存しました';
            state.error = null;
        } else if (result.status === 'canceled') {
            // キャンセルの場合はエラーとして扱わない
            return;
        } else {
            throw new Error(result.message);
        }
    } catch (error) {
        state.error = `保存中にエラーが発生しました: ${error.message}`;
        state.success = null;
    }
    updateUI();
}

// アプリケーションの初期化
initialize(); 