const { contextBridge, ipcRenderer } = require('electron');

// レンダラープロセスに公開するAPI
contextBridge.exposeInMainWorld('api', {
    // ファイル選択
    selectFiles: async (options = { multiple: false }) => {
        return await ipcRenderer.invoke('select-files', options);
    },

    // 単一画像のレタッチ
    retouchSingle: async (filePath, modelName) => {
        return await ipcRenderer.invoke('retouch-single', filePath, modelName);
    },

    // 複数画像の一括レタッチ
    retouchBatch: async (filePaths, modelName) => {
        return await ipcRenderer.invoke('retouch-batch', filePaths, modelName);
    },

    // モデル一覧の取得
    getModels: async () => {
        return await ipcRenderer.invoke('get-models');
    },

    // モデルの追加学習
    trainModel: async (data) => {
        return await ipcRenderer.invoke('train-model', data);
    },

    // ファイルの保存
    saveFiles: async (files) => {
        return await ipcRenderer.invoke('save-files', files);
    }
}); 