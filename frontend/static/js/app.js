document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initSplitFunctionality();
    initSmartSplitFunctionality();
    initMergeFunctionality();
    initCompactMergeFunctionality();
    initToolboxFunctionality();
    initFileInputUX();
    initCanvasZoom();
});

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

function updateCanvasInfo(text) {
    document.getElementById('canvas-info').textContent = text;
}

function initNavigation() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const targetModule = btn.dataset.tab;
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            const targetPanel = document.getElementById(`${targetModule}-panel`);
            if (targetPanel) targetPanel.classList.add('active');

            if (targetModule === 'removebg') {
                switchCanvasLayer('tool-bg-canvas');
            } else if (targetModule === 'split') {
                switchCanvasLayer('split-panel-canvas');
            } else if (targetModule === 'merge') {
                switchCanvasLayer('merge-panel-canvas');
            }

            const activeSubTab = targetPanel?.querySelector('.segment-btn.active');
            if (activeSubTab) activeSubTab.click();
        });
    });

    document.querySelectorAll('.segment-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const parentPanel = btn.closest('.panel');

            parentPanel.querySelectorAll('.segment-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            parentPanel.querySelectorAll('.subpanel').forEach(sp => sp.classList.remove('active'));
            const subTabId = btn.dataset.subtab;
            const targetSubPanel = document.getElementById(`${subTabId}-subpanel`);
            if (targetSubPanel) targetSubPanel.classList.add('active');

            updateCanvasDisplayBasedOnSubTab(subTabId);
        });
    });
}

function updateCanvasDisplayBasedOnSubTab(subTabId) {
    const splitLayer = document.getElementById('split-panel-canvas');
    const mergeLayer = document.getElementById('merge-panel-canvas');

    Array.from(splitLayer.children).forEach(c => c.classList.add('hidden'));
    Array.from(mergeLayer.children).forEach(c => c.classList.add('hidden'));

    if (subTabId === 'grid') {
        switchCanvasLayer('split-panel-canvas');
        document.getElementById('split-preview').classList.remove('hidden');
        if (document.getElementById('split-files').children.length > 0) {
            document.getElementById('split-result').classList.remove('hidden');
        }
    } else if (subTabId === 'irregular') {
        switchCanvasLayer('split-panel-canvas');
        const mode = document.getElementById('smart-split-mode').value;
        if (mode === 'grid') {
            document.getElementById('irregular-preview').classList.remove('hidden');
        } else {
            document.getElementById('bbox-preview').classList.remove('hidden');
        }
    } else if (subTabId === 'merge-grid') {
        switchCanvasLayer('merge-panel-canvas');
        document.getElementById('merge-preview').classList.remove('hidden');
    } else if (subTabId === 'merge-compact') {
        switchCanvasLayer('merge-panel-canvas');
        document.getElementById('compact-merge-preview').classList.remove('hidden');
    }
}

function switchCanvasLayer(layerId) {
    document.querySelectorAll('.canvas-layer').forEach(l => l.classList.remove('active'));
    document.getElementById(layerId).classList.add('active');
}

function initSplitFunctionality() {
    const input = document.getElementById('split-image');
    
    input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleSingleImagePreview(file, 'split-preview');
            document.getElementById('split-grid-preview').classList.add('hidden');
            document.getElementById('split-result').classList.add('hidden');
        }
    });

    document.getElementById('split-preview-btn').addEventListener('click', async () => {
        if (!input.files[0]) return alert('请先选择图片');
        
        const formData = new FormData();
        formData.append('image', input.files[0]);
        appendSplitParams(formData);

        showLoading();
        try {
            const res = await fetch(apiUrl('/preview/split'), { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                const img = document.getElementById('split-grid-image');
                img.src = data.preview;
                document.getElementById('split-preview').classList.add('hidden');
                document.getElementById('split-grid-preview').classList.remove('hidden');
            } else {
                alert(data.error);
            }
        } catch (e) { alert(e.message); } finally { hideLoading(); }
    });

    document.getElementById('split-btn').addEventListener('click', async () => {
        if (!input.files[0]) return alert('请先选择图片');
        
        const formData = new FormData();
        formData.append('image', input.files[0]);
        appendSplitParams(formData);
        formData.append('prefix', document.getElementById('split-prefix').value);

        showLoading();
        try {
            const res = await fetch(apiUrl('/split'), { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                renderSplitResult(data, 'split-files', 'split-result', 'split-download-all');
            } else {
                alert(data.error);
            }
        } catch (e) { alert(e.message); } finally { hideLoading(); }
    });
}

function appendSplitParams(formData) {
    formData.append('rows', document.getElementById('split-rows').value);
    formData.append('cols', document.getElementById('split-cols').value);
    formData.append('margin_top', document.getElementById('margin-top').value);
    formData.append('margin_bottom', document.getElementById('margin-bottom').value);
    formData.append('margin_left', document.getElementById('margin-left').value);
    formData.append('margin_right', document.getElementById('margin-right').value);
}

function initSmartSplitFunctionality() {
    const modeSelect = document.getElementById('smart-split-mode');
    const irregularContent = document.getElementById('grid-mode-content');
    const bboxContent = document.getElementById('bbox-mode-content');

    modeSelect.addEventListener('change', (e) => {
        if (e.target.value === 'bbox') {
            irregularContent.classList.add('hidden');
            bboxContent.classList.remove('hidden');
            document.getElementById('irregular-preview').classList.add('hidden');
            document.getElementById('bbox-preview').classList.remove('hidden');
        } else {
            irregularContent.classList.remove('hidden');
            bboxContent.classList.add('hidden');
            document.getElementById('irregular-preview').classList.remove('hidden');
            document.getElementById('bbox-preview').classList.add('hidden');
        }
    });

    const irrInput = document.getElementById('irregular-image');
    irrInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleSingleImagePreview(file, 'irregular-preview');
    });

    document.getElementById('detect-btn').addEventListener('click', async () => {
        if (!irrInput.files[0]) return alert('请先选择图片');
        const formData = new FormData();
        formData.append('image', irrInput.files[0]);
        formData.append('margin_top', document.getElementById('irregular-margin-top').value);
        formData.append('margin_bottom', document.getElementById('irregular-margin-bottom').value);
        formData.append('margin_left', document.getElementById('irregular-margin-left').value);
        formData.append('margin_right', document.getElementById('irregular-margin-right').value);
        formData.append('threshold', document.getElementById('detect-threshold').value);
        formData.append('min_gap', document.getElementById('detect-min-gap').value);

        showLoading();
        try {
            const res = await fetch(apiUrl('/detect'), { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                document.getElementById('rows-split-input').value = data.rows_split.join(', ');
                document.getElementById('cols-split-input').value = data.cols_split.join(', ');
                document.getElementById('split-lines-editor').classList.remove('hidden');

                document.getElementById('irregular-split-btn').disabled = false;

                await previewIrr();

                alert(`检测到 ${data.rows} 行, ${data.cols} 列`);
            } else { alert(data.error); }
        } catch (e) { alert(e.message); } finally { hideLoading(); }
    });

    const previewIrr = async () => {
        const rowsStr = document.getElementById('rows-split-input').value;
        const colsStr = document.getElementById('cols-split-input').value;
        const parse = (s) => s.split(',').map(n => parseInt(n.trim())).filter(n => !isNaN(n)).sort((a,b)=>a-b);

        const formData = new FormData();
        formData.append('image', irrInput.files[0]);
        formData.append('rows_split', JSON.stringify(parse(rowsStr)));
        formData.append('cols_split', JSON.stringify(parse(colsStr)));
        formData.append('margin_top', document.getElementById('irregular-margin-top').value);
        formData.append('margin_bottom', document.getElementById('irregular-margin-bottom').value);
        formData.append('margin_left', document.getElementById('irregular-margin-left').value);
        formData.append('margin_right', document.getElementById('irregular-margin-right').value);

        showLoading();
        try {
            const res = await fetch(apiUrl('/preview/split/irregular'), { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                document.getElementById('irregular-grid-image').src = data.preview;
                document.getElementById('irregular-preview').classList.add('hidden');
                document.getElementById('irregular-grid-preview').classList.remove('hidden');
            } else { alert(data.error); }
        } catch (e) { alert(e.message); } finally { hideLoading(); }
    };

    document.getElementById('update-irregular-preview').addEventListener('click', previewIrr);

    document.getElementById('irregular-split-btn').addEventListener('click', async () => {
         const rowsStr = document.getElementById('rows-split-input').value;
         const colsStr = document.getElementById('cols-split-input').value;
         const parse = (s) => s.split(',').map(n => parseInt(n.trim())).filter(n => !isNaN(n)).sort((a,b)=>a-b);

         const formData = new FormData();
         formData.append('image', irrInput.files[0]);
         formData.append('rows_split', JSON.stringify(parse(rowsStr)));
         formData.append('cols_split', JSON.stringify(parse(colsStr)));
         formData.append('margin_top', document.getElementById('irregular-margin-top').value);
         formData.append('margin_bottom', document.getElementById('irregular-margin-bottom').value);
         formData.append('margin_left', document.getElementById('irregular-margin-left').value);
         formData.append('margin_right', document.getElementById('irregular-margin-right').value);
         formData.append('prefix', document.getElementById('irregular-prefix').value);

         showLoading();
         try {
             const res = await fetch(apiUrl('/split/irregular'), { method: 'POST', body: formData });
             const data = await res.json();
             if (data.success) {
                 renderSplitResult(data, 'irregular-files', 'irregular-result', 'irregular-download-all');
             } else { alert(data.error); }
         } catch(e) { alert(e.message); } finally { hideLoading(); }
    });

    const bboxInput = document.getElementById('bbox-image');
    let currentBboxes = [];

    bboxInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if(file) handleSingleImagePreview(file, 'bbox-preview');
    });

    document.getElementById('bbox-uniform-size').addEventListener('change', (e) => {
        document.getElementById('bbox-uniform-size-options').classList.toggle('hidden', !e.target.checked);
    });

    const runBBoxDetect = async () => {
        if (!bboxInput.files[0]) return alert('请先选择图片');
        const formData = new FormData();
        formData.append('image', bboxInput.files[0]);
        formData.append('threshold', document.getElementById('bbox-threshold').value);
        formData.append('min_size', document.getElementById('bbox-min-size').value);
        formData.append('margin_top', document.getElementById('bbox-margin-top').value);
        formData.append('margin_bottom', document.getElementById('bbox-margin-bottom').value);
        formData.append('margin_left', document.getElementById('bbox-margin-left').value);
        formData.append('margin_right', document.getElementById('bbox-margin-right').value);

        showLoading();
        try {
            const res = await fetch(apiUrl('/preview/split/bboxes'), { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                currentBboxes = data.bboxes;
                document.getElementById('bbox-grid-image').src = data.preview;
                document.getElementById('bbox-preview').classList.add('hidden');
                document.getElementById('bbox-grid-preview').classList.remove('hidden');

                document.getElementById('bbox-split-btn').disabled = false;

                if (data.bboxes.length > 0) {
                    const maxW = Math.max(...data.bboxes.map(b => b.width));
                    const maxH = Math.max(...data.bboxes.map(b => b.height));
                    document.getElementById('bbox-uniform-width').placeholder = `Max: ${maxW}`;
                    document.getElementById('bbox-uniform-height').placeholder = `Max: ${maxH}`;
                }
            } else { alert(data.error); }
        } catch (e) { alert(e.message); } finally { hideLoading(); }
    };

    document.getElementById('bbox-detect-btn').addEventListener('click', runBBoxDetect);

    document.getElementById('bbox-split-btn').addEventListener('click', async () => {
        if (currentBboxes.length === 0) return alert('请先检测边界');
        const formData = new FormData();
        formData.append('image', bboxInput.files[0]);
        formData.append('bboxes', JSON.stringify(currentBboxes));
        formData.append('prefix', document.getElementById('bbox-prefix').value);
        
        const uniform = document.getElementById('bbox-uniform-size').checked;
        formData.append('uniform_size', uniform);
        if (uniform) {
            formData.append('uniform_width', document.getElementById('bbox-uniform-width').value);
            formData.append('uniform_height', document.getElementById('bbox-uniform-height').value);
        }

        showLoading();
        try {
            const res = await fetch(apiUrl('/split/bboxes'), { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                renderSplitResult(data, 'bbox-files', 'bbox-result', 'bbox-download-all');
            } else { alert(data.error); }
        } catch(e) { alert(e.message); } finally { hideLoading(); }
    });
}

function renderSplitResult(data, containerId, resultLayerId, downloadBtnId) {
    const container = document.getElementById(containerId);
    container.innerHTML = data.files.map((f) => `
        <div class="preview-item" data-filename="${f}">
            <img src="${outputUrl(`${data.session_id}_split/${f}`)}">
            <div class="preview-info">${f}</div>
            <button class="preview-download" title="下载">⬇</button>
            <button class="preview-remove" title="删除">×</button>
        </div>
    `).join('');

    container.querySelectorAll('.preview-download').forEach(btn => {
        btn.addEventListener('click', () => {
            const filename = btn.closest('.preview-item').dataset.filename;
            const link = document.createElement('a');
            link.href = downloadUrl(data.session_id, filename);
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    });

    container.querySelectorAll('.preview-remove').forEach(btn => {
        btn.addEventListener('click', () => {
            const item = btn.closest('.preview-item');
            item.remove();
            if (container.children.length === 0) {
                document.getElementById(downloadBtnId).classList.add('hidden');
            }
        });
    });

    const parentLayer = container.closest('.canvas-layer');
    const previewSelectors = ['#split-preview', '#split-grid-preview', '#irregular-preview',
                               '#irregular-grid-preview', '#bbox-preview', '#bbox-grid-preview'];
    previewSelectors.forEach(selector => {
        const el = parentLayer.querySelector(selector);
        if (el) el.classList.add('hidden');
    });

    const resultLayer = document.getElementById(resultLayerId);
    if (resultLayer) {
        resultLayer.classList.remove('hidden');
        const resultContainer = resultLayer.querySelector('div');
        if (resultContainer) resultContainer.classList.remove('hidden');
    }

    const btn = document.getElementById(downloadBtnId);
    btn.classList.remove('hidden');
    btn.onclick = () => {
        const remainingFiles = Array.from(container.querySelectorAll('.preview-item'))
            .map(item => item.dataset.filename);
        remainingFiles.forEach((file, i) => {
            setTimeout(() => {
                const link = document.createElement('a');
                link.href = downloadUrl(data.session_id, file);
                link.download = file;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }, i * 200);
        });
    };
}

let mergeFiles = [];

function initMergeFunctionality() {
    const input = document.getElementById('merge-images');
    const container = document.getElementById('merge-preview');

    input.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        const newFiles = files.filter(f => !mergeFiles.some(mf => mf.name === f.name));
        mergeFiles = mergeFiles.concat(newFiles);
        renderMergePreview(mergeFiles, container, (idx) => {
            mergeFiles.splice(idx, 1);
            renderMergePreview(mergeFiles, container);
        });
        document.getElementById('merge-clear-btn').style.display = 'inline-block';
        input.value = '';
    });

    document.getElementById('merge-clear-btn').addEventListener('click', () => {
        mergeFiles = [];
        renderMergePreview([], container);
        document.getElementById('merge-clear-btn').style.display = 'none';
    });

    document.getElementById('merge-preview-btn').addEventListener('click', async () => {
        if (mergeFiles.length === 0) return alert('请添加图片');
        const formData = new FormData();
        formData.append('rows', document.getElementById('merge-rows').value);
        formData.append('cols', document.getElementById('merge-cols').value);
        formData.append('cell_width', document.getElementById('cell-width').value);
        formData.append('cell_height', document.getElementById('cell-height').value);
        formData.append('padding', document.getElementById('merge-padding').value);
        
        showLoading();
        try {
            const res = await fetch(apiUrl('/preview/merge'), { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                document.getElementById('merge-grid-image').src = data.preview;
                container.classList.add('hidden');
                document.getElementById('merge-grid-preview').classList.remove('hidden');
            } else { alert(data.error); }
        } catch(e) { alert(e.message); } finally { hideLoading(); }
    });

    document.getElementById('merge-btn').addEventListener('click', async () => {
        if (mergeFiles.length === 0) return alert('请添加图片');
        const formData = new FormData();
        mergeFiles.forEach(f => formData.append('images', f));
        formData.append('rows', document.getElementById('merge-rows').value);
        formData.append('cols', document.getElementById('merge-cols').value);
        formData.append('cell_width', document.getElementById('cell-width').value);
        formData.append('cell_height', document.getElementById('cell-height').value);
        formData.append('padding', document.getElementById('merge-padding').value);

        showLoading();
        try {
            const res = await fetch(apiUrl('/merge'), { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                showMergeResult(data, 'merged-image', 'merge-result');
            } else { alert(data.error); }
        } catch(e) { alert(e.message); } finally { hideLoading(); }
    });
}

let compactFiles = [];

function initCompactMergeFunctionality() {
    const input = document.getElementById('compact-merge-images');
    const container = document.getElementById('compact-merge-preview');
    const previewBtn = document.getElementById('compact-merge-preview-btn');
    let isPreviewing = false;

    input.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        const newFiles = files.filter(f => !compactFiles.some(cf => cf.name === f.name));
        compactFiles = compactFiles.concat(newFiles);
        renderMergePreview(compactFiles, container, (idx) => {
            compactFiles.splice(idx, 1);
            renderMergePreview(compactFiles, container);
        });
        document.getElementById('compact-merge-clear-btn').style.display = 'inline-block';
        input.value = '';
    });

    document.getElementById('compact-merge-clear-btn').addEventListener('click', () => {
        compactFiles = [];
        renderMergePreview([], container);
        document.getElementById('compact-merge-clear-btn').style.display = 'none';
        if (isPreviewing) {
            exitPreview();
        }
    });

    function exitPreview() {
        container.classList.remove('hidden');
        document.getElementById('compact-merge-grid-preview').classList.add('hidden');
        previewBtn.textContent = '预览';
        isPreviewing = false;
    }

    previewBtn.addEventListener('click', async () => {
        if (isPreviewing) {
            exitPreview();
            return;
        }

        if (compactFiles.length === 0) return alert('请添加图片');
        const formData = new FormData();
        compactFiles.forEach(f => formData.append('images', f));
        const size = document.getElementById('compact-atlas-size').value;
        if(size) formData.append('atlas_size', size);
        formData.append('padding', document.getElementById('compact-padding').value);

        showLoading();
        try {
            const res = await fetch(apiUrl('/preview/merge/compact'), { method: 'POST', body: formData });
            const data = await res.json();
            if(data.success) {
                document.getElementById('compact-merge-grid-image').src = data.preview;
                container.classList.add('hidden');
                document.getElementById('compact-merge-grid-preview').classList.remove('hidden');
                previewBtn.textContent = '取消预览';
                isPreviewing = true;
            } else { alert(data.error); }
        } catch(e) { alert(e.message); } finally { hideLoading(); }
    });

    document.getElementById('compact-merge-btn').addEventListener('click', async () => {
        if (compactFiles.length === 0) return alert('请添加图片');
        const formData = new FormData();
        compactFiles.forEach(f => formData.append('images', f));
        const size = document.getElementById('compact-atlas-size').value;
        if(size) formData.append('atlas_size', size);
        formData.append('padding', document.getElementById('compact-padding').value);

        showLoading();
        try {
            const res = await fetch(apiUrl('/merge/compact'), { method: 'POST', body: formData });
            const data = await res.json();
            if(data.success) {
                showMergeResult(data, 'compact-merged-image', 'compact-merge-result');
            } else { alert(data.error); }
        } catch(e) { alert(e.message); } finally { hideLoading(); }
    });
}

function renderMergePreview(files, container, onRemove) {
    container.innerHTML = '';
    files.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const div = document.createElement('div');
            div.className = 'preview-item';
            div.innerHTML = `
                <img src="${e.target.result}">
                <div class="preview-info">${file.name}</div>
                <button class="preview-remove">&times;</button>
            `;
            div.querySelector('.preview-remove').onclick = () => onRemove(index);
            container.appendChild(div);
        };
        reader.readAsDataURL(file);
    });
}

function showMergeResult(data, imgId, containerId) {
    const img = document.getElementById(imgId);
    img.src = `${outputUrl(data.filename)}?t=${Date.now()}`;
    
    const container = document.getElementById(containerId);
    const parentLayer = container.closest('.canvas-layer');
    Array.from(parentLayer.children).forEach(c => c.classList.add('hidden'));
    container.classList.remove('hidden');

    const link = document.createElement('a');
    link.href = outputUrl(data.filename);
    link.download = data.filename;
    link.click();
}

let bgImageFile = null;
let bgImageObj = null;
let selectedColor = null;

function initToolboxFunctionality() {
    const input = document.getElementById('bg-image');
    const methodSelect = document.getElementById('bg-method');
    const zoomBtn = document.getElementById('open-zoom-modal-btn');
    const canvas = document.getElementById('bg-canvas');
    const ctx = canvas.getContext('2d');

    input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            bgImageFile = file;
            handleSingleImagePreview(file, 'bg-preview-container');

            const reader = new FileReader();
            reader.onload = (evt) => {
                const img = new Image();
                img.onload = () => {
                    bgImageObj = img;
                    document.getElementById('bg-preview-btn').disabled = false;
                    checkPickerState();
                };
                img.src = evt.target.result;
            };
            reader.readAsDataURL(file);
        }
    });

    methodSelect.addEventListener('change', () => {
        const val = methodSelect.value;
        document.getElementById('tolerance-group').style.display = (val === 'color' || val === 'picker') ? 'block' : 'none';
        document.getElementById('selected-color-group').classList.toggle('hidden', val !== 'picker');

        checkPickerState();
    });

    function checkPickerState() {
        const val = methodSelect.value;
        const clickHint = document.getElementById('click-hint');
        if (val === 'picker' && bgImageObj) {
            zoomBtn.classList.remove('hidden');
            if (clickHint) clickHint.style.display = 'block';
        } else {
            zoomBtn.classList.add('hidden');
            if (clickHint) clickHint.style.display = 'none';
        }
    }

    document.getElementById('bg-tolerance').addEventListener('input', (e) => {
        document.getElementById('tolerance-value').textContent = e.target.value;
    });

    document.getElementById('bg-preview-btn').addEventListener('click', async () => {
        if (!bgImageFile) return alert('请上传图片');
        const method = methodSelect.value;
        if (method === 'picker' && !selectedColor) return alert('请先取色');

        const formData = new FormData();
        formData.append('image', bgImageFile);
        formData.append('method', method);
        formData.append('tolerance', document.getElementById('bg-tolerance').value);
        if(selectedColor) formData.append('target_color', selectedColor);

        showLoading();
        try {
            const res = await fetch(apiUrl('/remove-bg'), { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                const resImg = document.getElementById('bg-result-image');
                resImg.src = `${outputUrl(data.filename)}?t=${Date.now()}`;
                document.getElementById('bg-preview-container').classList.add('hidden');
                document.getElementById('bg-result').classList.remove('hidden');

                const dlBtn = document.getElementById('bg-download-btn');
                dlBtn.classList.remove('hidden');
                dlBtn.onclick = () => {
                    const l = document.createElement('a');
                    l.href = outputUrl(data.filename);
                    l.download = data.filename;
                    l.click();
                };
            } else { alert(data.error); }
        } catch(e) { alert(e.message); } finally { hideLoading(); }
    });

    initZoomModal();
}

function initZoomModal() {
    const modal = document.getElementById('zoom-modal');
    const zCanvas = document.getElementById('zoom-modal-canvas');
    const zCtx = zCanvas.getContext('2d');
    let scale = 1;

    document.getElementById('open-zoom-modal-btn').addEventListener('click', () => {
        if (!bgImageObj) return;
        modal.classList.remove('hidden');
        scale = 1;
        draw();
    });

    document.getElementById('zoom-modal-close').addEventListener('click', () => modal.classList.add('hidden'));

    function draw() {
        zCanvas.width = bgImageObj.width * scale;
        zCanvas.height = bgImageObj.height * scale;
        zCtx.drawImage(bgImageObj, 0, 0, zCanvas.width, zCanvas.height);
        document.getElementById('modal-zoom-level').textContent = Math.round(scale * 100) + '%';
    }

    document.getElementById('modal-zoom-in-btn').addEventListener('click', () => {
        if(scale < 5) { scale += 0.5; draw(); }
    });
    document.getElementById('modal-zoom-out-btn').addEventListener('click', () => {
        if(scale > 0.5) { scale -= 0.5; draw(); }
    });

    zCanvas.addEventListener('click', (e) => {
        const rect = zCanvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / scale;
        const y = (e.clientY - rect.top) / scale;

        const tempC = document.createElement('canvas');
        tempC.width = 1; tempC.height = 1;
        const tempCtx = tempC.getContext('2d');
        tempCtx.drawImage(bgImageObj, -x, -y);
        const [r,g,b] = tempCtx.getImageData(0,0,1,1).data;
        
        const hex = "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase();
        
        document.getElementById('modal-selected-color-box').style.backgroundColor = hex;
        document.getElementById('modal-selected-color-hex').textContent = hex;

        modal.dataset.tempColor = hex;
    });

    document.getElementById('modal-confirm-color-btn').addEventListener('click', () => {
        const hex = modal.dataset.tempColor;
        if (!hex) return alert('请点击图片取色');
        selectedColor = hex;
        
        document.getElementById('selected-color-box').style.backgroundColor = hex;
        document.getElementById('selected-color-hex').textContent = hex;
        document.getElementById('selected-color-group').classList.remove('hidden');
        modal.classList.add('hidden');
    });

    document.getElementById('clear-color-btn').addEventListener('click', () => {
        selectedColor = null;
        document.getElementById('selected-color-group').classList.add('hidden');
    });
}

function handleSingleImagePreview(file, containerId) {
    const container = document.getElementById(containerId);
    const reader = new FileReader();
    reader.onload = (e) => {
        container.innerHTML = `<img src="${e.target.result}" class="overlay-img">`;
        updateCanvasInfo(`${file.name} (${formatFileSize(file.size)})`);
        container.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function initFileInputUX() {
    document.querySelectorAll('.hidden-input').forEach(input => {
        input.addEventListener('change', (e) => {
            const label = input.nextElementSibling;
            if (e.target.files.length > 0) {
                const count = e.target.files.length;
                const text = count === 1 ? e.target.files[0].name : `${count} files selected`;
                const iconHtml = label.querySelector('i') ? label.querySelector('i').outerHTML : '';
                label.innerHTML = `${iconHtml} ${text}`;
                label.classList.add('text-primary');
            }
        });
    });
}

let currentZoomLevel = 1.0;
const ZOOM_STEP = 0.1;
const MIN_ZOOM = 0.1;
const MAX_ZOOM = 5.0;

function initCanvasZoom() {
    const zoomInBtn = document.querySelector('.zoom-controls .tool-btn:nth-child(3)');
    const zoomOutBtn = document.querySelector('.zoom-controls .tool-btn:nth-child(1)');
    const zoomText = document.querySelector('.zoom-text');

    if (!zoomInBtn || !zoomOutBtn || !zoomText) return;

    zoomInBtn.addEventListener('click', () => {
        if (currentZoomLevel < MAX_ZOOM) {
            currentZoomLevel = Math.min(MAX_ZOOM, currentZoomLevel + ZOOM_STEP);
            applyZoom();
        }
    });

    zoomOutBtn.addEventListener('click', () => {
        if (currentZoomLevel > MIN_ZOOM) {
            currentZoomLevel = Math.max(MIN_ZOOM, currentZoomLevel - ZOOM_STEP);
            applyZoom();
        }
    });

    const viewport = document.querySelector('.canvas-viewport');
    if (viewport) {
        viewport.addEventListener('wheel', (e) => {
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                if (e.deltaY < 0) {
                    currentZoomLevel = Math.min(MAX_ZOOM, currentZoomLevel + ZOOM_STEP);
                } else {
                    currentZoomLevel = Math.max(MIN_ZOOM, currentZoomLevel - ZOOM_STEP);
                }
                applyZoom();
            }
        }, { passive: false });
    }
}

function applyZoom() {
    const zoomText = document.querySelector('.zoom-text');
    const activeLayer = document.querySelector('.canvas-layer.active');

    if (zoomText) {
        zoomText.textContent = Math.round(currentZoomLevel * 100) + '%';
    }

    if (activeLayer) {
        const images = activeLayer.querySelectorAll('img');
        images.forEach(img => {
            img.style.transform = `scale(${currentZoomLevel})`;
            img.style.transformOrigin = 'center center';
            img.style.transition = 'transform 0.15s ease-out';
        });
    }
}

function resetZoom() {
    currentZoomLevel = 1.0;
    applyZoom();
}

window.resetZoom = resetZoom;
