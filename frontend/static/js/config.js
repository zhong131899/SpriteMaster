/**
 * API 配置
 * 前后端分离模式:
 *   - 后端 API: 动态从配置文件读取
 *   - 前端页面: 动态端口
 */

// 从启动脚本生成的端口配置文件读取后端端口
let BACKEND_URL = 'http://localhost:8000';

// 尝试从 port_config.txt 读取后端端口
async function loadBackendPort() {
    try {
        const response = await fetch('/port_config.txt');
        if (response.ok) {
            const text = await response.text();
            const match = text.match(/BACKEND_PORT=(\d+)/);
            if (match) {
                BACKEND_URL = `http://localhost:${match[1]}`;
                console.log('后端端口已自动配置:', BACKEND_URL);
            }
        }
    } catch (e) {
        console.log('使用默认后端配置:', BACKEND_URL);
    }
}

// 立即加载端口配置
loadBackendPort();

/**
 * 获取当前的后端 URL (动态获取)
 * @returns {string} 后端 URL
 */
function getBackendUrl() {
    return BACKEND_URL;
}

/**
 * 构建 API 请求 URL
 * @param {string} endpoint - API 端点路径 (如 '/split', '/merge')
 * @returns {string} 完整的 API URL
 */
function apiUrl(endpoint) {
    return `${getBackendUrl()}/api${endpoint}`;
}

/**
 * 构建输出文件 URL
 * @param {string} path - 输出文件路径
 * @returns {string} 完整的输出文件 URL
 */
function outputUrl(path) {
    return `${getBackendUrl()}/output/${path}`;
}

/**
 * 构建下载 URL
 * @param {string} sessionId - 会话 ID
 * @param {string} filename - 文件名
 * @returns {string} 完整的下载 URL
 */
function downloadUrl(sessionId, filename) {
    return `${getBackendUrl()}/api/download/${sessionId}/${filename}`;
}

/**
 * 通用 API 请求函数
 * @param {string} endpoint - API 端点
 * @param {object} options - fetch 选项
 * @returns {Promise} fetch Promise
 */
async function apiRequest(endpoint, options = {}) {
    const url = apiUrl(endpoint);
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    const finalOptions = { ...defaultOptions, ...options };

    // 如果是 FormData，移除 Content-Type 让浏览器自动设置
    if (options.body instanceof FormData) {
        delete finalOptions.headers['Content-Type'];
    }

    const response = await fetch(url, finalOptions);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    return response.json();
}
