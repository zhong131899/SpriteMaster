"""
图片分割与图集合并工具 - Flask API 服务
"""
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import uuid
import time
from image_processor import ImageProcessor
from cache_manager import CacheManager

app = Flask(__name__)
# 前后端分离模式 CORS 配置 - 允许所有来源访问 API 和输出文件
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/output/*": {"origins": "*"}
})

UPLOAD_FOLDER = '../uploads'
OUTPUT_FOLDER = '../outputs'
CACHE_FOLDER = '../cache'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

processor = ImageProcessor()
cache_manager = CacheManager(cache_dir=CACHE_FOLDER)


def cleanup_before_operation():
    cache_stats = cache_manager.clean_old_cache(days=1, min_ref_count=0)
    if cache_stats['deleted'] > 0:
        print(f"清理缓存: 删除 {cache_stats['deleted']} 个文件, 释放 {cache_stats['freed_bytes'] / 1024:.1f} KB")
    clean_old_cache(app.config['OUTPUT_FOLDER'], days=1)


def save_and_cache_file(file):
    session_id = str(uuid.uuid4())
    temp_filename = f"{session_id}_{file.filename}"
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
    file.save(temp_path)
    cached_path = cache_manager.get_cached_path(temp_path)
    os.remove(temp_path)
    return cached_path, session_id


def clean_old_cache(folder_path, days=1):
    if not os.path.exists(folder_path):
        return

    current_time = time.time()
    cutoff_time = current_time - (days * 24 * 60 * 60)

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if os.path.isfile(item_path):
            file_mtime = os.path.getmtime(item_path)
            if file_mtime < cutoff_time:
                try:
                    os.remove(item_path)
                    print(f"已删除过期文件: {item_path}")
                except Exception as e:
                    print(f"删除文件失败 {item_path}: {e}")
        elif os.path.isdir(item_path):
            dir_mtime = os.path.getmtime(item_path)
            if dir_mtime < cutoff_time:
                try:
                    for root, dirs, files in os.walk(item_path, topdown=False):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                os.remove(file_path)
                            except Exception as e:
                                print(f"删除文件失败 {file_path}: {e}")
                        for dir in dirs:
                            dir_path = os.path.join(root, dir)
                            try:
                                os.rmdir(dir_path)
                            except Exception as e:
                                print(f"删除目录失败 {dir_path}: {e}")
                    os.rmdir(item_path)
                    print(f"已删除过期目录: {item_path}")
                except Exception as e:
                    print(f"删除目录失败 {item_path}: {e}")


@app.route('/')
def index():
    return jsonify({'status': 'ok', 'message': 'SpriteMaster API is running'})


@app.route('/api/split', methods=['POST'])
def split_image():
    try:
        cleanup_before_operation()

        file = request.files.get('image')
        if not file:
            return jsonify({'error': '请上传图片'}), 400

        rows = int(request.form.get('rows', 1))
        cols = int(request.form.get('cols', 1))
        margin_top = int(request.form.get('margin_top', 0))
        margin_bottom = int(request.form.get('margin_bottom', 0))
        margin_left = int(request.form.get('margin_left', 0))
        margin_right = int(request.form.get('margin_right', 0))
        prefix = request.form.get('prefix', 'sprite')

        input_path, session_id = save_and_cache_file(file)
        if not input_path:
            return jsonify({'error': '文件处理失败'}), 500
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], f"{session_id}_split")
        os.makedirs(output_dir, exist_ok=True)

        output_files = processor.split_image(
            image_path=input_path,
            output_dir=output_dir,
            rows=rows,
            cols=cols,
            margin_top=margin_top,
            margin_bottom=margin_bottom,
            margin_left=margin_left,
            margin_right=margin_right,
            prefix=prefix
        )

        return jsonify({
            'success': True,
            'count': len(output_files),
            'files': [os.path.basename(f) for f in output_files],
            'session_id': session_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/merge', methods=['POST'])
def merge_images():
    try:
        cleanup_before_operation()

        files = request.files.getlist('images')
        if not files or len(files) == 0:
            return jsonify({'error': '请上传图片'}), 400

        rows = int(request.form.get('rows', 1))
        cols = int(request.form.get('cols', 1))
        cell_width = int(request.form.get('cell_width', 64))
        cell_height = int(request.form.get('cell_height', 64))
        padding = int(request.form.get('padding', 0))

        image_paths = []
        for file in files:
            if file.filename:
                cached_path, _ = save_and_cache_file(file)
                if cached_path:
                    image_paths.append(cached_path)

        output_filename = f"{str(uuid.uuid4())}_merged.png"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        result_path = processor.merge_images(
            image_paths=image_paths,
            output_path=output_path,
            rows=rows,
            cols=cols,
            cell_width=cell_width,
            cell_height=cell_height,
            padding=padding
        )

        return jsonify({
            'success': True,
            'filename': output_filename,
            'session_id': output_filename[:36]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/merge/compact', methods=['POST'])
def merge_compact():
    try:
        cleanup_before_operation()

        files = request.files.getlist('images')
        if not files or len(files) == 0:
            return jsonify({'error': '请上传图片'}), 400

        atlas_size_str = request.form.get('atlas_size', '')
        atlas_size = int(atlas_size_str) if atlas_size_str else None
        padding = int(request.form.get('padding', 5))

        image_paths = []
        for file in files:
            if file.filename:
                cached_path, _ = save_and_cache_file(file)
                if cached_path:
                    image_paths.append(cached_path)

        output_filename = f"{str(uuid.uuid4())}_compact_merged.png"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        result_path, positions = processor.merge_compact(
            image_paths=image_paths,
            output_path=output_path,
            atlas_size=atlas_size,
            padding=padding
        )

        return jsonify({
            'success': True,
            'filename': output_filename,
            'session_id': output_filename[:36],
            'positions': positions,
            'atlas_size': atlas_size
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview/merge/compact', methods=['POST'])
def preview_compact_merge():
    try:
        files = request.files.getlist('images')
        if not files or len(files) == 0:
            return jsonify({'error': '请上传图片'}), 400

        atlas_size_str = request.form.get('atlas_size', '')
        atlas_size = int(atlas_size_str) if atlas_size_str else None
        padding = int(request.form.get('padding', 5))

        image_paths = []
        for file in files:
            if file.filename:
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}_{file.filename}")
                file.save(temp_path)
                image_paths.append(temp_path)

        preview_base64, total_width, total_height, positions = processor.preview_compact_merge(
            image_paths=image_paths,
            atlas_size=atlas_size,
            padding=padding
        )

        for temp_path in image_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return jsonify({
            'success': True,
            'preview': f"data:image/png;base64,{preview_base64}",
            'total_width': total_width,
            'total_height': total_height,
            'positions': positions,
            'atlas_size': atlas_size if atlas_size else '自动'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/output/<path:filepath>')
def get_output_file(filepath):
    if '/' in filepath:
        parts = filepath.split('/', 1)
        first_part = parts[0]
        filename = parts[1]

        if first_part.endswith('_split'):
            file_path = os.path.join(app.config['OUTPUT_FOLDER'], first_part, filename)
        else:
            file_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{first_part}_split", filename)
    else:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filepath)

    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    return "File not found", 404


@app.route('/api/download/<session_id>/<filename>')
def download_split_file(session_id, filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{session_id}_split", filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=filename, mimetype='image/png')
    return "File not found", 404


@app.route('/api/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)


@app.route('/api/preview/split', methods=['POST'])
def preview_split():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': '请上传图片'}), 400

        rows = int(request.form.get('rows', 1))
        cols = int(request.form.get('cols', 1))
        margin_top = int(request.form.get('margin_top', 0))
        margin_bottom = int(request.form.get('margin_bottom', 0))
        margin_left = int(request.form.get('margin_left', 0))
        margin_right = int(request.form.get('margin_right', 0))

        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}_{file.filename}")
        file.save(temp_path)

        try:
            preview_base64, cell_width, cell_height = processor.preview_split_grid(
                image_path=temp_path,
                rows=rows,
                cols=cols,
                margin_top=margin_top,
                margin_bottom=margin_bottom,
                margin_left=margin_left,
                margin_right=margin_right
            )

            return jsonify({
                'success': True,
                'preview': f"data:image/png;base64,{preview_base64}",
                'cell_width': cell_width,
                'cell_height': cell_height
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview/merge', methods=['POST'])
def preview_merge():
    try:
        rows = int(request.form.get('rows', 1))
        cols = int(request.form.get('cols', 1))
        cell_width = int(request.form.get('cell_width', 64))
        cell_height = int(request.form.get('cell_height', 64))
        padding = int(request.form.get('padding', 0))

        preview_base64, total_width, total_height = processor.preview_merge_grid(
            rows=rows,
            cols=cols,
            cell_width=cell_width,
            cell_height=cell_height,
            padding=padding
        )

        return jsonify({
            'success': True,
            'preview': f"data:image/png;base64,{preview_base64}",
            'total_width': total_width,
            'total_height': total_height
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect', methods=['POST'])
def detect_boundaries():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': '请上传图片'}), 400

        margin_top = int(request.form.get('margin_top', 0))
        margin_bottom = int(request.form.get('margin_bottom', 0))
        margin_left = int(request.form.get('margin_left', 0))
        margin_right = int(request.form.get('margin_right', 0))
        threshold = int(request.form.get('threshold', 10))
        min_gap = int(request.form.get('min_gap', 5))

        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}_{file.filename}")
        file.save(temp_path)

        try:
            result = processor.detect_boundaries(
                image_path=temp_path,
                margin_top=margin_top,
                margin_bottom=margin_bottom,
                margin_left=margin_left,
                margin_right=margin_right,
                threshold=threshold,
                min_gap=min_gap
            )

            return jsonify({
                'success': True,
                'rows_split': result['rows_split'],
                'cols_split': result['cols_split'],
                'rows': result['rows'],
                'cols': result['cols']
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/split/irregular', methods=['POST'])
def split_image_irregular():
    try:
        cleanup_before_operation()

        file = request.files.get('image')
        if not file:
            return jsonify({'error': '请上传图片'}), 400

        import json
        rows_split = json.loads(request.form.get('rows_split', '[0]'))
        cols_split = json.loads(request.form.get('cols_split', '[0]'))
        margin_top = int(request.form.get('margin_top', 0))
        margin_bottom = int(request.form.get('margin_bottom', 0))
        margin_left = int(request.form.get('margin_left', 0))
        margin_right = int(request.form.get('margin_right', 0))
        prefix = request.form.get('prefix', 'sprite')

        input_path, session_id = save_and_cache_file(file)
        if not input_path:
            return jsonify({'error': '文件处理失败'}), 500

        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], f"{session_id}_split")
        os.makedirs(output_dir, exist_ok=True)

        output_files = processor.split_image_irregular(
            image_path=input_path,
            output_dir=output_dir,
            rows_split=rows_split,
            cols_split=cols_split,
            margin_top=margin_top,
            margin_bottom=margin_bottom,
            margin_left=margin_left,
            margin_right=margin_right,
            prefix=prefix
        )

        return jsonify({
            'success': True,
            'count': len(output_files),
            'files': [os.path.basename(f) for f in output_files],
            'session_id': session_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview/split/irregular', methods=['POST'])
def preview_split_irregular():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': '请上传图片'}), 400

        import json
        rows_split = json.loads(request.form.get('rows_split', '[0]'))
        cols_split = json.loads(request.form.get('cols_split', '[0]'))
        margin_top = int(request.form.get('margin_top', 0))
        margin_bottom = int(request.form.get('margin_bottom', 0))
        margin_left = int(request.form.get('margin_left', 0))
        margin_right = int(request.form.get('margin_right', 0))

        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}_{file.filename}")
        file.save(temp_path)

        try:
            preview_base64, cells_info = processor.preview_split_grid_irregular(
                image_path=temp_path,
                rows_split=rows_split,
                cols_split=cols_split,
                margin_top=margin_top,
                margin_bottom=margin_bottom,
                margin_left=margin_left,
                margin_right=margin_right
            )

            return jsonify({
                'success': True,
                'preview': f"data:image/png;base64,{preview_base64}",
                'cells_info': cells_info
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/remove-bg', methods=['POST'])
def remove_background():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': '请上传图片'}), 400

        method = request.form.get('method', 'color')
        tolerance = int(request.form.get('tolerance', 30))
        target_color = request.form.get('target_color', None)

        input_path, session_id = save_and_cache_file(file)
        if not input_path:
            return jsonify({'error': '文件处理失败'}), 500

        output_filename = f"{session_id}_nobg.png"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        if method == 'color':
            result_path = processor.remove_background_color(
                image_path=input_path,
                output_path=output_path,
                tolerance=tolerance
            )
        elif method == 'picker':
            if not target_color:
                return jsonify({'error': '请先点击图片选择要去除的背景颜色'}), 400
            target_color = target_color.lstrip('#')
            rgb = tuple(int(target_color[i:i+2], 16) for i in (0, 2, 4))
            result_path = processor.remove_background_picker(
                image_path=input_path,
                output_path=output_path,
                target_color=rgb,
                tolerance=tolerance
            )
        else:
            return jsonify({'error': '不支持的方法'}), 400

        return jsonify({
            'success': True,
            'filename': output_filename,
            'session_id': session_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview/remove-bg', methods=['POST'])
def preview_remove_background():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': '请上传图片'}), 400

        method = request.form.get('method', 'color')
        tolerance = int(request.form.get('tolerance', 30))
        target_color = request.form.get('target_color', None)

        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}_{file.filename}")
        file.save(temp_path)

        try:
            if method == 'color':
                preview_base64 = processor.preview_remove_background_color(
                    image_path=temp_path,
                    tolerance=tolerance
                )
            elif method == 'picker':
                if not target_color:
                    return jsonify({'error': '请先选择要去除的背景颜色'}), 400
                target_color = target_color.lstrip('#')
                rgb = tuple(int(target_color[i:i+2], 16) for i in (0, 2, 4))
                preview_base64 = processor.preview_remove_background_picker(
                    image_path=temp_path,
                    target_color=rgb,
                    tolerance=tolerance
                )
            else:
                return jsonify({'error': '不支持的方法'}), 400

            return jsonify({
                'success': True,
                'preview': f"data:image/png;base64,{preview_base64}"
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/bboxes', methods=['POST'])
def detect_bounding_boxes():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': '请上传图片'}), 400

        margin_top = int(request.form.get('margin_top', 0))
        margin_bottom = int(request.form.get('margin_bottom', 0))
        margin_left = int(request.form.get('margin_left', 0))
        margin_right = int(request.form.get('margin_right', 0))
        threshold = int(request.form.get('threshold', 10))
        min_size = int(request.form.get('min_size', 10))

        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}_{file.filename}")
        file.save(temp_path)

        try:
            bboxes = processor.detect_bounding_boxes(
                image_path=temp_path,
                margin_top=margin_top,
                margin_bottom=margin_bottom,
                margin_left=margin_left,
                margin_right=margin_right,
                threshold=threshold,
                min_size=min_size
            )

            return jsonify({
                'success': True,
                'bboxes': bboxes,
                'count': len(bboxes)
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/split/bboxes', methods=['POST'])
def split_image_by_bboxes():
    try:
        cleanup_before_operation()

        file = request.files.get('image')
        if not file:
            return jsonify({'error': '请上传图片'}), 400

        import json
        bboxes = json.loads(request.form.get('bboxes', '[]'))
        prefix = request.form.get('prefix', 'sprite')

        uniform_size = request.form.get('uniform_size', 'false').lower() == 'true'
        uniform_width_str = request.form.get('uniform_width', '')
        uniform_height_str = request.form.get('uniform_height', '')

        uniform_width = int(uniform_width_str) if uniform_width_str and uniform_width_str.isdigit() else None
        uniform_height = int(uniform_height_str) if uniform_height_str and uniform_height_str.isdigit() else None

        input_path, session_id = save_and_cache_file(file)
        if not input_path:
            return jsonify({'error': '文件处理失败'}), 500

        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], f"{session_id}_split")
        os.makedirs(output_dir, exist_ok=True)

        output_files = processor.split_image_by_bboxes(
            image_path=input_path,
            output_dir=output_dir,
            bboxes=bboxes,
            prefix=prefix,
            uniform_size=uniform_size,
            uniform_width=uniform_width,
            uniform_height=uniform_height
        )

        return jsonify({
            'success': True,
            'count': len(output_files),
            'files': [os.path.basename(f) for f in output_files],
            'session_id': session_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview/split/bboxes', methods=['POST'])
def preview_split_bboxes():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': '请上传图片'}), 400

        margin_top = int(request.form.get('margin_top', 0))
        margin_bottom = int(request.form.get('margin_bottom', 0))
        margin_left = int(request.form.get('margin_left', 0))
        margin_right = int(request.form.get('margin_right', 0))
        threshold = int(request.form.get('threshold', 10))
        min_size = int(request.form.get('min_size', 10))

        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}_{file.filename}")
        file.save(temp_path)

        try:
            bboxes = processor.detect_bounding_boxes(
                image_path=temp_path,
                margin_top=margin_top,
                margin_bottom=margin_bottom,
                margin_left=margin_left,
                margin_right=margin_right,
                threshold=threshold,
                min_size=min_size
            )

            preview_base64 = processor.preview_bounding_boxes(
                image_path=temp_path,
                bboxes=bboxes
            )

            return jsonify({
                'success': True,
                'preview': f"data:image/png;base64,{preview_base64}",
                'bboxes': bboxes,
                'count': len(bboxes)
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


import sys

if __name__ == '__main__':
    # 解析命令行参数
    port = 8000
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--port' and i + 1 < len(sys.argv) - 1:
            port = int(sys.argv[i + 2])

    print(f"启动服务: http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
