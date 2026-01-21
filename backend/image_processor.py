"""
图片处理核心模块
提供图片分割和图集合并功能
"""
from PIL import Image, ImageDraw
import os
import io
import base64
from typing import List, Tuple, Optional
import numpy as np
from scipy import ndimage


class ImageProcessor:
    """图片处理器类"""

    @staticmethod
    def split_image(
        image_path: str,
        output_dir: str,
        rows: int,
        cols: int,
        margin_top: int = 0,
        margin_bottom: int = 0,
        margin_left: int = 0,
        margin_right: int = 0,
        prefix: str = "sprite"
    ) -> List[str]:
        """
        将一张图集分割成多个独立图片

        Args:
            image_path: 输入图片路径
            output_dir: 输出目录
            rows: 分割行数
            cols: 分割列数
            margin_top: 上边距（像素）
            margin_bottom: 下边距（像素）
            margin_left: 左边距（像素）
            margin_right: 右边距（像素）
            prefix: 输出文件名前缀

        Returns:
            分割后的图片文件路径列表
        """
        # 打开原始图片
        img = Image.open(image_path)
        img_width, img_height = img.size

        # 计算有效区域（去除边距）
        effective_width = img_width - margin_left - margin_right
        effective_height = img_height - margin_top - margin_bottom

        if effective_width <= 0 or effective_height <= 0:
            raise ValueError("边距设置过大，超出了图片尺寸")

        # 计算每个单元格的尺寸
        cell_width = effective_width // cols
        cell_height = effective_height // rows

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        output_files = []

        # 分割图片
        for row in range(rows):
            for col in range(cols):
                # 计算当前单元格的坐标
                left = margin_left + col * cell_width
                top = margin_top + row * cell_height
                right = left + cell_width
                bottom = top + cell_height

                # 裁剪图片
                sprite = img.crop((left, top, right, bottom))

                # 保存图片
                filename = f"{prefix}_{row}_{col}.png"
                output_path = os.path.join(output_dir, filename)
                sprite.save(output_path)
                output_files.append(output_path)

        return output_files

    @staticmethod
    def detect_boundaries(image_path: str, margin_top: int = 0, margin_bottom: int = 0,
                         margin_left: int = 0, margin_right: int = 0,
                         threshold: int = 10, min_gap: int = 5) -> dict:
        """
        使用投影分析法智能检测图片分割边界
        支持透明背景和纯色背景的图集

        Args:
            image_path: 输入图片路径
            margin_top: 上边距（像素）
            margin_bottom: 下边距（像素）
            margin_left: 左边距（像素）
            margin_right: 右边距（像素）
            threshold: 判断为"非空白"的透明度/颜色差异阈值 (0-255)
            min_gap: 分割线之间的最小间隔（像素），避免过度分割

        Returns:
            包含 rows_split 和 cols_split 的字典
        """
        img = Image.open(image_path).convert("RGBA")
        img_width, img_height = img.size

        effective_width = img_width - margin_left - margin_right
        effective_height = img_height - margin_top - margin_bottom

        if effective_width <= 0 or effective_height <= 0:
            raise ValueError("边距设置过大，超出了图片尺寸")

        effective_region = img.crop((margin_left, margin_top, img_width - margin_right, img_height - margin_bottom))

        arr = np.array(effective_region)
        height, width = arr.shape[:2]

        rgb = arr[:, :, :3].astype(np.float32)

        background_color = ImageProcessor._sample_corners_background(rgb)

        alpha = arr[:, :, 3]
        is_content_alpha = alpha > threshold

        rgb = arr[:, :, :3].astype(np.float32)

        color_diff_euclidean = np.sqrt(np.sum((rgb - background_color) ** 2, axis=2))
        color_diff_max = np.max(np.abs(rgb - background_color), axis=2)
        color_diff_mean = np.mean(np.abs(rgb - background_color), axis=2)

        global_diff_mean = np.mean(color_diff_euclidean)
        global_diff_std = np.std(color_diff_euclidean)

        adaptive_threshold = max(threshold, int(global_diff_mean * 0.3))

        try:
            blurred = ndimage.gaussian_filter(rgb, sigma=1)
            diff_from_blurred = np.abs(rgb - blurred).max(axis=2)
            has_detail = diff_from_blurred > threshold * 0.2
        except:
            has_detail = np.zeros((height, width), dtype=bool)

        is_content_bg_diff = (
            (color_diff_euclidean > adaptive_threshold) |
            (color_diff_max > adaptive_threshold * 0.5) |
            (color_diff_mean > adaptive_threshold * 0.3)
        )

        is_content = np.logical_or.reduce([
            is_content_alpha,
            is_content_bg_diff,
            has_detail
        ])

        horizontal_projection = np.sum(is_content, axis=1)
        vertical_projection = np.sum(is_content, axis=0)

        def find_splits(projection, min_gap):
            if len(projection) == 0:
                return [0]

            window_size = max(5, min_gap // 2)
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(projection, kernel, mode='same')

            p_min = np.min(smoothed)
            p_max = np.max(smoothed)
            p_median = np.median(smoothed)
            p_mean = np.mean(smoothed)
            p_std = np.std(smoothed)

            variation = p_max - p_min

            if variation < p_mean * 0.2:
                search_window = max(min_gap, len(smoothed) // 30)
                local_mins = []

                for i in range(search_window, len(smoothed) - search_window):
                    left_mean = np.mean(smoothed[i-search_window:i])
                    right_mean = np.mean(smoothed[i:i+search_window])
                    center_val = smoothed[i]

                    if center_val < left_mean * 0.92 and center_val < right_mean * 0.92:
                        local_mins.append((i, center_val))

                local_mins.sort(key=lambda x: x[1])

                splits = [0]
                if local_mins:
                    candidates = local_mins[:min(len(local_mins), len(smoothed) // (min_gap * 2))]
                    candidates.sort(key=lambda x: x[0])

                    for pos, val in candidates:
                        if all(abs(pos - s) >= min_gap for s in splits):
                            splits.append(pos)

                splits.append(int(len(projection)))
                splits = sorted(set(splits))

                if len(splits) <= 2:
                    gradient = np.diff(smoothed)
                    second_gradient = np.diff(gradient)

                    local_minima = []
                    for i in range(1, len(second_gradient)):
                        if second_gradient[i-1] < 0 and second_gradient[i] > 0:
                            local_minima.append(i)

                    if len(local_minima) > 0:
                        merged_mins = [local_minima[0]]
                        for pos in local_minima[1:]:
                            if pos - merged_mins[-1] >= min_gap:
                                merged_mins.append(pos)

                        splits = [0] + merged_mins + [len(projection)]
                        splits = sorted(set(splits))

                    if len(splits) <= 2:
                        threshold_low = p_mean - p_std * 0.3
                        low_regions = []

                        i = 0
                        while i < len(smoothed):
                            if smoothed[i] < threshold_low:
                                start = i
                                while i < len(smoothed) and smoothed[i] < threshold_low:
                                    i += 1
                                end = i - 1
                                if end - start >= min_gap // 2:
                                    low_regions.append((start + end) // 2)
                            else:
                                i += 1

                        if len(low_regions) > 0:
                            splits = [0] + low_regions + [int(len(projection))]
                            splits = sorted(set(splits))

                if len(splits) >= 2:
                    return [int(x) for x in splits]

            content_threshold = max(p_min + (p_max - p_min) * 0.15, p_median * 0.4)
            has_content = smoothed > content_threshold

            splits = [0]

            i = 0
            while i < len(projection):
                while i < len(projection) and not has_content[i]:
                    i += 1

                if i >= len(projection):
                    break

                content_start = i

                while i < len(projection) and has_content[i]:
                    i += 1

                content_end = i - 1

                gap_start = i
                while i < len(projection) and not has_content[i]:
                    i += 1

                gap_end = i - 1

                if gap_end - gap_start >= min_gap:
                    split_pos = (gap_start + gap_end) // 2
                    splits.append(split_pos)
                elif gap_end > gap_start and i < len(projection):
                    region = smoothed[gap_start:gap_end + 1]
                    if len(region) > 0:
                        min_idx = int(gap_start + np.argmin(region))
                        splits.append(min_idx)

            if len(splits) ==1 or splits[-1] != len(projection):
                splits.append(int(len(projection)))

            splits = sorted(set(splits))

            if len(splits) < 2:
                return [0, int(len(projection))]

            return [int(x) for x in splits]

        def find_splits_by_components(mask, min_gap_rows=5, min_gap_cols=5):
            labeled, num = ndimage.label(mask)
            if num <= 1:
                return [0, mask.shape[0]], [0, mask.shape[1]]

            objects = ndimage.find_objects(labeled)
            components = []

            total_area = mask.shape[0] * mask.shape[1]
            min_area = max(1, int(total_area * 0.001))

            for sl in objects:
                if sl is None:
                    continue
                y_slice, x_slice = sl
                y1, y2 = y_slice.start, y_slice.stop
                x1, x2 = x_slice.start, x_slice.stop
                area = (y2 - y1) * (x2 - x1)
                if area < min_area:
                    continue
                cy = (y1 + y2) / 2.0
                cx = (x1 + x2) / 2.0
                components.append({
                    "y1": y1, "y2": y2, "x1": x1, "x2": x2,
                    "cy": cy, "cx": cx
                })

            if len(components) == 0:
                return [0, int(mask.shape[0])], [0, int(mask.shape[1])]

            comps_by_y = sorted(components, key=lambda c: c["cy"])
            ys = [c["cy"] for c in comps_by_y]
            if len(ys) == 1:
                row_groups = [comps_by_y]
            else:
                gaps_y = np.diff(ys)
                mean_gap_y = np.mean(gaps_y)
                threshold_y = mean_gap_y * 1.5
                row_groups = []
                current_group = [comps_by_y[0]]
                for i, gap in enumerate(gaps_y):
                    if gap > threshold_y:
                        row_groups.append(current_group)
                        current_group = [comps_by_y[i + 1]]
                    else:
                        current_group.append(comps_by_y[i + 1])
                if current_group:
                    row_groups.append(current_group)

            row_tops = []
            row_bottoms = []
            for g in row_groups:
                row_tops.append(min(c["y1"] for c in g))
                row_bottoms.append(max(c["y2"] for c in g))

            row_splits = [0]
            for i in range(len(row_groups) - 1):
                split_pos = int((row_bottoms[i] + row_tops[i + 1]) / 2)
                if split_pos - row_splits[-1] >= min_gap_rows:
                    row_splits.append(split_pos)
            row_splits.append(mask.shape[0])

            comps_by_x = sorted(components, key=lambda c: c["cx"])
            xs = [c["cx"] for c in comps_by_x]
            if len(xs) == 1:
                col_groups = [comps_by_x]
            else:
                gaps_x = np.diff(xs)
                mean_gap_x = np.mean(gaps_x)
                threshold_x = mean_gap_x * 1.5
                col_groups = []
                current_group = [comps_by_x[0]]
                for i, gap in enumerate(gaps_x):
                    if gap > threshold_x:
                        col_groups.append(current_group)
                        current_group = [comps_by_x[i + 1]]
                    else:
                        current_group.append(comps_by_x[i + 1])
                if current_group:
                    col_groups.append(current_group)

            col_lefts = []
            col_rights = []
            for g in col_groups:
                col_lefts.append(min(c["x1"] for c in g))
                col_rights.append(max(c["x2"] for c in g))

            col_splits = [0]
            for i in range(len(col_groups) - 1):
                split_pos = int((col_rights[i] + col_lefts[i + 1]) / 2)
                if split_pos - col_splits[-1] >= min_gap_cols:
                    col_splits.append(split_pos)
            col_splits.append(mask.shape[1])

            if len(row_splits) < 2:
                row_splits = [0, int(mask.shape[0])]
            if len(col_splits) < 2:
                col_splits = [0, int(mask.shape[1])]

            return [int(x) for x in sorted(set(row_splits))], [int(x) for x in sorted(set(col_splits))]

        row_splits = find_splits(horizontal_projection, min_gap)
        col_splits = find_splits(vertical_projection, min_gap)

        if len(row_splits) == 2 and len(col_splits) == 2:
            strict_threshold = global_diff_mean + global_diff_std * 0.5
            strong_fg = color_diff_euclidean > strict_threshold
            if np.count_nonzero(strong_fg) < (height * width * 0.01):
                strong_fg = is_content

            comp_rows, comp_cols = find_splits_by_components(strong_fg, min_gap_rows=min_gap, min_gap_cols=min_gap)
            if len(comp_rows) > 2 or len(comp_cols) > 2:
                row_splits, col_splits = comp_rows, comp_cols

        return {
            'rows_split': row_splits,
            'cols_split': col_splits,
            'rows': len(row_splits) - 1,
            'cols': len(col_splits) - 1
        }

    @staticmethod
    def split_image_irregular(
        image_path: str,
        output_dir: str,
        rows_split: List[int],
        cols_split: List[int],
        margin_top: int = 0,
        margin_bottom: int = 0,
        margin_left: int = 0,
        margin_right: int = 0,
        prefix: str = "sprite"
    ) -> List[str]:
        """
        按照不规则网格分割图片

        Args:
            image_path: 输入图片路径
            output_dir: 输出目录
            rows_split: 行分割点列表 [0, 100, 250, 400]
            cols_split: 列分割点列表 [0, 80, 200, 300]
            margin_top: 上边距（像素）
            margin_bottom: 下边距（像素）
            margin_left: 左边距（像素）
            margin_right: 右边距（像素）
            prefix: 输出文件名前缀

        Returns:
            分割后的图片文件路径列表
        """
        img = Image.open(image_path)
        img_width, img_height = img.size

        os.makedirs(output_dir, exist_ok=True)

        output_files = []

        for row in range(len(rows_split) - 1):
            for col in range(len(cols_split) - 1):
                left = margin_left + cols_split[col]
                top = margin_top + rows_split[row]
                right = margin_left + cols_split[col + 1]
                bottom = margin_top + rows_split[row + 1]

                if right <= left or bottom <= top:
                    continue

                sprite = img.crop((left, top, right, bottom))

                filename = f"{prefix}_{row}_{col}.png"
                output_path = os.path.join(output_dir, filename)
                sprite.save(output_path)
                output_files.append(output_path)

        return output_files

    @staticmethod
    def merge_images(
        image_paths: List[str],
        output_path: str,
        rows: int,
        cols: int,
        cell_width: int,
        cell_height: int,
        padding: int = 0,
        background_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ) -> str:
        """
        将多个图片合并成一张图集

        Args:
            image_paths: 输入图片路径列表
            output_path: 输出图片路径
            rows: 图集行数
            cols: 图集列数
            cell_width: 每个单元格宽度
            cell_height: 每个单元格高度
            padding: 单元格间距（像素）
            background_color: 背景色（RGBA）

        Returns:
            合并后的图片文件路径
        """
        total_width = cols * cell_width + (cols - 1) * padding
        total_height = rows * cell_height + (rows - 1) * padding

        if len(background_color) == 4:
            output_img = Image.new("RGBA", (total_width, total_height), background_color)
        else:
            output_img = Image.new("RGB", (total_width, total_height), background_color)

        max_sprites = rows * cols
        valid_paths = image_paths[:max_sprites]

        for index, img_path in enumerate(valid_paths):
            if not os.path.exists(img_path):
                continue

            row = index // cols
            col = index % cols

            x = col * (cell_width + padding)
            y = row * (cell_height + padding)

            sprite = Image.open(img_path)

            if sprite.mode != "RGBA":
                sprite = sprite.convert("RGBA")

            sprite = sprite.resize((cell_width, cell_height), Image.Resampling.LANCZOS)

            output_img.paste(sprite, (x, y), sprite)

        output_img.save(output_path)
        return output_path

    @staticmethod
    def merge_compact(
        image_paths: List[str],
        output_path: str,
        atlas_size: Optional[int] = None,
        padding: int = 5,
        background_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ) -> Tuple[str, List[Tuple[int, int, int, int]]]:
        """
        紧凑合并多个图片到一张图集（使用MaxRects算法）

        Args:
            image_paths: 输入图片路径列表
            output_path: 输出图片路径
            atlas_size: 图集大小（正方形边长，64-2048，2的幂次），None表示自动选择
            padding: 图片间隔（像素），默认为5
            background_color: 背景色（RGBA）

        Returns:
            (输出路径, 位置信息列表)
            位置信息: [(x, y, width, height), ...] 表示每张图片在图集中的位置
        """
        if not image_paths:
            raise ValueError("没有提供图片")

        images = []
        sizes = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path)
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            images.append(img)
            sizes.append(img.size)

        if not images:
            raise ValueError("没有有效的图片")

        if atlas_size is None:
            atlas_size = ImageProcessor._calculate_atlas_size(sizes, padding)

        atlas_size = max(64, min(2048, atlas_size))
        if atlas_size < 64:
            atlas_size = 64
        elif atlas_size <= 64:
            atlas_size = 64
        elif atlas_size <= 128:
            atlas_size = 128
        elif atlas_size <= 256:
            atlas_size = 256
        elif atlas_size <= 512:
            atlas_size = 512
        elif atlas_size <= 1024:
            atlas_size = 1024
        elif atlas_size <= 2048:
            atlas_size = 2048

        positions = ImageProcessor._max_rects_bin_packing(sizes, atlas_size, atlas_size, padding)

        while positions is None and atlas_size < 2048:
            if atlas_size < 128:
                atlas_size = 128
            elif atlas_size < 256:
                atlas_size = 256
            elif atlas_size < 512:
                atlas_size = 512
            elif atlas_size < 1024:
                atlas_size = 1024
            elif atlas_size < 2048:
                atlas_size = 2048
            else:
                break

            positions = ImageProcessor._max_rects_bin_packing(sizes, atlas_size, atlas_size, padding)

        if positions is None:
            raise ValueError(f"无法将所有图片放入2048x2048的图集中，请减少图片数量或使用更大的图集大小")

        if len(background_color) == 4:
            output_img = Image.new("RGBA", (atlas_size, atlas_size), background_color)
        else:
            output_img = Image.new("RGB", (atlas_size, atlas_size), background_color)

        for i, (img, (x, y)) in enumerate(zip(images, positions)):
            output_img.paste(img, (x, y), img)

        output_img.save(output_path)

        full_positions = [(pos[0], pos[1], sizes[i][0], sizes[i][1]) for i, pos in enumerate(positions)]

        return output_path, full_positions

    @staticmethod
    def _calculate_atlas_size(sizes: List[Tuple[int, int]], padding: int) -> int:
        total_area = sum((w + padding) * (h + padding) for w, h in sizes)

        max_width = max((w + padding for w, h in sizes), default=0)
        max_height = max((h + padding for w, h in sizes), default=0)

        for size in [64, 128, 256, 512, 1024, 2048]:
            if max_width > size or max_height > size:
                continue
            if total_area <= size * size * 0.5:
                return size

        return 2048

    @staticmethod
    def _max_rects_bin_packing(
        sizes: List[Tuple[int, int]],
        bin_width: int,
        bin_height: int,
        padding: int
    ) -> Optional[List[Tuple[int, int]]]:
        padded_sizes = [(w + padding, h + padding) for w, h in sizes]

        indexed_sizes = [(i, padded_sizes[i]) for i in range(len(padded_sizes))]
        indexed_sizes.sort(key=lambda x: x[1][0] * x[1][1], reverse=True)

        free_rects = [(0, 0, bin_width, bin_height)]

        position_dict = {}

        for original_index, (width, height) in indexed_sizes:
            best_rect = None
            best_free_rect_index = -1
            min_hole_area = float('inf')

            for i, (rx, ry, rw, rh) in enumerate(free_rects):
                if rw < width or rh < height:
                    continue

                hole_area = (rw - width) * rh + (rh - height) * width

                if hole_area < min_hole_area:
                    min_hole_area = hole_area
                    best_rect = (rx, ry, width, height)
                    best_free_rect_index = i

            if best_rect is None:
                return None

            position_dict[original_index] = (best_rect[0], best_rect[1])

            free_rects = ImageProcessor._split_free_rectangle(free_rects, best_free_rect_index, best_rect)

        positions = [position_dict[i] for i in range(len(sizes))]
        return positions

    @staticmethod
    def _split_free_rectangle(
        free_rects: List[Tuple[int, int, int, int]],
        rect_index: int,
        placed_rect: Tuple[int, int, int, int]
    ) -> List[Tuple[int, int, int, int]]:
        x, y, w, h = placed_rect
        rx, ry, rw, rh = free_rects.pop(rect_index)

        new_rects = []

        if y + h < ry + rh:
            new_rects.append((rx, y + h, rw, ry + rh - (y + h)))

        if y > ry:
            new_rects.append((rx, ry, rw, y - ry))

        if x > rx:
            new_rects.append((rx, y, x - rx, h))

        if x + w < rx + rw:
            new_rects.append((x + w, y, rx + rw - (x + w), h))

        filtered_rects = []
        for rect in new_rects + free_rects:
            is_contained = False
            for other in new_rects + free_rects:
                if rect != other and rect[0] >= other[0] and rect[1] >= other[1] and \
                   rect[0] + rect[2] <= other[0] + other[2] and rect[1] + rect[3] <= other[1] + other[3]:
                    is_contained = True
                    break
            if not is_contained and rect[2] > 0 and rect[3] > 0:
                filtered_rects.append(rect)

        return filtered_rects

    @staticmethod
    def preview_compact_merge(
        image_paths: List[str],
        atlas_size: Optional[int] = None,
        padding: int = 5
    ) -> Tuple[str, int, int, List[Tuple[int, int, int, int]]]:
        images = []
        sizes = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path)
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            images.append(img)
            sizes.append(img.size)

        if not images:
            raise ValueError("没有有效的图片")

        if atlas_size is None:
            atlas_size = ImageProcessor._calculate_atlas_size(sizes, padding)

        atlas_size = max(64, min(2048, atlas_size))
        if atlas_size < 64:
            atlas_size = 64
        elif atlas_size <= 64:
            atlas_size = 64
        elif atlas_size <= 128:
            atlas_size = 128
        elif atlas_size <= 256:
            atlas_size = 256
        elif atlas_size <= 512:
            atlas_size = 512
        elif atlas_size <= 1024:
            atlas_size = 1024
        elif atlas_size <= 2048:
            atlas_size = 2048

        positions = ImageProcessor._max_rects_bin_packing(sizes, atlas_size, atlas_size, padding)

        while positions is None and atlas_size < 2048:
            if atlas_size < 128:
                atlas_size = 128
            elif atlas_size < 256:
                atlas_size = 256
            elif atlas_size < 512:
                atlas_size = 512
            elif atlas_size < 1024:
                atlas_size = 1024
            elif atlas_size < 2048:
                atlas_size = 2048
            else:
                break

            positions = ImageProcessor._max_rects_bin_packing(sizes, atlas_size, atlas_size, padding)

        if positions is None:
            raise ValueError(f"无法将所有图片放入2048x2048的图集中，请减少图片数量或使用更大的图集大小")

        output_img = Image.new("RGBA", (atlas_size, atlas_size), (0, 0, 0, 0))

        for i, (img, (x, y)) in enumerate(zip(images, positions)):
            output_img.paste(img, (x, y), img)

        buffered = io.BytesIO()
        output_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        full_positions = [(pos[0], pos[1], sizes[i][0], sizes[i][1]) for i, pos in enumerate(positions)]

        return img_base64, atlas_size, atlas_size, full_positions

    @staticmethod
    def get_image_info(image_path: str) -> Tuple[int, int]:
        with Image.open(image_path) as img:
            return img.size

    @staticmethod
    def preview_split_grid(
        image_path: str,
        rows: int,
        cols: int,
        margin_top: int = 0,
        margin_bottom: int = 0,
        margin_left: int = 0,
        margin_right: int = 0
    ) -> str:
        img = Image.open(image_path).convert("RGBA")
        img_width, img_height = img.size

        effective_width = img_width - margin_left - margin_right
        effective_height = img_height - margin_top - margin_bottom

        if effective_width <= 0 or effective_height <= 0:
            raise ValueError("边距设置过大，超出了图片尺寸")

        cell_width = effective_width // cols
        cell_height = effective_height // rows

        preview = img.copy()
        draw = ImageDraw.Draw(preview)

        if margin_top > 0:
            draw.rectangle([0, 0, img_width, margin_top], fill=(255, 0, 0, 50))
        if margin_bottom > 0:
            draw.rectangle([0, img_height - margin_bottom, img_width, img_height], fill=(255, 0, 0, 50))
        if margin_left > 0:
            draw.rectangle([0, 0, margin_left, img_height], fill=(255, 0, 0, 50))
        if margin_right > 0:
            draw.rectangle([img_width - margin_right, 0, img_width, img_height], fill=(255, 0, 0, 50))

        for col in range(1, cols):
            x = margin_left + col * cell_width
            draw.line([(x, margin_top), (x, img_height - margin_bottom)], fill=(0, 255, 0, 200), width=2)

        for row in range(1, rows):
            y = margin_top + row * cell_height
            draw.line([(margin_left, y), (img_width - margin_right, y)], fill=(0, 255, 0, 200), width=2)

        draw.rectangle(
            [margin_left, margin_top, img_width - margin_right, img_height - margin_bottom],
            outline=(0, 255, 0, 200),
            width=3
        )

        buffer = io.BytesIO()
        preview.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8'), cell_width, cell_height

    @staticmethod
    def preview_merge_grid(
        rows: int,
        cols: int,
        cell_width: int,
        cell_height: int,
        padding: int = 0
    ) -> str:
        total_width = cols * cell_width + (cols - 1) * padding
        total_height = rows * cell_height + (rows - 1) * padding

        preview = Image.new("RGBA", (total_width, total_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(preview)

        for row in range(rows):
            for col in range(cols):
                x = col * (cell_width + padding)
                y = row * (cell_height + padding)

                draw.rectangle(
                    [x, y, x + cell_width, y + cell_height],
                    outline=(100, 100, 100, 255),
                    width=2
                )

                text = f"{row * cols + col + 1}"
                text_x = x + cell_width // 2 - len(text) * 3
                text_y = y + cell_height // 2 - 5
                for i, char in enumerate(text):
                    draw.text((text_x + i * 6, text_y), char, fill=(0, 0, 0, 200))

        buffer = io.BytesIO()
        preview.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8'), total_width, total_height

    @staticmethod
    def preview_split_grid_irregular(
        image_path: str,
        rows_split: List[int],
        cols_split: List[int],
        margin_top: int = 0,
        margin_bottom: int = 0,
        margin_left: int = 0,
        margin_right: int = 0
    ) -> str:
        img = Image.open(image_path).convert("RGBA")
        img_width, img_height = img.size

        preview = img.copy()
        draw = ImageDraw.Draw(preview)

        if margin_top > 0:
            draw.rectangle([0, 0, img_width, margin_top], fill=(255, 0, 0, 50))
        if margin_bottom > 0:
            draw.rectangle([0, img_height - margin_bottom, img_width, img_height], fill=(255, 0, 0, 50))
        if margin_left > 0:
            draw.rectangle([0, 0, margin_left, img_height], fill=(255, 0, 0, 50))
        if margin_right > 0:
            draw.rectangle([img_width - margin_right, 0, img_width, img_height], fill=(255, 0, 0, 50))

        cells_info = []

        for row_idx in range(len(rows_split) - 1):
            for col_idx in range(len(cols_split) - 1):
                x1 = margin_left + cols_split[col_idx]
                y1 = margin_top + rows_split[row_idx]
                x2 = margin_left + cols_split[col_idx + 1]
                y2 = margin_top + rows_split[row_idx + 1]

                cells_info.append({
                    'row': row_idx,
                    'col': col_idx,
                    'x': int(x1),
                    'y': int(y1),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1)
                })

                draw.rectangle([x1, y1, x2, y2], outline=(0, 150, 255, 220), width=2)

                text = f"{row_idx},{col_idx}"
                try:
                    from PIL import ImageFont
                    font = ImageFont.load_default()
                except:
                    font = None

                text_x = x1 + 5
                text_y = y1 + 5
                if font:
                    draw.text((text_x, text_y), text, fill=(255, 255, 0, 255), font=font)
                else:
                    for i, char in enumerate(text):
                        draw.text((text_x + i * 6, text_y), char, fill=(255, 255, 0, 255))

        draw.rectangle(
            [margin_left, margin_top, img_width - margin_right, img_height - margin_bottom],
            outline=(0, 150, 255, 220),
            width=3
        )

        buffer = io.BytesIO()
        preview.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8'), cells_info

    @staticmethod
    def _sample_corners_background(img_array: np.ndarray) -> Tuple[int, int, int]:
        height, width = img_array.shape[:2]
        corner_size = min(20, height // 3, width // 3)

        corners = np.vstack([
            img_array[:corner_size, :corner_size].reshape(-1, 3),
            img_array[:corner_size, -corner_size:].reshape(-1, 3),
            img_array[-corner_size:, :corner_size].reshape(-1, 3),
            img_array[-corner_size:, -corner_size:].reshape(-1, 3)
        ])

        quantized = np.round(corners / 5) * 5
        unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
        return tuple(int(c) for c in unique_colors[np.argmax(counts)])

    @staticmethod
    def _remove_background_by_color(
        img_array: np.ndarray,
        target_color: Tuple[int, int, int],
        tolerance: int = 30
    ) -> np.ndarray:
        alpha = img_array[:, :, 3] if img_array.shape[2] == 4 else np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.uint8) * 255
        rgb = img_array[:, :, :3].astype(np.float32)
        color_diff = np.sqrt(np.sum((rgb - np.array(target_color)) ** 2, axis=2))
        alpha[color_diff <= tolerance] = 0
        img_array[:, :, 3] = alpha
        return img_array

    @staticmethod
    def remove_background_color(
        image_path: str,
        output_path: str,
        tolerance: int = 30
    ) -> str:
        """
        使用颜色阈值法去除背景（自动检测背景色）

        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径
            tolerance: 颜色容差（0-255）

        Returns:
            输出图片路径
        """
        img = Image.open(image_path).convert("RGBA")
        img_array = np.array(img)
        background_color = ImageProcessor._sample_corners_background(img_array[:, :, :3])
        result = Image.fromarray(ImageProcessor._remove_background_by_color(img_array, background_color, tolerance), 'RGBA')
        result.save(output_path)
        return output_path

    @staticmethod
    def remove_background_picker(
        image_path: str,
        output_path: str,
        target_color: Tuple[int, int, int],
        tolerance: int = 30
    ) -> str:
        """
        使用手动选择的颜色去除背景

        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径
            target_color: 要去除的目标颜色 (R, G, B)
            tolerance: 颜色容差（0-255）

        Returns:
            输出图片路径
        """
        img = Image.open(image_path).convert("RGBA")
        img_array = np.array(img)
        result = Image.fromarray(ImageProcessor._remove_background_by_color(img_array, target_color, tolerance), 'RGBA')
        result.save(output_path)
        return output_path

    @staticmethod
    def preview_remove_background_color(
        image_path: str,
        tolerance: int = 30
    ) -> str:
        img = Image.open(image_path).convert("RGBA")
        img_array = np.array(img)
        background_color = ImageProcessor._sample_corners_background(img_array[:, :, :3])
        result = Image.fromarray(ImageProcessor._remove_background_by_color(img_array, background_color, tolerance), 'RGBA')
        buffer = io.BytesIO()
        result.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    @staticmethod
    def preview_remove_background_picker(
        image_path: str,
        target_color: Tuple[int, int, int],
        tolerance: int = 30
    ) -> str:
        img = Image.open(image_path).convert("RGBA")
        img_array = np.array(img)
        result = Image.fromarray(ImageProcessor._remove_background_by_color(img_array, target_color, tolerance), 'RGBA')
        buffer = io.BytesIO()
        result.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    @staticmethod
    def detect_bounding_boxes(
        image_path: str,
        margin_top: int = 0,
        margin_bottom: int = 0,
        margin_left: int = 0,
        margin_right: int = 0,
        threshold: int = 10,
        min_size: int = 10,
        max_gap: int = 20
    ) -> List[dict]:
        img = Image.open(image_path).convert("RGBA")
        img_width, img_height = img.size

        effective_width = img_width - margin_left - margin_right
        effective_height = img_height - margin_top - margin_bottom

        if effective_width <= 0 or effective_height <= 0:
            raise ValueError("边距设置过大，超出了图片尺寸")

        effective_region = img.crop((
            margin_left, margin_top,
            img_width - margin_right, img_height - margin_bottom
        ))

        arr = np.array(effective_region)
        height, width = arr.shape[:2]

        alpha = arr[:, :, 3]
        is_content_alpha = alpha > threshold

        rgb = arr[:, :, :3].astype(np.float32)

        background_color = ImageProcessor._sample_corners_background(rgb)

        color_diff = np.sqrt(np.sum((rgb - background_color) ** 2, axis=2))

        global_diff_mean = np.mean(color_diff)
        adaptive_threshold = max(threshold, int(global_diff_mean * 0.3))

        is_content = np.logical_or(
            is_content_alpha,
            color_diff > adaptive_threshold
        )

        labeled, num_features = ndimage.label(is_content)

        if num_features == 0:
            return []

        objects = ndimage.find_objects(labeled)
        boxes = []

        total_area = width * height
        min_area = max(min_size * min_size, int(total_area * 0.001))

        for label_idx, slice_obj in enumerate(objects, start=1):
            if slice_obj is None:
                continue

            y_slice, x_slice = slice_obj
            y1, y2 = y_slice.start, y_slice.stop
            x1, x2 = x_slice.start, x_slice.stop

            box_width = x2 - x1
            box_height = y2 - y1
            area = box_width * box_height

            if area < min_area:
                continue

            component_mask = (labeled == label_idx)
            component_region = is_content & component_mask

            rows = np.any(component_region, axis=1)
            cols = np.any(component_region, axis=0)

            if not np.any(rows) or not np.any(cols):
                continue

            rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
            cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

            padding = 2
            rmin = max(0, rmin - padding)
            cmin = max(0, cmin - padding)
            rmax = min(height, rmax + padding + 1)
            cmax = min(width, cmax + padding + 1)

            boxes.append({
                'x': int(margin_left + cmin),
                'y': int(margin_top + rmin),
                'width': int(cmax - cmin),
                'height': int(rmax - rmin)
            })

        if not boxes:
            return [{
                'x': int(margin_left),
                'y': int(margin_top),
                'width': int(effective_width),
                'height': int(effective_height)
            }]

        def merge_overlapping_boxes(boxes_list, overlap_threshold=0.5):
            if not boxes_list:
                return []

            boxes_list = sorted(boxes_list, key=lambda b: b['width'] * b['height'], reverse=True)

            merged = []
            while boxes_list:
                current = boxes_list.pop(0)
                merged.append(current)

                i = 0
                while i < len(boxes_list):
                    other = boxes_list[i]

                    x_overlap = max(0, min(current['x'] + current['width'], other['x'] + other['width']) -
                                   max(current['x'], other['x']))
                    y_overlap = max(0, min(current['y'] + current['height'], other['y'] + other['height']) -
                                   max(current['y'], other['y']))

                    overlap_area = x_overlap * y_overlap
                    current_area = current['width'] * current['height']
                    other_area = other['width'] * other['height']
                    union_area = current_area + other_area - overlap_area

                    if union_area > 0 and overlap_area / union_area > overlap_threshold:
                        new_x = int(min(current['x'], other['x']))
                        new_y = int(min(current['y'], other['y']))
                        new_width = int(max(current['x'] + current['width'], other['x'] + other['width']) - new_x)
                        new_height = int(max(current['y'] + current['height'], other['y'] + other['height']) - new_y)

                        current['x'] = new_x
                        current['y'] = new_y
                        current['width'] = new_width
                        current['height'] = new_height

                        boxes_list.pop(i)
                    else:
                        i += 1

            return merged

        boxes = merge_overlapping_boxes(boxes)

        boxes.sort(key=lambda b: (b['y'], b['x']))

        if boxes:
            y_centers = [b['y'] + b['height'] / 2 for b in boxes]
            if len(y_centers) > 1:
                y_gaps = np.diff(sorted(y_centers))
                mean_y_gap = np.mean(y_gaps) if len(y_gaps) > 0 else 1
                y_threshold = mean_y_gap * 0.6

                rows = []
                current_row = [boxes[0]]
                current_y = boxes[0]['y']

                for box in boxes[1:]:
                    if abs(box['y'] - current_y) < y_threshold:
                        current_row.append(box)
                    else:
                        rows.append(sorted(current_row, key=lambda b: b['x']))
                        current_row = [box]
                        current_y = box['y']
                rows.append(sorted(current_row, key=lambda b: b['x']))

                for row_idx, row in enumerate(rows):
                    for col_idx, box in enumerate(row):
                        box['row'] = row_idx
                        box['col'] = col_idx
            else:
                boxes[0]['row'] = 0
                boxes[0]['col'] = 0

        return boxes

    @staticmethod
    def split_image_by_bboxes(
        image_path: str,
        output_dir: str,
        bboxes: List[dict],
        prefix: str = "sprite",
        uniform_size: bool = False,
        uniform_width: Optional[int] = None,
        uniform_height: Optional[int] = None
    ) -> List[str]:
        img = Image.open(image_path)
        img_width, img_height = img.size

        os.makedirs(output_dir, exist_ok=True)

        output_files = []

        sorted_bboxes = sorted(bboxes, key=lambda b: (b.get('row', 0), b.get('col', 0)))

        target_width = uniform_width
        target_height = uniform_height

        if uniform_size:
            if target_width is None or target_height is None:
                max_w = 0
                max_h = 0
                for bbox in sorted_bboxes:
                    max_w = max(max_w, bbox['width'])
                    max_h = max(max_h, bbox['height'])
                if target_width is None:
                    target_width = max_w
                if target_height is None:
                    target_height = max_h

        for idx, bbox in enumerate(sorted_bboxes):
            x = bbox['x']
            y = bbox['y']
            width = bbox['width']
            height = bbox['height']

            if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
                continue

            if width <= 0 or height <= 0:
                continue

            sprite = img.crop((x, y, x + width, y + height))

            if uniform_size and target_width and target_height:
                sprite = ImageProcessor._normalize_size(sprite, target_width, target_height)

            row = bbox.get('row', idx)
            col = bbox.get('col', 0)
            filename = f"{prefix}_{row}_{col}.png"
            output_path = os.path.join(output_dir, filename)
            sprite.save(output_path)
            output_files.append(output_path)

        return output_files

    @staticmethod
    def _normalize_size(
        image: Image.Image,
        target_width: int,
        target_height: int
    ) -> Image.Image:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        img_width, img_height = image.size

        if img_width == target_width and img_height == target_height:
            return image

        result = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))

        paste_x = (target_width - img_width) // 2
        paste_y = (target_height - img_height) // 2

        if img_width > target_width or img_height > target_height:
            crop_left = max(0, (img_width - target_width) // 2)
            crop_top = max(0, (img_height - target_height) // 2)
            crop_right = min(img_width, crop_left + target_width)
            crop_bottom = min(img_height, crop_top + target_height)

            cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))

            paste_x = (target_width - cropped.width) // 2
            paste_y = (target_height - cropped.height) // 2
            result.paste(cropped, (paste_x, paste_y), cropped)
        else:
            result.paste(image, (paste_x, paste_y), image)

        return result

    @staticmethod
    def preview_bounding_boxes(
        image_path: str,
        bboxes: List[dict]
    ) -> str:
        img = Image.open(image_path).convert("RGBA")
        preview = img.copy()
        draw = ImageDraw.Draw(preview)

        colors = [
            (255, 0, 0, 200),
            (0, 255, 0, 200),
            (0, 0, 255, 200),
            (255, 255, 0, 200),
            (255, 0, 255, 200),
            (0, 255, 255, 200),
        ]

        for idx, bbox in enumerate(bboxes):
            color = colors[idx % len(colors)]
            x, y = bbox['x'], bbox['y']
            w, h = bbox['width'], bbox['height']

            draw.rectangle(
                [x, y, x + w, y + h],
                outline=color,
                width=3
            )

            label = f"{bbox.get('row', '?')},{bbox.get('col', '?')}"
            try:
                from PIL import ImageFont
                font = ImageFont.load_default()
            except:
                font = None

            text_x = x + 5
            text_y = y + 5
            if font:
                draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=font)
            else:
                for i, char in enumerate(label):
                    draw.text((text_x + i * 6, text_y), char, fill=(255, 255, 255, 255))

        buffer = io.BytesIO()
        preview.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
