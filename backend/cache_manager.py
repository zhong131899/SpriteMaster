"""
文件缓存管理器 - 基于哈希去重
使用 SHA256 哈希值实现文件去重和引用计数
"""
import hashlib
import os
import json
import time
from threading import Lock


class CacheManager:
    """文件缓存管理器"""

    def __init__(self, cache_dir='cache', metadata_file='cache_metadata.json'):
        self.cache_dir = cache_dir
        self.metadata_file = os.path.join(cache_dir, metadata_file)
        self.lock = Lock()

        os.makedirs(cache_dir, exist_ok=True)
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_metadata(self):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _calculate_hash(self, file_path):
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def get_cached_path(self, file_path, extension=None):
        if not os.path.exists(file_path):
            return None

        file_hash = self._calculate_hash(file_path)

        with self.lock:
            if extension is None:
                _, original_ext = os.path.splitext(file_path)
                extension = original_ext or '.bin'

            cached_filename = f"{file_hash}{extension}"
            cached_path = os.path.join(self.cache_dir, cached_filename)

            if file_hash in self.metadata:
                self.metadata[file_hash]['ref_count'] += 1
                self.metadata[file_hash]['last_accessed'] = time.time()
                self._save_metadata()

                if not os.path.exists(cached_path):
                    if 'original_path' in self.metadata[file_hash] and os.path.exists(self.metadata[file_hash]['original_path']):
                        import shutil
                        shutil.copy2(self.metadata[file_hash]['original_path'], cached_path)
                    else:
                        self.metadata[file_hash]['ref_count'] -= 1
                        self._save_metadata()
                        return None
            else:
                import shutil
                shutil.copy2(file_path, cached_path)

                self.metadata[file_hash] = {
                    'original_filename': os.path.basename(file_path),
                    'original_path': file_path,
                    'cached_path': cached_path,
                    'ref_count': 1,
                    'created_at': time.time(),
                    'last_accessed': time.time(),
                    'file_size': os.path.getsize(cached_path)
                }
                self._save_metadata()

            return cached_path

    def release_path(self, cached_path):
        if not cached_path or not os.path.exists(cached_path):
            return

        cached_filename = os.path.basename(cached_path)
        file_hash = os.path.splitext(cached_filename)[0]

        with self.lock:
            if file_hash in self.metadata:
                self.metadata[file_hash]['ref_count'] -= 1
                self.metadata[file_hash]['last_accessed'] = time.time()
                self._save_metadata()

    def clean_old_cache(self, days=1, min_ref_count=0):
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)

        stats = {
            'deleted': 0,
            'kept': 0,
            'freed_bytes': 0
        }

        with self.lock:
            entries_to_delete = []

            for file_hash, meta in self.metadata.items():
                should_delete = (
                    meta['ref_count'] <= min_ref_count and
                    meta['last_accessed'] < cutoff_time
                )

                if should_delete:
                    entries_to_delete.append(file_hash)
                else:
                    stats['kept'] += 1

            for file_hash in entries_to_delete:
                meta = self.metadata[file_hash]
                cached_path = meta['cached_path']

                try:
                    if os.path.exists(cached_path):
                        stats['freed_bytes'] += meta['file_size']
                        os.remove(cached_path)

                    del self.metadata[file_hash]
                    stats['deleted'] += 1
                except OSError as e:
                    print(f"删除缓存文件失败 {cached_path}: {e}")

            if entries_to_delete:
                self._save_metadata()

        return stats

    def get_stats(self):
        total_files = len(self.metadata)
        total_refs = sum(meta['ref_count'] for meta in self.metadata.values())
        total_size = sum(meta['file_size'] for meta in self.metadata.values())

        return {
            'total_files': total_files,
            'total_references': total_refs,
            'total_size_bytes': total_size,
            'cache_dir': self.cache_dir
        }

    def clear_all(self):
        with self.lock:
            for file_hash, meta in self.metadata.items():
                cached_path = meta['cached_path']
                try:
                    if os.path.exists(cached_path):
                        os.remove(cached_path)
                except OSError as e:
                    print(f"删除缓存文件失败 {cached_path}: {e}")

            self.metadata = {}
            self._save_metadata()

    def get_original_path(self, cached_path):
        cached_filename = os.path.basename(cached_path)
        file_hash = os.path.splitext(cached_filename)[0]

        with self.lock:
            if file_hash in self.metadata:
                return self.metadata[file_hash]['original_path']
        return None
