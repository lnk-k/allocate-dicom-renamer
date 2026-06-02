# DICOM 序列文件夹重命名工具

这个仓库包含 `rename.py`，用于递归扫描 DICOM 序列文件夹，并根据序列号重命名文件夹。同时，它会把可能属于 localizer/scout 的序列单独标记出来。

## 使用方法

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 修改 `rename.py` 里的 `DATA_DIR`，让它指向你本地的 `data` 文件夹。

3. 建议先设置 `DRY_RUN = True` 预览重命名结果。确认输出没有问题后，再把它改成 `False`，正式重命名文件夹并生成 JSON 日志。

```bash
python rename.py
```

本地 DICOM 数据和生成的输出文件已经在 `.gitignore` 中忽略，不会被提交到 GitHub。
