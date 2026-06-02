# DICOM 序列整理与批量配准 QC 工具

这个仓库包含两个主要脚本：

- `rename.py`：递归扫描 DICOM 序列文件夹，并根据序列号重命名文件夹。同时，它会把可能属于 localizer/scout 的序列单独标记出来。
- `register_qc.py`：批量读取同一病例下的 `seriesXXX`，进行配准前 header 筛选、identity/rigid 对比、无标签 QC，并导出 nnU-Net 多通道输入。

## 安装

建议先创建虚拟环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

## DICOM 文件夹重命名

修改 `rename.py` 里的 `DATA_DIR`，让它指向你本地的 `data` 文件夹。

建议先设置 `DRY_RUN = True` 预览重命名结果。确认输出没有问题后，再把它改成 `False`，正式重命名文件夹并生成 JSON 日志。

```bash
python rename.py
```

## 批量配准和 QC

病例目录建议整理为：

```text
data/
  CASE001/
    series003/
    series005/
    series006/
```

运行：

```bash
python register_qc.py \
  --case-dir data/CASE001 \
  --output-dir registration_qc_output \
  --case-id CASE001
```

脚本会自动选择切片数最多的序列作为 fixed image，然后处理其余 `seriesXXX`。

主要输出：

```text
registration_qc_output/
  imagesTr/
  uncropped/
  qc_png/
  logs/
```

- `logs/*_header_compare.csv`：配准前 header 筛选结果；
- `logs/*_qc_results.csv`：identity 和 rigid 的 QC 指标对比；
- `qc_png/`：fixed、moving、红绿叠加和差异图；
- `imagesTr/`：最终保留下来的 nnU-Net 多通道输入。

默认情况下，`PASS` 和 `WARNING` 都会保留，只有 `FAIL` 会排除。如果只想保留 `PASS`：

```bash
python register_qc.py \
  --case-dir data/CASE001 \
  --output-dir registration_qc_output \
  --case-id CASE001 \
  --only-keep-qc-pass
```

本地 DICOM 数据和生成的输出文件已经在 `.gitignore` 中忽略，不会被提交到 GitHub。
