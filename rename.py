import os
import json
import pydicom
import SimpleITK as sitk


# =========================================================
# 1. 改这里：写到 data 总文件夹
# =========================================================

DATA_DIR = "/你的/data/路径"

# 第一次先 True，只预览不真正改名
DRY_RUN = False

# 少于这个数量的 DICOM 文件，认为是 localizer/scout
MIN_SERIES_IMAGES = 10


# =========================================================
# 2. 读取 DICOM：pydicom + SimpleITK 双保险
# =========================================================

def try_pydicom(path):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return ds
    except Exception:
        return None


def try_sitk_metadata(path):
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        def get_tag(tag):
            if reader.HasMetaDataKey(tag):
                return reader.GetMetaData(tag)
            return ""

        return {
            "SeriesNumber": get_tag("0020|0011"),
            "SeriesDescription": get_tag("0008|103e"),
            "ImageType": get_tag("0008|0008"),
            "SeriesInstanceUID": get_tag("0020|000e"),
        }
    except Exception:
        return None


def is_probably_dicom(path):
    name = os.path.basename(path)

    # 跳过 macOS 生成的隐藏垃圾文件
    if name.startswith("._"):
        return False

    if try_pydicom(path) is not None:
        return True

    if try_sitk_metadata(path) is not None:
        return True

    return False


# =========================================================
# 3. 在一个乱码文件夹里递归找 DICOM
# =========================================================

def get_all_dicoms_recursive(folder):
    dicom_files = []

    for root, dirs, files in os.walk(folder):
        for name in sorted(files):
            if name.startswith("._"):
                continue

            path = os.path.join(root, name)

            if not os.path.isfile(path):
                continue

            if is_probably_dicom(path):
                dicom_files.append(path)

    return dicom_files


def get_series_info(folder):
    dicom_files = get_all_dicoms_recursive(folder)

    if len(dicom_files) == 0:
        return None

    first_file = dicom_files[0]

    # 优先用 pydicom 读
    ds = try_pydicom(first_file)

    if ds is not None:
        series_number = getattr(ds, "SeriesNumber", None)
        series_description = getattr(ds, "SeriesDescription", "")
        image_type = getattr(ds, "ImageType", "")
        series_uid = getattr(ds, "SeriesInstanceUID", "")

        return {
            "folder": folder,
            "dicom_count": len(dicom_files),
            "example_file": first_file,
            "reader": "pydicom",
            "series_number": series_number,
            "series_description": str(series_description),
            "image_type": str(image_type),
            "series_uid": str(series_uid),
        }

    # 如果 pydicom 失败，用 SimpleITK 读 tag
    meta = try_sitk_metadata(first_file)

    if meta is not None:
        return {
            "folder": folder,
            "dicom_count": len(dicom_files),
            "example_file": first_file,
            "reader": "SimpleITK",
            "series_number": meta["SeriesNumber"],
            "series_description": meta["SeriesDescription"],
            "image_type": meta["ImageType"],
            "series_uid": meta["SeriesInstanceUID"],
        }

    return None


# =========================================================
# 4. 命名规则
# =========================================================

def parse_series_number(series_number):
    if series_number is None:
        return None

    text = str(series_number).strip()

    if text == "":
        return None

    try:
        return int(float(text))
    except Exception:
        return None


def is_localizer_like(info):
    text = (
        str(info.get("series_description", "")) + " " +
        str(info.get("image_type", ""))
    ).lower()

    keywords = [
        "localizer",
        "scout",
        "topogram",
        "surview",
        "locator",
    ]

    if info["dicom_count"] < MIN_SERIES_IMAGES:
        return True

    for kw in keywords:
        if kw in text:
            return True

    return False


def make_unique_path(parent, base_name, old_path=None):
    candidate = os.path.join(parent, base_name)

    # 如果目标路径就是自己，允许
    if old_path is not None and os.path.abspath(candidate) == os.path.abspath(old_path):
        return candidate

    if not os.path.exists(candidate):
        return candidate

    idx = 1

    while True:
        candidate = os.path.join(parent, f"{base_name}_{idx:02d}")

        if old_path is not None and os.path.abspath(candidate) == os.path.abspath(old_path):
            return candidate

        if not os.path.exists(candidate):
            return candidate

        idx += 1


# =========================================================
# 5. 处理单个 case
# =========================================================

def process_one_case(case_dir):
    case_name = os.path.basename(case_dir)

    print("\n" + "#" * 100)
    print(f"开始处理 case: {case_name}")
    print("#" * 100)

    child_folders = [
        os.path.join(case_dir, name)
        for name in sorted(os.listdir(case_dir))
        if os.path.isdir(os.path.join(case_dir, name))
    ]

    print(f"在 {case_name} 下发现 {len(child_folders)} 个子文件夹")

    case_logs = []
    fallback_idx = 1

    for folder in child_folders:
        old_name = os.path.basename(folder)

        print("\n" + "-" * 80)
        print("原文件夹:", old_name)

        info = get_series_info(folder)

        if info is None:
            print("没有读到 DICOM，跳过")
            continue

        print("读取方式:", info["reader"])
        print("DICOM 数量:", info["dicom_count"])
        print("示例 DICOM:", info["example_file"])
        print("SeriesNumber:", info["series_number"])
        print("SeriesDescription:", info["series_description"])
        print("ImageType:", info["image_type"])

        sn = parse_series_number(info["series_number"])

        if sn is None:
            number_part = f"unknown_{fallback_idx:03d}"
            fallback_idx += 1
        else:
            number_part = f"{sn:03d}"

        if is_localizer_like(info):
            new_base = f"localizer{number_part}"
        else:
            new_base = f"series{number_part}"

        new_path = make_unique_path(case_dir, new_base, old_path=folder)
        new_name = os.path.basename(new_path)

        print("新文件夹:", new_name)

        log_item = {
            "case": case_name,
            "old_path": folder,
            "new_path": new_path,
            "old_name": old_name,
            "new_name": new_name,
            "reader": info["reader"],
            "dicom_count": info["dicom_count"],
            "example_file": info["example_file"],
            "SeriesNumber": str(info["series_number"]),
            "SeriesDescription": info["series_description"],
            "ImageType": info["image_type"],
            "SeriesInstanceUID": info["series_uid"],
        }

        case_logs.append(log_item)

        if not DRY_RUN:
            if os.path.abspath(folder) != os.path.abspath(new_path):
                os.rename(folder, new_path)

    # 每个 case 单独保存日志
    case_log_path = os.path.join(case_dir, "rename_series_log.json")

    with open(case_log_path, "w", encoding="utf-8") as f:
        json.dump(case_logs, f, indent=4, ensure_ascii=False)

    print("\ncase 日志保存到:")
    print(case_log_path)

    return case_logs


# =========================================================
# 6. 批量处理所有 case
# =========================================================

def main():
    data_dir = os.path.abspath(DATA_DIR)

    print("DATA_DIR:", data_dir)
    print("存在:", os.path.exists(data_dir))
    print("是文件夹:", os.path.isdir(data_dir))

    case_dirs = [
        os.path.join(data_dir, name)
        for name in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, name))
    ]

    print(f"\n在 data 文件夹下发现 {len(case_dirs)} 个 case")

    all_logs = []

    for case_dir in case_dirs:
        case_logs = process_one_case(case_dir)
        all_logs.extend(case_logs)

    # 总日志保存在 data 根目录
    summary_log_path = os.path.join(data_dir, "rename_all_cases_log.json")

    with open(summary_log_path, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 100)

    if DRY_RUN:
        print("当前 DRY_RUN=True，只预览，不真正改名。")
        print("确认输出正确后，把 DRY_RUN 改成 False 再运行。")
    else:
        print("已完成所有 case 的重命名。")

    print("总日志保存到:")
    print(summary_log_path)


if __name__ == "__main__":
    main()
