import os
import csv
import json
import math
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


# =========================================================
# 1. 默认参数
# =========================================================

CASE_DIR = "data/CASE001"
OUTPUT_DIR = "registration_qc_output"

DEFAULT_PIXEL_VALUE = -1024.0
BODY_THRESHOLD = -500
CROP_MARGIN = 10

# 配准前 header 筛选阈值
MIN_SLICES = 10
HEADER_PASS_Z_OVERLAP = 0.60
HEADER_WARNING_Z_OVERLAP = 0.40
HEADER_PASS_DIRECTION_DIFF = 0.01
HEADER_WARNING_DIRECTION_DIFF = 0.05
HEADER_PASS_SPACING_REL_DIFF = 0.30
HEADER_WARNING_SPACING_REL_DIFF = 0.80

# 配准后 QC 阈值
QC_PASS_BODY_DICE = 0.70
QC_FAIL_BODY_DICE = 0.50

QC_PASS_COMMON_RATIO = 0.50
QC_FAIL_COMMON_RATIO = 0.30

# 质心距离只作为 warning，不作为 fail
QC_PASS_CENTROID_DIST = 60.0
QC_WARNING_CENTROID_DIST = 120.0

# 是否允许 header WARNING 也进入配准
ALLOW_HEADER_WARNING_TO_REGISTER = True

# 建议 False：PASS 和 WARNING 都保留，FAIL 排除
ONLY_KEEP_QC_PASS = False

# rigid 和 identity 比较阈值
MI_DROP_TOL = 1e-6
NCC_DROP_TOL = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch register DICOM CT series in one case and export QC results."
    )

    parser.add_argument(
        "--case-dir",
        default=CASE_DIR,
        help="病例文件夹，下面应包含 seriesXXX 子文件夹。",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="输出文件夹。",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="nnU-Net case 名称；默认使用 case-dir 的文件夹名。",
    )
    parser.add_argument(
        "--only-keep-qc-pass",
        action="store_true",
        help="只保留 QC PASS 的序列；默认保留 PASS 和 WARNING。",
    )
    parser.add_argument(
        "--no-header-warning",
        action="store_true",
        help="不允许 header WARNING 的序列进入配准。",
    )
    parser.add_argument(
        "--default-pixel-value",
        type=float,
        default=DEFAULT_PIXEL_VALUE,
        help="重采样外部区域填充值，CT 默认 -1024。",
    )
    parser.add_argument(
        "--body-threshold",
        type=float,
        default=BODY_THRESHOLD,
        help="身体区域粗分割阈值，CT 默认 -500。",
    )
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=CROP_MARGIN,
        help="共同有效区域裁剪时额外保留的体素边缘。",
    )

    return parser.parse_args()


# =========================================================
# 2. 基础工具函数
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_series_dirs(case_dir):
    series_dirs = []

    for name in sorted(os.listdir(case_dir)):
        path = os.path.join(case_dir, name)

        if not os.path.isdir(path):
            continue

        if name.lower().startswith("series"):
            series_dirs.append(path)

    return series_dirs


def read_dicom_series(dicom_dir):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)

    if not series_ids:
        raise RuntimeError(f"没有在该文件夹中找到 DICOM series: {dicom_dir}")

    if len(series_ids) > 1:
        print(f"警告：{dicom_dir} 中发现多个 series，默认读取第一个：")
        for i, sid in enumerate(series_ids):
            print(f"  [{i}] {sid}")

    series_id = series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)

    reader.SetFileNames(file_names)
    image = reader.Execute()
    image = sitk.Cast(image, sitk.sitkFloat32)

    return image


# =========================================================
# 3. header 空间信息提取
# =========================================================

def get_physical_corners(image):
    size = image.GetSize()

    corners_index = [
        (0, 0, 0),
        (size[0] - 1, 0, 0),
        (0, size[1] - 1, 0),
        (0, 0, size[2] - 1),
        (size[0] - 1, size[1] - 1, 0),
        (size[0] - 1, 0, size[2] - 1),
        (0, size[1] - 1, size[2] - 1),
        (size[0] - 1, size[1] - 1, size[2] - 1),
    ]

    return [image.TransformIndexToPhysicalPoint(idx) for idx in corners_index]


def get_axis_ranges(image):
    corners = get_physical_corners(image)

    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    zs = [p[2] for p in corners]

    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "z_min": min(zs),
        "z_max": max(zs),
    }


def overlap_1d(a_min, a_max, b_min, b_max):
    left = max(a_min, b_min)
    right = min(a_max, b_max)
    return max(0.0, right - left)


def compute_overlap_ratio(fixed_range, moving_range, axis="z"):
    f_min = fixed_range[f"{axis}_min"]
    f_max = fixed_range[f"{axis}_max"]
    m_min = moving_range[f"{axis}_min"]
    m_max = moving_range[f"{axis}_max"]

    f_len = abs(f_max - f_min)
    m_len = abs(m_max - m_min)

    overlap = overlap_1d(f_min, f_max, m_min, m_max)

    ratio_fixed = overlap / f_len if f_len > 0 else 0.0
    ratio_moving = overlap / m_len if m_len > 0 else 0.0

    return overlap, ratio_fixed, ratio_moving


def direction_diff(direction_a, direction_b):
    return max(abs(a - b) for a, b in zip(direction_a, direction_b))


def spacing_relative_diff(spacing_a, spacing_b):
    diffs = []

    for a, b in zip(spacing_a, spacing_b):
        denom = max(abs(a), abs(b), 1e-8)
        diffs.append(abs(a - b) / denom)

    return max(diffs)


def origin_distance(origin_a, origin_b):
    return math.sqrt(
        (origin_a[0] - origin_b[0]) ** 2 +
        (origin_a[1] - origin_b[1]) ** 2 +
        (origin_a[2] - origin_b[2]) ** 2
    )


def get_series_info(series_dir):
    image = read_dicom_series(series_dir)
    ranges = get_axis_ranges(image)

    return {
        "series_name": os.path.basename(series_dir),
        "series_dir": series_dir,
        "image": image,
        "size": image.GetSize(),
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
        "num_slices": image.GetSize()[2],
        "x_min": ranges["x_min"],
        "x_max": ranges["x_max"],
        "y_min": ranges["y_min"],
        "y_max": ranges["y_max"],
        "z_min": ranges["z_min"],
        "z_max": ranges["z_max"],
    }


def select_fixed_by_max_slices(series_infos):
    return sorted(series_infos, key=lambda x: x["num_slices"], reverse=True)[0]


def judge_header_status(row):
    if row["moving_num_slices"] < MIN_SLICES:
        return "FAIL", "too_few_slices"

    if row["z_overlap_ratio_fixed"] < HEADER_WARNING_Z_OVERLAP:
        return "FAIL", "z_overlap_too_low"

    if row["direction_diff"] > HEADER_WARNING_DIRECTION_DIFF:
        return "FAIL", "direction_diff_too_large"

    if row["spacing_rel_diff"] > HEADER_WARNING_SPACING_REL_DIFF:
        return "FAIL", "spacing_diff_too_large"

    warnings = []

    if row["z_overlap_ratio_fixed"] < HEADER_PASS_Z_OVERLAP:
        warnings.append("z_overlap_warning")

    if row["direction_diff"] > HEADER_PASS_DIRECTION_DIFF:
        warnings.append("direction_warning")

    if row["spacing_rel_diff"] > HEADER_PASS_SPACING_REL_DIFF:
        warnings.append("spacing_warning")

    if warnings:
        return "WARNING", ";".join(warnings)

    return "PASS", "ok"


def compare_header_to_fixed(fixed_info, moving_info):
    z_overlap, z_ratio_fixed, z_ratio_moving = compute_overlap_ratio(
        fixed_info,
        moving_info,
        axis="z"
    )

    d_diff = direction_diff(fixed_info["direction"], moving_info["direction"])
    s_diff = spacing_relative_diff(fixed_info["spacing"], moving_info["spacing"])
    o_dist = origin_distance(fixed_info["origin"], moving_info["origin"])

    row = {
        "fixed_series": fixed_info["series_name"],
        "moving_series": moving_info["series_name"],

        "fixed_num_slices": fixed_info["num_slices"],
        "moving_num_slices": moving_info["num_slices"],

        "fixed_size": str(fixed_info["size"]),
        "moving_size": str(moving_info["size"]),

        "fixed_spacing": str(fixed_info["spacing"]),
        "moving_spacing": str(moving_info["spacing"]),

        "fixed_origin": str(fixed_info["origin"]),
        "moving_origin": str(moving_info["origin"]),

        "fixed_direction": str(fixed_info["direction"]),
        "moving_direction": str(moving_info["direction"]),

        "fixed_z_min": fixed_info["z_min"],
        "fixed_z_max": fixed_info["z_max"],
        "moving_z_min": moving_info["z_min"],
        "moving_z_max": moving_info["z_max"],

        "z_overlap_mm": z_overlap,
        "z_overlap_ratio_fixed": z_ratio_fixed,
        "z_overlap_ratio_moving": z_ratio_moving,

        "direction_diff": d_diff,
        "spacing_rel_diff": s_diff,
        "origin_distance_mm": o_dist,
    }

    status, reason = judge_header_status(row)
    row["header_status"] = status
    row["header_reason"] = reason

    return row


# =========================================================
# 4. 配准与重采样
# =========================================================

def rigid_registration(fixed, moving):
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_f,
        moving_f,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration_method = sitk.ImageRegistrationMethod()

    registration_method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50
    )

    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )

    registration_method.SetOptimizerScalesFromPhysicalShift()

    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(fixed_f, moving_f)

    metric_value = registration_method.GetMetricValue()
    stop_condition = registration_method.GetOptimizerStopConditionDescription()

    return final_transform, metric_value, stop_condition


def resample_moving_to_fixed(fixed, moving, transform):
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    registered = sitk.Resample(
        moving_f,
        fixed,
        transform,
        sitk.sitkLinear,
        DEFAULT_PIXEL_VALUE,
        moving_f.GetPixelID()
    )

    return registered


def identity_resample_moving_to_fixed(fixed, moving):
    identity = sitk.Transform(3, sitk.sitkIdentity)
    return resample_moving_to_fixed(fixed, moving, identity)


# =========================================================
# 5. 无标签 QC 指标
# =========================================================

def image_to_array(image):
    return sitk.GetArrayFromImage(image).astype(np.float32)


def body_mask_array(image, threshold=BODY_THRESHOLD):
    arr = image_to_array(image)
    return arr > threshold


def dice_binary(mask_a, mask_b):
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)

    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()

    if denom == 0:
        return 0.0

    return 2.0 * inter / denom


def common_ratio(mask_fixed, mask_moving):
    common = np.logical_and(mask_fixed, mask_moving)

    fixed_sum = mask_fixed.sum()
    moving_sum = mask_moving.sum()

    ratio_fixed = common.sum() / fixed_sum if fixed_sum > 0 else 0.0
    ratio_moving = common.sum() / moving_sum if moving_sum > 0 else 0.0

    return ratio_fixed, ratio_moving


def centroid_physical(image, mask):
    indices_zyx = np.argwhere(mask)

    if indices_zyx.shape[0] == 0:
        return None

    centroid_zyx = indices_zyx.mean(axis=0)

    idx_xyz = (
        int(round(centroid_zyx[2])),
        int(round(centroid_zyx[1])),
        int(round(centroid_zyx[0]))
    )

    return image.TransformIndexToPhysicalPoint(idx_xyz)


def centroid_distance_mm(image_a, mask_a, image_b, mask_b):
    ca = centroid_physical(image_a, mask_a)
    cb = centroid_physical(image_b, mask_b)

    if ca is None or cb is None:
        return float("inf")

    return math.sqrt(
        (ca[0] - cb[0]) ** 2 +
        (ca[1] - cb[1]) ** 2 +
        (ca[2] - cb[2]) ** 2
    )


def ncc_metric(image_a, image_b, mask=None):
    a = image_to_array(image_a)
    b = image_to_array(image_b)

    if a.shape != b.shape:
        return None

    if mask is not None and mask.sum() > 10:
        a = a[mask]
        b = b[mask]
    else:
        a = a.reshape(-1)
        b = b.reshape(-1)

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    a = a - a.mean()
    b = b - b.mean()

    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())

    if denom == 0:
        return 0.0

    return float((a * b).sum() / denom)


def mutual_information_np(image_a, image_b, mask=None, bins=64):
    a = image_to_array(image_a)
    b = image_to_array(image_b)

    if a.shape != b.shape:
        return None

    if mask is not None and mask.sum() > 10:
        a = a[mask]
        b = b[mask]
    else:
        a = a.reshape(-1)
        b = b.reshape(-1)

    if a.size < 10:
        return 0.0

    a_low, a_high = np.percentile(a, [1, 99])
    b_low, b_high = np.percentile(b, [1, 99])

    if a_high <= a_low or b_high <= b_low:
        return 0.0

    a = np.clip(a, a_low, a_high)
    b = np.clip(b, b_low, b_high)

    hist_2d, _, _ = np.histogram2d(a, b, bins=bins)

    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]

    nz = pxy > 0

    mi = np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz]))

    return float(mi)


def header_same_after_resample(fixed, registered):
    return (
        fixed.GetSize() == registered.GetSize()
        and fixed.GetSpacing() == registered.GetSpacing()
        and fixed.GetOrigin() == registered.GetOrigin()
        and fixed.GetDirection() == registered.GetDirection()
    )


def compute_qc_metrics(fixed, moving_registered):
    header_ok = header_same_after_resample(fixed, moving_registered)

    fixed_mask = body_mask_array(fixed)
    moving_mask = body_mask_array(moving_registered)
    common_mask = np.logical_and(fixed_mask, moving_mask)

    body_dice = dice_binary(fixed_mask, moving_mask)
    cr_fixed, cr_moving = common_ratio(fixed_mask, moving_mask)
    c_dist = centroid_distance_mm(fixed, fixed_mask, moving_registered, moving_mask)

    mi = mutual_information_np(fixed, moving_registered, mask=common_mask)
    ncc = ncc_metric(fixed, moving_registered, mask=common_mask)

    return {
        "header_ok": header_ok,
        "body_dice": body_dice,
        "common_ratio_fixed": cr_fixed,
        "common_ratio_moving": cr_moving,
        "centroid_distance_mm": c_dist,
        "mi": mi,
        "ncc": ncc,
    }


def judge_qc_status(qc, used_transform):
    """
    这里是优化后的核心逻辑：

    1. header 不一致、body Dice 太低、common ratio 太低，才 FAIL。
    2. centroid distance 只作为 warning。
    3. 如果 used_transform == identity，则不因为 MI/NCC 没提升而 warning。
    4. 如果 used_transform == rigid，则 rigid 没有提升 MI/NCC 才 warning。
    """

    if not qc["header_ok"]:
        return "FAIL", "registered_header_not_equal_fixed"

    if qc["body_dice"] < QC_FAIL_BODY_DICE:
        return "FAIL", "body_dice_too_low"

    if qc["common_ratio_fixed"] < QC_FAIL_COMMON_RATIO:
        return "FAIL", "common_ratio_too_low"

    warnings = []

    if qc["body_dice"] < QC_PASS_BODY_DICE:
        warnings.append("body_dice_warning")

    if qc["common_ratio_fixed"] < QC_PASS_COMMON_RATIO:
        warnings.append("common_ratio_warning")

    if qc["centroid_distance_mm"] > QC_WARNING_CENTROID_DIST:
        warnings.append("centroid_distance_large")
    elif qc["centroid_distance_mm"] > QC_PASS_CENTROID_DIST:
        warnings.append("centroid_distance_warning")

    # identity 情况下，不检查 MI/NCC 是否提升
    if used_transform == "identity":
        pass

    # rigid 情况下，才检查 MI/NCC 有没有比 identity 好
    if used_transform == "rigid":
        if qc.get("mi_improvement") is not None and qc["mi_improvement"] <= 0:
            warnings.append("mi_not_improved")

        if qc.get("ncc_improvement") is not None and qc["ncc_improvement"] <= 0:
            warnings.append("ncc_not_improved")

    if warnings:
        return "WARNING", ";".join(warnings)

    return "PASS", "ok"


def should_use_rigid(identity_qc, rigid_qc):
    id_mi = identity_qc["mi"]
    rg_mi = rigid_qc["mi"]

    id_ncc = identity_qc["ncc"]
    rg_ncc = rigid_qc["ncc"]

    id_dice = identity_qc["body_dice"]
    rg_dice = rigid_qc["body_dice"]

    id_common = identity_qc["common_ratio_fixed"]
    rg_common = rigid_qc["common_ratio_fixed"]

    if rg_dice + 0.05 < id_dice:
        return False, "rigid_body_dice_worse"

    if rg_common + 0.05 < id_common:
        return False, "rigid_common_ratio_worse"

    mi_worse = (rg_mi is not None and id_mi is not None and rg_mi < id_mi - MI_DROP_TOL)
    ncc_worse = (rg_ncc is not None and id_ncc is not None and rg_ncc < id_ncc - NCC_DROP_TOL)

    if mi_worse and ncc_worse:
        return False, "rigid_mi_and_ncc_worse"

    return True, "rigid_accepted"


# =========================================================
# 6. 共同区域裁剪
# =========================================================

def get_common_body_bbox(images, threshold=BODY_THRESHOLD, margin=CROP_MARGIN):
    if len(images) == 0:
        raise RuntimeError("images 为空，无法计算共同有效区域。")

    common_mask = None

    for image in images:
        image_f = sitk.Cast(image, sitk.sitkFloat32)
        mask = image_f > threshold

        if common_mask is None:
            common_mask = mask
        else:
            common_mask = common_mask & mask

    cc = sitk.ConnectedComponent(common_mask)
    relabel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest = relabel == 1

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(largest)

    if not stats.HasLabel(1):
        raise RuntimeError("没有找到共同有效区域，请检查 series 是否同部位。")

    bbox = stats.GetBoundingBox(1)
    image_size = images[0].GetSize()

    x, y, z, sx, sy, sz = bbox

    start = [
        max(0, x - margin),
        max(0, y - margin),
        max(0, z - margin)
    ]

    end = [
        min(image_size[0], x + sx + margin),
        min(image_size[1], y + sy + margin),
        min(image_size[2], z + sz + margin)
    ]

    crop_size = [
        end[0] - start[0],
        end[1] - start[1],
        end[2] - start[2]
    ]

    return start, crop_size, bbox


def crop_image(image, start, crop_size):
    return sitk.RegionOfInterest(image, size=crop_size, index=start)


# =========================================================
# 7. 自动生成 overlay QC 图片
# =========================================================

def normalize_slice(slice_2d):
    x = slice_2d.astype(np.float32)
    low, high = np.percentile(x, [1, 99])

    if high <= low:
        return np.zeros_like(x)

    x = np.clip(x, low, high)
    return (x - low) / (high - low)


def save_overlay_qc_png(fixed, moving_registered, output_path, title="QC"):
    fixed_arr = image_to_array(fixed)
    moving_arr = image_to_array(moving_registered)

    if fixed_arr.shape != moving_arr.shape:
        return

    z = fixed_arr.shape[0] // 2

    f = normalize_slice(fixed_arr[z])
    m = normalize_slice(moving_arr[z])

    overlay = np.zeros((f.shape[0], f.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = f
    overlay[..., 1] = m
    overlay[..., 2] = 0

    diff = np.abs(f - m)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(f, cmap="gray")
    plt.title("fixed")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(m, cmap="gray")
    plt.title("moving selected")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(overlay)
    plt.title("overlay R=fixed G=moving")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(diff, cmap="gray")
    plt.title("abs diff")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =========================================================
# 8. CSV 保存
# =========================================================

def save_csv(path, rows):
    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# =========================================================
# 9. 主流程
# =========================================================

def main():
    global DEFAULT_PIXEL_VALUE
    global BODY_THRESHOLD
    global CROP_MARGIN
    global ALLOW_HEADER_WARNING_TO_REGISTER
    global ONLY_KEEP_QC_PASS

    args = parse_args()

    DEFAULT_PIXEL_VALUE = args.default_pixel_value
    BODY_THRESHOLD = args.body_threshold
    CROP_MARGIN = args.crop_margin
    ALLOW_HEADER_WARNING_TO_REGISTER = not args.no_header_warning
    ONLY_KEEP_QC_PASS = args.only_keep_qc_pass

    case_dir = os.path.abspath(args.case_dir)
    output_dir = os.path.abspath(args.output_dir)
    case_id = args.case_id or os.path.basename(case_dir.rstrip("/"))

    imagesTr_dir = os.path.join(output_dir, "imagesTr")
    uncropped_dir = os.path.join(output_dir, "uncropped")
    qc_png_dir = os.path.join(output_dir, "qc_png")
    logs_dir = os.path.join(output_dir, "logs")

    for d in [imagesTr_dir, uncropped_dir, qc_png_dir, logs_dir]:
        ensure_dir(d)

    print("CASE_DIR:", case_dir)
    print("OUTPUT_DIR:", output_dir)
    print("case_id:", case_id)

    # -----------------------------------------------------
    # Step 1: 读取所有 seriesXXX
    # -----------------------------------------------------

    series_dirs = list_series_dirs(case_dir)

    if len(series_dirs) < 2:
        raise RuntimeError("series 数量少于 2，无法配准。")

    print("\n发现 series 文件夹：")
    for d in series_dirs:
        print("  ", os.path.basename(d))

    series_infos = []

    for d in series_dirs:
        info = get_series_info(d)
        series_infos.append(info)

        print("-" * 80)
        print("Series:", info["series_name"])
        print("Size:", info["size"])
        print("Spacing:", info["spacing"])
        print("Z range:", info["z_min"], "to", info["z_max"])
        print("Slices:", info["num_slices"])

    # -----------------------------------------------------
    # Step 2: 自动选择 fixed
    # -----------------------------------------------------

    fixed_info = select_fixed_by_max_slices(series_infos)
    fixed = fixed_info["image"]
    fixed_dir = fixed_info["series_dir"]

    print("\n" + "=" * 100)
    print("自动选择 fixed:", fixed_info["series_name"])
    print("Size:", fixed_info["size"])
    print("Spacing:", fixed_info["spacing"])
    print("Slices:", fixed_info["num_slices"])
    print("=" * 100)

    fixed_uncropped_path = os.path.join(
        uncropped_dir,
        f"{case_id}_0000_{fixed_info['series_name']}_fixed_uncropped.nii.gz"
    )
    sitk.WriteImage(fixed, fixed_uncropped_path)

    # -----------------------------------------------------
    # Step 3: 配准前 header 筛选
    # -----------------------------------------------------

    header_rows = []
    moving_candidates = []

    for moving_info in series_infos:
        if moving_info["series_name"] == fixed_info["series_name"]:
            continue

        row = compare_header_to_fixed(fixed_info, moving_info)
        row["case"] = case_id
        header_rows.append(row)

        if row["header_status"] == "PASS":
            moving_candidates.append((moving_info, row))
        elif row["header_status"] == "WARNING" and ALLOW_HEADER_WARNING_TO_REGISTER:
            moving_candidates.append((moving_info, row))

    header_csv_path = os.path.join(logs_dir, f"{case_id}_header_compare.csv")
    save_csv(header_csv_path, header_rows)

    print("\n配准前 header 筛选结果：")
    for row in header_rows:
        print(
            row["moving_series"],
            row["header_status"],
            row["header_reason"],
            "z_overlap_fixed=",
            round(row["z_overlap_ratio_fixed"], 3)
        )

    print("\n进入配准的 moving：")
    for item, row in moving_candidates:
        print("  ", item["series_name"], row["header_status"])

    # -----------------------------------------------------
    # Step 4: 逐个 moving 处理：identity / rigid 自动选择
    # -----------------------------------------------------

    qc_rows = []
    registered_keep_images = [fixed]
    registered_keep_logs = []

    channel_logs = [{
        "channel": 0,
        "role": "fixed",
        "series_name": fixed_info["series_name"],
        "input_dir": fixed_dir,
        "uncropped_output": fixed_uncropped_path,
        "size": str(fixed.GetSize()),
        "spacing": str(fixed.GetSpacing()),
        "origin": str(fixed.GetOrigin()),
        "direction": str(fixed.GetDirection())
    }]

    next_channel_idx = 1

    for moving_info, header_row in moving_candidates:
        moving_name = moving_info["series_name"]
        moving = moving_info["image"]

        print("\n" + "=" * 100)
        print("开始处理:", moving_name, "→", fixed_info["series_name"])

        # identity resample
        moving_identity = identity_resample_moving_to_fixed(fixed, moving)
        identity_qc = compute_qc_metrics(fixed, moving_identity)

        # rigid registration
        transform, metric_value, stop_condition = rigid_registration(fixed, moving)
        moving_rigid = resample_moving_to_fixed(fixed, moving, transform)
        rigid_qc = compute_qc_metrics(fixed, moving_rigid)

        use_rigid, transform_decision_reason = should_use_rigid(identity_qc, rigid_qc)

        if use_rigid:
            selected_image = moving_rigid
            selected_qc = rigid_qc
            used_transform = "rigid"
        else:
            selected_image = moving_identity
            selected_qc = identity_qc
            used_transform = "identity"

        # 记录相对 identity 的提升。identity 自己没有 improvement 概念。
        selected_qc["mi_improvement"] = None
        selected_qc["ncc_improvement"] = None

        if used_transform == "rigid":
            selected_qc["mi_improvement"] = None if identity_qc["mi"] is None or rigid_qc["mi"] is None else rigid_qc["mi"] - identity_qc["mi"]
            selected_qc["ncc_improvement"] = None if identity_qc["ncc"] is None or rigid_qc["ncc"] is None else rigid_qc["ncc"] - identity_qc["ncc"]

        qc_status, qc_reason = judge_qc_status(selected_qc, used_transform)

        selected_uncropped_path = os.path.join(
            uncropped_dir,
            f"{case_id}_{next_channel_idx:04d}_{moving_name}_{used_transform}_uncropped.nii.gz"
        )
        sitk.WriteImage(selected_image, selected_uncropped_path)

        qc_png_path = os.path.join(
            qc_png_dir,
            f"{case_id}_{moving_name}_{used_transform}_qc_{qc_status}.png"
        )

        save_overlay_qc_png(
            fixed,
            selected_image,
            qc_png_path,
            title=f"{case_id} {moving_name} {used_transform} {qc_status}"
        )

        qc = {
            "case": case_id,
            "fixed_series": fixed_info["series_name"],
            "moving_series": moving_name,

            "header_status_before": header_row["header_status"],
            "header_reason_before": header_row["header_reason"],

            "used_transform": used_transform,
            "transform_decision_reason": transform_decision_reason,

            "identity_body_dice": identity_qc["body_dice"],
            "identity_common_ratio_fixed": identity_qc["common_ratio_fixed"],
            "identity_common_ratio_moving": identity_qc["common_ratio_moving"],
            "identity_centroid_distance_mm": identity_qc["centroid_distance_mm"],
            "identity_mi": identity_qc["mi"],
            "identity_ncc": identity_qc["ncc"],

            "rigid_body_dice": rigid_qc["body_dice"],
            "rigid_common_ratio_fixed": rigid_qc["common_ratio_fixed"],
            "rigid_common_ratio_moving": rigid_qc["common_ratio_moving"],
            "rigid_centroid_distance_mm": rigid_qc["centroid_distance_mm"],
            "rigid_mi": rigid_qc["mi"],
            "rigid_ncc": rigid_qc["ncc"],

            "registered_header_ok": selected_qc["header_ok"],
            "body_dice": selected_qc["body_dice"],
            "common_ratio_fixed": selected_qc["common_ratio_fixed"],
            "common_ratio_moving": selected_qc["common_ratio_moving"],
            "centroid_distance_mm": selected_qc["centroid_distance_mm"],

            "mi": selected_qc["mi"],
            "ncc": selected_qc["ncc"],
            "mi_improvement_vs_identity": selected_qc["mi_improvement"],
            "ncc_improvement_vs_identity": selected_qc["ncc_improvement"],

            "registration_metric_value": metric_value,
            "stop_condition": stop_condition,

            "qc_status": qc_status,
            "qc_reason": qc_reason,
            "uncropped_output": selected_uncropped_path,
            "qc_png": qc_png_path,
        }

        qc_rows.append(qc)

        print("used_transform:", used_transform, transform_decision_reason)
        print("QC status:", qc_status, qc_reason)
        print("identity MI/NCC:", identity_qc["mi"], identity_qc["ncc"])
        print("rigid MI/NCC:", rigid_qc["mi"], rigid_qc["ncc"])
        print("selected body_dice:", round(selected_qc["body_dice"], 4))
        print("selected common_ratio_fixed:", round(selected_qc["common_ratio_fixed"], 4))
        print("selected centroid_distance_mm:", round(selected_qc["centroid_distance_mm"], 3))
        print("QC PNG:", qc_png_path)

        keep_this = (qc_status == "PASS") if ONLY_KEEP_QC_PASS else (qc_status in ["PASS", "WARNING"])

        if keep_this:
            registered_keep_images.append(selected_image)

            registered_keep_logs.append({
                "channel": next_channel_idx,
                "role": "moving",
                "series_name": moving_name,
                "input_dir": moving_info["series_dir"],
                "image": selected_image,
                "used_transform": used_transform,
                "uncropped_output": selected_uncropped_path,
                "qc_status": qc_status,
                "qc_reason": qc_reason,
                "qc_png": qc_png_path,
            })

            next_channel_idx += 1

    qc_csv_path = os.path.join(logs_dir, f"{case_id}_qc_results.csv")
    save_csv(qc_csv_path, qc_rows)

    # -----------------------------------------------------
    # Step 5: 对 fixed + 保留 moving 做共同区域裁剪
    # -----------------------------------------------------

    if len(registered_keep_images) < 2:
        print("\n警告：没有保留的 moving，最终只保留 fixed，未生成有效多通道。")
        return

    print("\n" + "=" * 100)
    print("开始对 fixed + 保留 moving 做共同区域裁剪...")

    start, crop_size, raw_bbox = get_common_body_bbox(
        registered_keep_images,
        threshold=BODY_THRESHOLD,
        margin=CROP_MARGIN
    )

    print("Crop start:", start)
    print("Crop size:", crop_size)
    print("Raw bbox:", raw_bbox)

    fixed_cropped = crop_image(fixed, start, crop_size)
    fixed_output = os.path.join(imagesTr_dir, f"{case_id}_0000.nii.gz")
    sitk.WriteImage(fixed_cropped, fixed_output)

    channel_logs[0]["cropped_output"] = fixed_output
    channel_logs[0]["cropped_size"] = str(fixed_cropped.GetSize())
    channel_logs[0]["cropped_spacing"] = str(fixed_cropped.GetSpacing())
    channel_logs[0]["cropped_origin"] = str(fixed_cropped.GetOrigin())
    channel_logs[0]["cropped_direction"] = str(fixed_cropped.GetDirection())

    for idx, item in enumerate(registered_keep_logs, start=1):
        moving_cropped = crop_image(item["image"], start, crop_size)

        moving_output = os.path.join(imagesTr_dir, f"{case_id}_{idx:04d}.nii.gz")
        sitk.WriteImage(moving_cropped, moving_output)

        channel_logs.append({
            "channel": idx,
            "role": "moving",
            "series_name": item["series_name"],
            "input_dir": item["input_dir"],
            "used_transform": item["used_transform"],
            "uncropped_output": item["uncropped_output"],
            "cropped_output": moving_output,
            "qc_status": item["qc_status"],
            "qc_reason": item["qc_reason"],
            "qc_png": item["qc_png"],
            "cropped_size": str(moving_cropped.GetSize()),
            "cropped_spacing": str(moving_cropped.GetSpacing()),
            "cropped_origin": str(moving_cropped.GetOrigin()),
            "cropped_direction": str(moving_cropped.GetDirection())
        })

    final_log = {
        "case_id": case_id,
        "case_dir": case_dir,
        "output_dir": output_dir,
        "fixed_series": fixed_info["series_name"],
        "fixed_selection_rule": "choose seriesXXX with maximum z-slices",
        "header_compare_csv": header_csv_path,
        "qc_results_csv": qc_csv_path,
        "crop_start": start,
        "crop_size": crop_size,
        "raw_bbox": raw_bbox,
        "only_keep_qc_pass": ONLY_KEEP_QC_PASS,
        "channels": channel_logs,
    }

    final_log_path = os.path.join(logs_dir, f"{case_id}_final_log.json")

    with open(final_log_path, "w", encoding="utf-8") as f:
        json.dump(final_log, f, indent=4, ensure_ascii=False, default=str)

    print("\n" + "=" * 100)
    print("完成。最终 nnU-Net 输入在：")
    print(imagesTr_dir)

    print("\nHeader 筛选结果：")
    print(header_csv_path)

    print("\nQC 结果：")
    print(qc_csv_path)

    print("\nQC 图片：")
    print(qc_png_dir)

    print("\n最终日志：")
    print(final_log_path)


if __name__ == "__main__":
    main()
