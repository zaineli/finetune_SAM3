import argparse
import collections
import json
import random
import shutil
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch, Rectangle
from preview_export import (
    default_preview_root,
    prepare_preview_dirs,
    save_preview_assets,
    write_preview_annotations,
)

try:
    from pycocotools.mask import encode as mask_encode
except ImportError:
    mask_encode = None

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


CLASS_INFO = {
    1: {"name": "box", "category_name": "boxes", "text_input": "boxes"},
    2: {"name": "dot", "category_name": "dots", "text_input": "dots"},
    3: {"name": "line", "category_name": "lines", "text_input": "lines"},
    4: {"name": "annotation", "category_name": "annotations", "text_input": "annotations"},
    5: {"name": "legend", "category_name": "legend", "text_input": "legend"},
}

DPI = 180
MASK_THRESHOLD = 10

CLINICAL_BOX_PALETTES = [
    ["#ff6a4d", "#95e483", "#f6c23e", "#a40000", "#de68e6", "#0f8f8f", "#4564d8"],
    ["#4878d0", "#d65f5f", "#2d2db5", "#8b0000", "#74c476", "#e7ba52", "#ad7fa8"],
    ["#ff8f80", "#9b72ff", "#5cc96c", "#f4b942", "#2a66e8", "#b13fdb", "#0e9aa7"],
]

VISIT_PALETTES = [
    ["#9aa0a6", "#f28e2b"],
    ["#4e79a7", "#e15759"],
    ["#b07aa1", "#59a14f"],
]

COHORT_PANEL_METRICS = [
    {"title": "Swiss Cohort Immunoassay Aβ42", "ylabel": "pg/mL", "range": (260.0, 1550.0), "distribution": "positive"},
    {"title": "Swiss Cohort Immunoassay Total Tau", "ylabel": "pg/mL", "range": (40.0, 1850.0), "distribution": "positive"},
    {"title": "Swiss Cohort CSF/Serum Albumin Ratio", "ylabel": "Ratio", "range": (2.2, 15.5), "distribution": "positive"},
    {"title": "Trial Cohort Plasma pTau181", "ylabel": "pg/mL", "range": (0.8, 22.0), "distribution": "positive"},
    {"title": "Phase II Serum NfL", "ylabel": "pg/mL", "range": (4.0, 120.0), "distribution": "positive"},
    {"title": "CSF GFAP Immunoassay", "ylabel": "ng/L", "range": (55.0, 980.0), "distribution": "positive"},
]

DEMOGRAPHIC_METRICS = [
    {"title": "Immunoassay Total Tau by Demographic", "ylabel": "Z-Score", "range": (-1.7, 3.9), "distribution": "centered"},
    {"title": "Plasma NfL by Demographic", "ylabel": "Z-Score", "range": (-1.9, 3.5), "distribution": "centered"},
    {"title": "CSF Aβ42 by Demographic", "ylabel": "Normalized score", "range": (-1.6, 3.2), "distribution": "centered"},
]

GROUPED_VISIT_METRICS = [
    {"title": "CSF Aβ42 by Treatment Arm", "ylabel": "pg/mL", "range": (180.0, 1550.0), "distribution": "positive"},
    {"title": "Plasma pTau181 by Treatment Arm", "ylabel": "pg/mL", "range": (0.8, 20.0), "distribution": "positive"},
    {"title": "Serum NfL Change by Visit", "ylabel": "pg/mL", "range": (6.0, 110.0), "distribution": "positive"},
    {"title": "Albumin Ratio by Study Visit", "ylabel": "Ratio", "range": (2.0, 14.0), "distribution": "positive"},
]

PANEL_LABELS = list("BCDEFGH")
DEMOGRAPHIC_HEADERS = [
    ("Control Cases", "Alzheimer's Disease"),
    ("Healthy Controls", "Mild Cognitive Impairment"),
    ("Standard of Care", "Investigational Arm"),
]
DEMOGRAPHIC_LABEL_SETS = [
    ["NHW\nMale", "NHW\nFemale", "AA\nMale", "AA\nFemale"],
    ["European\nMale", "European\nFemale", "African\nMale", "African\nFemale"],
    ["Younger\nMale", "Younger\nFemale", "Older\nMale", "Older\nFemale"],
]
TRIAL_ARM_SETS = [
    ["Placebo", "Low Dose", "High Dose"],
    ["Standard Care", "Dose 1", "Dose 2"],
    ["Control", "Responder", "Non-responder"],
]
VISIT_LABEL_SETS = [
    ["Baseline", "Week 24"],
    ["Screening", "Week 52"],
    ["Visit 1", "Visit 4"],
]


def make_text_item(
    text: str,
    x: float,
    y: float,
    coords: str = "data",
    fontsize: float = 10.0,
    ha: str = "left",
    va: str = "center",
    color: str = "black",
    fontweight: str = "normal",
):
    return {
        "text": text,
        "x": x,
        "y": y,
        "coords": coords,
        "fontsize": fontsize,
        "ha": ha,
        "va": va,
        "color": color,
        "fontweight": fontweight,
    }


def make_line_item(kind: str, **kwargs):
    item = {"kind": kind}
    item.update(kwargs)
    return item


def make_background_patch(kind: str = "data_rect", **kwargs):
    item = {"kind": kind}
    item.update(kwargs)
    return item


def darken_color(color, amount: float = 0.35):
    rgb = np.array(plt.matplotlib.colors.to_rgb(color), dtype=float)
    return tuple(np.clip(rgb * (1.0 - amount), 0.0, 1.0))


def encode_rle(binary_mask: np.ndarray):
    if binary_mask.sum() == 0:
        return {"size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])], "counts": ""}
    if mask_encode is not None:
        rle = mask_encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        counts = rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]
        return {"size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])], "counts": counts}

    flat_mask = np.asfortranarray(binary_mask.astype(np.uint8)).reshape(-1)
    counts = []
    current_value = 0
    run_length = 0
    for value in flat_mask:
        if value == current_value:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            current_value = int(value)
    counts.append(run_length)
    return {"size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])], "counts": counts}


def get_bbox_and_area(binary_mask: np.ndarray):
    ys, xs = np.nonzero(binary_mask)
    if len(ys) == 0:
        return None, 0
    xmin, ymin = int(xs.min()), int(ys.min())
    xmax, ymax = int(xs.max()), int(ys.max())
    bbox_xywh = [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1]
    area = int(binary_mask.sum())
    return bbox_xywh, area


def split_connected_components(binary_mask: np.ndarray, min_area: int = 1):
    if binary_mask.sum() == 0:
        return []

    if ndimage is not None:
        labeled_mask, component_count = ndimage.label(
            binary_mask.astype(bool),
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        instances = []
        for component_id in range(1, component_count + 1):
            component_mask = (labeled_mask == component_id).astype(np.uint8)
            if int(component_mask.sum()) < min_area:
                continue
            instances.append(component_mask)
        return instances

    mask = binary_mask.astype(bool)
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    instances = []

    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue

        queue = collections.deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        pixels = []

        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if not mask[ny, nx] or visited[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))

        if len(pixels) < min_area:
            continue
        component_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        ys, xs = zip(*pixels)
        component_mask[np.array(ys), np.array(xs)] = 1
        instances.append(component_mask)
    return instances


def format_p_value(strong: bool = True):
    if strong:
        mantissa = random.uniform(1.2, 9.8)
        exponent = random.randint(4, 54)
        template = random.choice(["p={:.2f}e-{:02d}", "p = {:.2f}e-{:02d}"])
        return template.format(mantissa, exponent)
    value = random.uniform(0.0008, 0.049)
    template = random.choice(["p={:.3f}", "p = {:.3f}"])
    return template.format(value)


def star_label(effect_ratio: float):
    if effect_ratio >= 0.85:
        return "****"
    if effect_ratio >= 0.6:
        return "***"
    if effect_ratio >= 0.35:
        return "**"
    if effect_ratio >= 0.18:
        return "*"
    return None


def sample_values(center: float, sample_count: int, distribution: str, low: float, high: float):
    if distribution == "centered":
        spread = max((high - low) * random.uniform(0.05, 0.11), random.uniform(0.15, 0.45))
        values = np.random.normal(center, spread, sample_count)
        if sample_count >= 10 and random.random() < 0.28:
            tail_mask = np.random.rand(sample_count) < random.uniform(0.08, 0.18)
            values[tail_mask] += np.random.normal(spread * random.choice([-2.2, 2.2]), spread * 0.55, tail_mask.sum())
        return np.clip(values, low, high)

    sigma = random.uniform(0.18, 0.42)
    mu = np.log(max(center, 1e-3))
    values = np.random.lognormal(mean=mu, sigma=sigma, size=sample_count)
    values *= center / max(np.median(values), 1e-6)
    if sample_count >= 8 and random.random() < 0.34:
        tail_mask = np.random.rand(sample_count) < random.uniform(0.08, 0.22)
        if tail_mask.any():
            values[tail_mask] += np.random.gamma(2.0, max(center * 0.08, (high - low) * 0.03), tail_mask.sum())
    return np.clip(values, low, high)


def compute_box_stats(values: np.ndarray):
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = max(q3 - q1, 1e-6)
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    inside = values[(values >= lower_limit) & (values <= upper_limit)]
    if inside.size == 0:
        inside = values
    whisker_low = float(inside.min())
    whisker_high = float(inside.max())
    return {
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
    }


def make_box_item(
    label: str,
    position: float,
    width: float,
    samples: np.ndarray,
    facecolor,
    edgecolor,
    alpha: float,
    linewidth: float,
):
    stats = compute_box_stats(samples)
    value_span = float(max(samples.max() - samples.min(), 1e-5))
    box_height = max(stats["q3"] - stats["q1"], value_span * 0.02, abs(float(samples.mean())) * 0.005, 1e-4)
    return {
        "label": label,
        "x": float(position),
        "width": float(width),
        "samples": samples.astype(float),
        "facecolor": facecolor,
        "edgecolor": edgecolor,
        "alpha": alpha,
        "linewidth": linewidth,
        "box_height": box_height,
        **stats,
    }


def make_box_structure_line_item(box_item: dict, color, linewidth: float, whisker_style: str, show_mean: bool):
    return make_line_item(
        "box_structure",
        x=box_item["x"],
        width=box_item["width"],
        q1=box_item["q1"],
        q3=box_item["q3"],
        median=box_item["median"],
        whisker_low=box_item["whisker_low"],
        whisker_high=box_item["whisker_high"],
        mean=float(box_item["samples"].mean()),
        color=color,
        linewidth=linewidth,
        whisker_style=whisker_style,
        show_mean=show_mean,
    )


def build_legend_config(items: list[dict], enabled: bool):
    if not enabled or not items:
        return {"enabled": False}

    family = random.choices(
        ["top_outside", "bottom_outside", "upper_right"],
        weights=[0.46, 0.22, 0.32],
        k=1,
    )[0]
    if family == "top_outside":
        return {"enabled": True, "items": items, "family": family, "loc": "lower center", "bbox_to_anchor": (0.5, 1.06), "ncol": len(items)}
    if family == "bottom_outside":
        return {"enabled": True, "items": items, "family": family, "loc": "upper center", "bbox_to_anchor": (0.5, -0.14), "ncol": len(items)}
    return {"enabled": True, "items": items, "family": family, "loc": "upper right", "bbox_to_anchor": None, "ncol": 1}


def build_dot_items(box_items: list[dict], style_family: str):
    dot_style = random.choices(
        ["filled_black", "hollow_black", "filled_color"],
        weights=[0.44, 0.34, 0.22],
        k=1,
    )[0]
    size = random.uniform(10.0, 19.0) if style_family != "demographic_split" else random.uniform(13.0, 21.0)
    dot_items = []

    for box_item in box_items:
        jitter_radius = min(box_item["width"] * 0.26, 0.12)
        jitters = np.random.normal(0.0, jitter_radius * 0.55, len(box_item["samples"]))
        jitters = np.clip(jitters, -jitter_radius, jitter_radius)
        for jitter, value in zip(jitters, box_item["samples"]):
            if dot_style == "filled_black":
                facecolor = "#111111"
                edgecolor = "#111111"
                linewidth = 0.6
            elif dot_style == "hollow_black":
                facecolor = "white"
                edgecolor = "#222222"
                linewidth = 1.0
            else:
                facecolor = box_item["facecolor"]
                edgecolor = darken_color(box_item["facecolor"], 0.42)
                linewidth = 0.85

            dot_items.append(
                {
                    "x": float(box_item["x"] + jitter),
                    "y": float(value),
                    "facecolor": facecolor,
                    "edgecolor": edgecolor,
                    "size": size,
                    "linewidth": linewidth,
                    "alpha": 0.94,
                }
            )
    return dot_items, dot_style


def add_bracket(line_items: list[dict], annotation_items: list[dict], x1: float, x2: float, y: float, arm: float, label: str, color: str = "#222222"):
    line_items.append(make_line_item("bracket", x1=float(x1), x2=float(x2), y=float(y), arm=float(arm), color=color, linewidth=1.6))
    annotation_items.append(
        make_text_item(
            label,
            (x1 + x2) / 2.0,
            y + arm * 0.25,
            coords="data",
            fontsize=11.5,
            ha="center",
            va="bottom",
            color=color,
            fontweight="bold",
        )
    )


def build_cohort_panel_spec():
    metric = random.choice(COHORT_PANEL_METRICS)
    palette = random.choice(CLINICAL_BOX_PALETTES)
    num_boxes = random.randint(5, 7)
    labels = [f"Subtype {idx + 1}" for idx in range(num_boxes)]
    positions = np.arange(num_boxes, dtype=float)
    width = random.uniform(0.42, 0.58)
    low, high = metric["range"]
    span = high - low

    centers = np.random.uniform(low + 0.18 * span, high - 0.22 * span, size=num_boxes)
    if random.random() < 0.56:
        major_idx = random.randrange(num_boxes)
        centers[major_idx] *= random.uniform(1.15, 1.5)
    centers = np.clip(centers, low + 0.08 * span, high - 0.06 * span)

    box_items = []
    line_items = []
    for idx, position in enumerate(positions):
        sample_count = random.randint(10, 26)
        samples = sample_values(float(centers[idx]), sample_count, metric["distribution"], low, high)
        facecolor = palette[idx % len(palette)]
        edgecolor = darken_color(facecolor, 0.3)
        box_item = make_box_item(labels[idx], position, width, samples, facecolor, edgecolor, alpha=0.9, linewidth=1.4)
        box_items.append(box_item)
        line_items.append(
            make_box_structure_line_item(
                box_item,
                color="#202020",
                linewidth=1.15,
                whisker_style=random.choice(["--", ":", "-."]),
                show_mean=random.random() < 0.18,
            )
        )

    dot_items, dot_style = build_dot_items(box_items, "cohort_panel")
    top_value = max(max(item["samples"]) for item in box_items)
    bottom_value = min(min(item["samples"]) for item in box_items)
    ymin = max(0.0, bottom_value - span * 0.08)
    ymax = top_value + span * 0.18

    annotation_items = [
        make_text_item(
            format_p_value(strong=True),
            0.02 if random.random() < 0.55 else 0.03,
            0.05 if random.random() < 0.34 else 0.95,
            coords="axes",
            fontsize=10.5,
            ha="left",
            va="bottom" if random.random() < 0.34 else "top",
            color="#ff4b3a",
            fontweight="bold",
        )
    ]

    sorted_idx = np.argsort(centers)
    left_idx = int(sorted_idx[0])
    right_idx = int(sorted_idx[-1])
    effect_ratio = abs(centers[right_idx] - centers[left_idx]) / max((centers[right_idx] + centers[left_idx]) * 0.5, 1e-6)
    label = star_label(effect_ratio) or "*"
    bracket_y = top_value + span * random.uniform(0.06, 0.1)
    add_bracket(line_items, annotation_items, positions[left_idx], positions[right_idx], bracket_y, span * 0.03, label)
    ymax = max(ymax, bracket_y + span * 0.08)

    return {
        "family": "cohort_panel",
        "figure_size": random.choice([(7.9, 6.0), (8.4, 6.1), (8.8, 6.0)]),
        "xlim": (-0.6, num_boxes - 0.4),
        "ylim": (ymin, ymax),
        "xticks": positions.tolist(),
        "xticklabels": labels,
        "xlabel": "",
        "ylabel": metric["ylabel"],
        "title": metric["title"],
        "title_fontsize": 14.5,
        "title_pad": 8,
        "label_fontsize": 11.5,
        "tick_fontsize": 9.8,
        "x_tick_rotation": random.choice([0, 0, 0, 18]),
        "show_grid": False,
        "grid_axis": "y",
        "grid_color": "#dddddd",
        "grid_linestyle": ":",
        "grid_linewidth": 0.6,
        "grid_alpha": 0.45,
        "hide_top_right": False,
        "facecolor": "white",
        "spine_linewidth": 1.6,
        "margins": {"left": 0.11, "right": 0.97, "top": 0.9, "bottom": 0.15},
        "background_patches": [],
        "box_items": box_items,
        "dot_items": dot_items,
        "line_items": line_items,
        "annotation_items": annotation_items,
        "legend": {"enabled": False},
        "legend_fontsize": 8.5,
        "metadata": {
            "family": "cohort_panel",
            "dot_style": dot_style,
            "legend_enabled": False,
        },
    }


def build_demographic_split_spec():
    metric = random.choice(DEMOGRAPHIC_METRICS)
    left_header, right_header = random.choice(DEMOGRAPHIC_HEADERS)
    demographic_labels = random.choice(DEMOGRAPHIC_LABEL_SETS)
    positions = np.arange(8, dtype=float)
    labels = demographic_labels + demographic_labels
    colors = ["#4878d0", "#e41a1c", "#1c1cb3", "#9b111e"] * 2
    width = random.uniform(0.56, 0.68)
    low, high = metric["range"]
    span = high - low

    left_centers = np.array([
        random.uniform(low + 0.16 * span, low + 0.32 * span),
        random.uniform(low + 0.12 * span, low + 0.34 * span),
        random.uniform(low + 0.10 * span, low + 0.28 * span),
        random.uniform(low + 0.08 * span, low + 0.24 * span),
    ])
    right_shift = np.array([
        random.uniform(0.18, 0.44),
        random.uniform(0.24, 0.56),
        random.uniform(0.1, 0.26),
        random.uniform(0.16, 0.38),
    ]) * span
    right_centers = np.clip(left_centers + right_shift, low + 0.18 * span, high - 0.08 * span)
    centers = np.concatenate([left_centers, right_centers])

    box_items = []
    line_items = []
    for idx, position in enumerate(positions):
        sample_count = random.randint(12, 42)
        samples = sample_values(float(centers[idx]), sample_count, metric["distribution"], low, high)
        facecolor = colors[idx]
        edgecolor = darken_color(facecolor, 0.3)
        box_item = make_box_item(labels[idx], position, width, samples, facecolor, edgecolor, alpha=0.88, linewidth=1.45)
        box_items.append(box_item)
        line_items.append(
            make_box_structure_line_item(
                box_item,
                color="#151515",
                linewidth=1.2,
                whisker_style="-",
                show_mean=random.random() < 0.1,
            )
        )

    line_items.append(make_line_item("vline", x=3.5, ymin=low, ymax=high, color="#666666", linewidth=1.2, linestyle="-"))
    dot_items, dot_style = build_dot_items(box_items, "demographic_split")

    top_value = max(max(item["samples"]) for item in box_items)
    bottom_value = min(min(item["samples"]) for item in box_items)
    ymin = bottom_value - span * 0.1
    ymax = top_value + span * 0.16
    annotation_items = [
        make_text_item(format_p_value(strong=True), 0.02, 0.95, coords="axes", fontsize=10.5, ha="left", va="top", color="black", fontweight="bold"),
        make_text_item(f"n = {sum(len(item['samples']) for item in box_items)}", 0.02, 0.885, coords="axes", fontsize=10.5, ha="left", va="top", color="black", fontweight="bold"),
        make_text_item(random.choice(PANEL_LABELS) + ".", -0.08, 1.075, coords="axes", fontsize=22.0, ha="left", va="center", color="black", fontweight="bold"),
        make_text_item(left_header, 0.25, 1.04, coords="axes", fontsize=11.0, ha="center", va="center", color="#1a1a1a", fontweight="bold"),
        make_text_item(right_header, 0.75, 1.04, coords="axes", fontsize=11.0, ha="center", va="center", color="white", fontweight="bold"),
    ]

    background_patches = [
        make_background_patch("axes_rect", x=0.0, y=0.0, width=0.5, height=1.0, facecolor="#f3f3f3", edgecolor="none", alpha=1.0),
        make_background_patch("axes_rect", x=0.5, y=0.0, width=0.5, height=1.0, facecolor="#e5e5e5", edgecolor="none", alpha=1.0),
        make_background_patch("axes_rect", x=0.0, y=1.0, width=0.5, height=0.095, facecolor="#d6d6d6", edgecolor="#222222", alpha=1.0),
        make_background_patch("axes_rect", x=0.5, y=1.0, width=0.5, height=0.095, facecolor="#111111", edgecolor="#222222", alpha=1.0),
    ]

    pair_candidates = [(4, 5), (4, 6), (5, 7)]
    selected_pairs = random.sample(pair_candidates, k=random.choice([1, 2]))
    used_y = top_value + span * 0.04
    for left_idx, right_idx in selected_pairs:
        effect_ratio = abs(centers[right_idx] - centers[left_idx]) / max(abs((centers[right_idx] + centers[left_idx]) * 0.5), 1e-6)
        label = star_label(effect_ratio) or random.choice(["*", "**"])
        add_bracket(line_items, annotation_items, positions[left_idx], positions[right_idx], used_y, span * 0.025, label, color="#777777")
        used_y += span * 0.06
    ymax = max(ymax, used_y + span * 0.06)

    return {
        "family": "demographic_split",
        "figure_size": random.choice([(8.4, 6.4), (8.9, 6.8), (9.2, 6.7)]),
        "xlim": (-0.8, 7.8),
        "ylim": (ymin, ymax),
        "xticks": positions.tolist(),
        "xticklabels": labels,
        "xlabel": "",
        "ylabel": metric["ylabel"],
        "title": metric["title"],
        "title_fontsize": 14.2,
        "title_pad": 32,
        "label_fontsize": 12.0,
        "tick_fontsize": 9.8,
        "x_tick_rotation": 0,
        "show_grid": False,
        "grid_axis": "y",
        "grid_color": "#dddddd",
        "grid_linestyle": ":",
        "grid_linewidth": 0.6,
        "grid_alpha": 0.35,
        "hide_top_right": False,
        "facecolor": "white",
        "spine_linewidth": 1.5,
        "margins": {"left": 0.11, "right": 0.97, "top": 0.84, "bottom": 0.14},
        "background_patches": background_patches,
        "box_items": box_items,
        "dot_items": dot_items,
        "line_items": line_items,
        "annotation_items": annotation_items,
        "legend": {"enabled": False},
        "legend_fontsize": 8.5,
        "metadata": {
            "family": "demographic_split",
            "dot_style": dot_style,
            "legend_enabled": False,
        },
    }


def build_grouped_visit_spec():
    metric = random.choice(GROUPED_VISIT_METRICS)
    arm_labels = random.choice(TRIAL_ARM_SETS)
    visit_labels = random.choice(VISIT_LABEL_SETS)
    num_groups = random.choice([2, 3])
    num_categories = 2
    group_labels = arm_labels[:num_groups]
    group_positions = np.arange(num_groups, dtype=float)
    width = random.uniform(0.24, 0.32)
    offsets = np.linspace(-width * 0.55, width * 0.55, num_categories)
    palette = random.choice(VISIT_PALETTES)
    low, high = metric["range"]
    span = high - low

    baseline_centers = np.random.uniform(low + 0.18 * span, low + 0.45 * span, size=num_groups)
    delta = np.linspace(random.uniform(-0.04, 0.02), random.uniform(0.12, 0.3), num_groups) * span
    endpoint_centers = np.clip(baseline_centers + delta, low + 0.08 * span, high - 0.08 * span)

    box_items = []
    line_items = []
    for group_idx in range(num_groups):
        for category_idx in range(num_categories):
            center = baseline_centers[group_idx] if category_idx == 0 else endpoint_centers[group_idx]
            position = group_positions[group_idx] + offsets[category_idx]
            sample_count = random.randint(10, 28)
            samples = sample_values(float(center), sample_count, metric["distribution"], low, high)
            facecolor = palette[category_idx]
            edgecolor = darken_color(facecolor, 0.28)
            box_item = make_box_item(
                visit_labels[category_idx],
                position,
                width,
                samples,
                facecolor,
                edgecolor,
                alpha=0.9,
                linewidth=1.35,
            )
            box_items.append(box_item)
            line_items.append(
                make_box_structure_line_item(
                    box_item,
                    color=edgecolor,
                    linewidth=1.2,
                    whisker_style=random.choice(["--", "-."]),
                    show_mean=random.random() < 0.16,
                )
            )

    dot_items, dot_style = build_dot_items(box_items, "grouped_visit")
    top_value = max(max(item["samples"]) for item in box_items)
    bottom_value = min(min(item["samples"]) for item in box_items)
    ymin = max(0.0, bottom_value - span * 0.08)
    ymax = top_value + span * 0.16

    annotation_items = [
        make_text_item(format_p_value(strong=random.random() < 0.78), 0.02, 0.95, coords="axes", fontsize=10.0, ha="left", va="top", color="black", fontweight="bold")
    ]
    for group_idx in range(num_groups):
        left_box = box_items[group_idx * 2]
        right_box = box_items[group_idx * 2 + 1]
        effect_ratio = abs(right_box["median"] - left_box["median"]) / max((right_box["median"] + left_box["median"]) * 0.5, 1e-6)
        label = star_label(effect_ratio)
        if not label and random.random() < 0.42:
            continue
        add_bracket(
            line_items,
            annotation_items,
            left_box["x"],
            right_box["x"],
            max(left_box["whisker_high"], right_box["whisker_high"]) + span * (0.04 + 0.03 * group_idx),
            span * 0.022,
            label or "*",
        )
        ymax = max(ymax, max(left_box["whisker_high"], right_box["whisker_high"]) + span * 0.12)

    legend_enabled = random.random() < 0.62
    legend_items = [
        {"label": visit_labels[0], "facecolor": palette[0], "edgecolor": darken_color(palette[0], 0.28)},
        {"label": visit_labels[1], "facecolor": palette[1], "edgecolor": darken_color(palette[1], 0.28)},
    ]
    legend = build_legend_config(legend_items, legend_enabled)

    margins = {"left": 0.11, "right": 0.97, "top": 0.89, "bottom": 0.16}
    if legend.get("family") == "top_outside":
        margins["top"] = 0.78
    elif legend.get("family") == "bottom_outside":
        margins["bottom"] = 0.23

    return {
        "family": "grouped_visit",
        "figure_size": random.choice([(8.2, 6.0), (8.8, 6.2), (9.0, 6.1)]),
        "xlim": (group_positions[0] - 0.6, group_positions[-1] + 0.6),
        "ylim": (ymin, ymax),
        "xticks": group_positions.tolist(),
        "xticklabels": group_labels,
        "xlabel": "Treatment arm",
        "ylabel": metric["ylabel"],
        "title": metric["title"],
        "title_fontsize": 14.5,
        "title_pad": 10,
        "label_fontsize": 11.2,
        "tick_fontsize": 9.8,
        "x_tick_rotation": random.choice([0, 0, 15]),
        "show_grid": False,
        "grid_axis": "y",
        "grid_color": "#dddddd",
        "grid_linestyle": ":",
        "grid_linewidth": 0.6,
        "grid_alpha": 0.4,
        "hide_top_right": False,
        "facecolor": "white",
        "spine_linewidth": 1.45,
        "margins": margins,
        "background_patches": [],
        "box_items": box_items,
        "dot_items": dot_items,
        "line_items": line_items,
        "annotation_items": annotation_items,
        "legend": legend,
        "legend_fontsize": 8.4,
        "metadata": {
            "family": "grouped_visit",
            "dot_style": dot_style,
            "legend_enabled": legend_enabled,
        },
    }


def generate_boxplot_spec(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    family = random.choices(
        ["cohort_panel", "demographic_split", "grouped_visit"],
        weights=[0.42, 0.3, 0.28],
        k=1,
    )[0]
    if family == "cohort_panel":
        return build_cohort_panel_spec()
    if family == "demographic_split":
        return build_demographic_split_spec()
    return build_grouped_visit_spec()


def configure_axes(ax, data, mask_mode: bool):
    text_color = "black"
    line_color = "black"
    ax.clear()
    ax.set_facecolor("black" if mask_mode else data["facecolor"])
    ax.set_xlim(*data["xlim"])
    ax.set_ylim(*data["ylim"])
    ax.set_xticks(data["xticks"])
    ax.set_xticklabels(data["xticklabels"])
    if data["xlabel"]:
        ax.set_xlabel(data["xlabel"], fontsize=data["label_fontsize"], color=text_color)
    ax.set_ylabel(data["ylabel"], fontsize=data["label_fontsize"], color=text_color)
    ax.set_title(
        data["title"],
        fontsize=data["title_fontsize"],
        color=text_color,
        pad=data.get("title_pad", 10),
        fontweight="bold",
    )

    if data["show_grid"]:
        ax.grid(
            True,
            axis=data["grid_axis"],
            color="black" if mask_mode else data["grid_color"],
            linestyle=data["grid_linestyle"],
            linewidth=data["grid_linewidth"],
            alpha=1.0 if mask_mode else data["grid_alpha"],
        )
    else:
        ax.grid(False)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(data["spine_linewidth"])
    if data["hide_top_right"]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax.tick_params(colors=text_color, labelsize=data["tick_fontsize"], width=1.1, length=5.5)
    if data["x_tick_rotation"]:
        for tick in ax.get_xticklabels():
            tick.set_rotation(data["x_tick_rotation"])
            tick.set_ha("right")
    ax.set_axisbelow(True)


def draw_background_patches(ax, data, mask_mode: bool):
    if mask_mode:
        return
    for patch in data["background_patches"]:
        if patch["kind"] == "axes_rect":
            rect = Rectangle(
                (patch["x"], patch["y"]),
                patch["width"],
                patch["height"],
                transform=ax.transAxes,
                facecolor=patch["facecolor"],
                edgecolor=patch["edgecolor"],
                linewidth=1.0 if patch["edgecolor"] != "none" else 0.0,
                alpha=patch["alpha"],
                clip_on=False,
                zorder=0.2,
            )
        else:
            rect = Rectangle(
                (patch["x"], patch["y"]),
                patch["width"],
                patch["height"],
                facecolor=patch["facecolor"],
                edgecolor=patch["edgecolor"],
                linewidth=1.0 if patch["edgecolor"] != "none" else 0.0,
                alpha=patch["alpha"],
                zorder=0.2,
            )
        ax.add_patch(rect)


def draw_box_items(ax, data, mask_mode: bool):
    for item in data["box_items"]:
        facecolor = "white" if mask_mode else item["facecolor"]
        edgecolor = "white" if mask_mode else item["edgecolor"]
        box_height = max(item["box_height"], 1e-4)
        y0 = item["q1"]
        rect = Rectangle(
            (item["x"] - item["width"] / 2.0, y0),
            item["width"],
            box_height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=item["linewidth"],
            alpha=1.0 if mask_mode else item["alpha"],
            zorder=2,
        )
        ax.add_patch(rect)


def draw_dot_items(ax, data, mask_mode: bool):
    for item in data["dot_items"]:
        facecolor = "white" if mask_mode else item["facecolor"]
        edgecolor = "white" if mask_mode else item["edgecolor"]
        ax.scatter(
            [item["x"]],
            [item["y"]],
            s=item["size"] + (6 if mask_mode else 0),
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=1.0 if mask_mode else item["linewidth"],
            alpha=1.0 if mask_mode else item["alpha"],
            zorder=3,
        )


def draw_line_items(ax, data, mask_mode: bool):
    for item in data["line_items"]:
        color = "white" if mask_mode else item["color"]
        linewidth = item["linewidth"] + (0.5 if mask_mode else 0.0)
        if item["kind"] == "box_structure":
            x = item["x"]
            half_width = item["width"] / 2.0
            cap_width = item["width"] * 0.42
            ax.plot([x, x], [item["q3"], item["whisker_high"]], color=color, linewidth=linewidth, linestyle=item["whisker_style"], zorder=4)
            ax.plot([x, x], [item["q1"], item["whisker_low"]], color=color, linewidth=linewidth, linestyle=item["whisker_style"], zorder=4)
            ax.plot([x - cap_width / 2.0, x + cap_width / 2.0], [item["whisker_high"], item["whisker_high"]], color=color, linewidth=linewidth, zorder=4)
            ax.plot([x - cap_width / 2.0, x + cap_width / 2.0], [item["whisker_low"], item["whisker_low"]], color=color, linewidth=linewidth, zorder=4)
            ax.plot([x - half_width, x + half_width], [item["median"], item["median"]], color=color, linewidth=linewidth + (0.4 if not mask_mode else 0.0), zorder=4)
            if item["show_mean"]:
                ax.plot([x], [item["mean"]], marker="+", markersize=7.0, markeredgewidth=1.2, color=color, zorder=4)
        elif item["kind"] == "bracket":
            ax.plot(
                [item["x1"], item["x1"], item["x2"], item["x2"]],
                [item["y"] - item["arm"], item["y"], item["y"], item["y"] - item["arm"]],
                color=color,
                linewidth=linewidth,
                zorder=4,
            )
        elif item["kind"] == "vline":
            ax.plot([item["x"], item["x"]], [item["ymin"], item["ymax"]], color=color, linewidth=linewidth, linestyle=item["linestyle"], zorder=3.5)


def draw_annotation_items(ax, data, mask_mode: bool):
    for item in data["annotation_items"]:
        color = "white" if mask_mode else item["color"]
        if item["coords"] == "axes":
            transform = ax.transAxes
            clip_on = False
        else:
            transform = ax.transData
            clip_on = True
        ax.text(
            item["x"],
            item["y"],
            item["text"],
            transform=transform,
            fontsize=item["fontsize"],
            ha=item["ha"],
            va=item["va"],
            color=color,
            fontweight=item["fontweight"],
            clip_on=clip_on,
            zorder=5,
        )


def build_legend_handles(data, mask_mode: bool):
    handles = []
    for item in data["legend"]["items"]:
        facecolor = "white" if mask_mode else item["facecolor"]
        edgecolor = "white" if mask_mode else item["edgecolor"]
        handles.append(
            Patch(
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.0,
                label=item["label"],
            )
        )
    return handles


def draw_legend(ax, data, mask_mode: bool):
    if not data["legend"]["enabled"]:
        return None
    legend = data["legend"]
    handles = build_legend_handles(data, mask_mode=mask_mode)
    leg = ax.legend(
        handles=handles,
        loc=legend["loc"],
        bbox_to_anchor=legend["bbox_to_anchor"],
        ncol=legend["ncol"],
        frameon=True,
        fontsize=data["legend_fontsize"],
        title="Groups",
        title_fontsize=data["legend_fontsize"] + 0.4,
        borderaxespad=0.5,
        columnspacing=1.1,
        handlelength=1.7,
    )
    frame = leg.get_frame()
    if mask_mode:
        frame.set_facecolor("white")
        frame.set_edgecolor("white")
        frame.set_alpha(1.0)
        leg.get_title().set_color("white")
        for text in leg.get_texts():
            text.set_color("white")
    else:
        frame.set_facecolor("white")
        frame.set_edgecolor("#c8c8c8")
        frame.set_alpha(0.94)
    return leg


def plot_boxplot(ax, data, mode: str = "rgb", only_class: str | None = None):
    mask_mode = mode == "mask"
    configure_axes(ax, data, mask_mode=mask_mode)
    draw_background_patches(ax, data, mask_mode=mask_mode)

    if only_class in (None, "box"):
        draw_box_items(ax, data, mask_mode=mask_mode)
    if only_class in (None, "dot"):
        draw_dot_items(ax, data, mask_mode=mask_mode)
    if only_class in (None, "line"):
        draw_line_items(ax, data, mask_mode=mask_mode)
    if only_class in (None, "annotation"):
        draw_annotation_items(ax, data, mask_mode=mask_mode)
    if only_class in (None, "legend"):
        draw_legend(ax, data, mask_mode=mask_mode)


def apply_layout(fig, data):
    margins = dict(data["margins"])
    if data["legend"].get("family") == "top_outside":
        margins["top"] = min(margins["top"], 0.78)
    elif data["legend"].get("family") == "bottom_outside":
        margins["bottom"] = max(margins["bottom"], 0.22)
    fig.subplots_adjust(
        left=margins["left"],
        right=margins["right"],
        top=margins["top"],
        bottom=margins["bottom"],
    )


def render_rgb_image(data):
    fig = plt.figure(figsize=data["figure_size"], dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    plot_boxplot(ax, data, mode="rgb", only_class=None)
    apply_layout(fig, data)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    buffer = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    rgb = buffer.reshape((height, width, 4))[:, :, :3].copy()
    plt.close(fig)
    return rgb, width, height


def render_class_mask(data, class_name: str):
    fig = plt.figure(figsize=data["figure_size"], dpi=DPI)
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111)
    plot_boxplot(ax, data, mode="mask", only_class=class_name)
    apply_layout(fig, data)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    buffer = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    rgb = buffer.reshape((height, width, 4))[:, :, :3]
    binary_mask = (rgb.max(axis=-1) > MASK_THRESHOLD).astype(np.uint8)
    plt.close(fig)
    return binary_mask


def render_instance_mask(data, class_name: str, instance_index: int | None = None):
    instance_data = dict(data)
    instance_data["background_patches"] = []
    instance_data["legend"] = {"enabled": False}
    instance_data["box_items"] = []
    instance_data["dot_items"] = []
    instance_data["line_items"] = []
    instance_data["annotation_items"] = []

    if class_name == "box":
        instance_data["box_items"] = [data["box_items"][instance_index]]
    elif class_name == "line":
        instance_data["line_items"] = [data["line_items"][instance_index]]
    elif class_name == "annotation":
        instance_data["annotation_items"] = [data["annotation_items"][instance_index]]
    elif class_name == "legend":
        instance_data["legend"] = data["legend"]
    else:
        raise ValueError(f"Unsupported instance render for class '{class_name}'")

    return render_class_mask(instance_data, class_name)


def extract_instance_masks(data, class_name: str):
    if class_name == "box":
        return [render_instance_mask(data, class_name, idx) for idx in range(len(data["box_items"]))]
    if class_name == "dot":
        return split_connected_components(render_class_mask(data, class_name), min_area=6)
    if class_name == "line":
        return [render_instance_mask(data, class_name, idx) for idx in range(len(data["line_items"]))]
    if class_name == "annotation":
        return [render_instance_mask(data, class_name, idx) for idx in range(len(data["annotation_items"]))]
    if class_name == "legend":
        if not data["legend"]["enabled"]:
            return []
        return [render_instance_mask(data, class_name)]
    return []


def prepare_output_dirs(output_root: Path, clean_output: bool):
    if clean_output and output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_root / "annotations").mkdir(parents=True, exist_ok=True)


def summarize_annotations(output_root: Path):
    summary = {}
    for split in ("train", "val"):
        json_path = output_root / "annotations" / f"{split}.json"
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        prompt_counter = collections.Counter(item["text_input"] for item in data["annotations"])
        category_counter = collections.Counter(item["category_id"] for item in data["annotations"])
        summary[split] = {
            "images": len(data["images"]),
            "annotations": len(data["annotations"]),
            "text_inputs": dict(prompt_counter),
            "category_ids": dict(category_counter),
        }
    return summary


def create_full_dataset(
    num_charts: int = 20,
    train_ratio: float = 0.8,
    output_dir: str = "boxplot_dataset",
    clean_output: bool = True,
    preview_dir: str | None = None,
    export_preview: bool = True,
):
    output_root = Path(output_dir)
    prepare_output_dirs(output_root, clean_output=clean_output)
    preview_root = None
    if export_preview:
        preview_root = Path(preview_dir) if preview_dir else default_preview_root(output_root)
        prepare_preview_dirs(preview_root, clean_output=clean_output)

    train_cutoff = max(1, min(num_charts - 1, int(round(num_charts * train_ratio)))) if num_charts > 1 else 1
    train_images = []
    train_annotations = []
    val_images = []
    val_annotations = []
    ann_id = 1
    generation_stats = collections.Counter()

    for idx in range(num_charts):
        data = generate_boxplot_spec(idx)
        rgb_image, width, height = render_rgb_image(data)
        image_id = idx + 1
        split_name = "train" if idx < train_cutoff else "val"
        image_filename = f"{image_id:06d}.png"
        mpimg.imsave(output_root / "images" / split_name / image_filename, rgb_image)

        generation_stats[f"family:{data['family']}"] += 1
        generation_stats[f"legend:{'yes' if data['legend']['enabled'] else 'no'}"] += 1
        generation_stats[f"dot_style:{data['metadata']['dot_style']}"] += 1

        split_images = train_images if idx < train_cutoff else val_images
        split_annotations = train_annotations if idx < train_cutoff else val_annotations
        split_images.append(
            {
                "id": int(image_id),
                "file_name": image_filename,
                "width": int(width),
                "height": int(height),
            }
        )

        preview_masks = None
        if preview_root is not None:
            preview_masks = {
                class_info["category_name"]: render_class_mask(data, class_info["name"])
                for class_info in CLASS_INFO.values()
            }
        for class_id, class_info in CLASS_INFO.items():
            class_name = class_info["name"]
            for instance_mask in extract_instance_masks(data, class_name):
                bbox, area = get_bbox_and_area(instance_mask)
                if area == 0 or bbox is None:
                    continue
                split_annotations.append(
                    {
                        "id": int(ann_id),
                        "image_id": int(image_id),
                        "category_id": int(class_id),
                        "bbox": bbox,
                        "segmentation": encode_rle(instance_mask),
                        "area": area,
                        "iscrowd": 0,
                        "text_input": class_info["text_input"],
                    }
                )
                ann_id += 1

        if preview_root is not None and preview_masks is not None:
            save_preview_assets(preview_root, split_name, image_filename, rgb_image, preview_masks)

    categories = [
        {"id": int(class_id), "name": class_info["category_name"]}
        for class_id, class_info in CLASS_INFO.items()
    ]
    info = {
        "description": "Synthetic clinical box plot dataset for SAM 3 fine-tuning",
        "version": "1.0",
        "year": 2026,
        "contributor": "Generated dataset",
        "date_created": "2026-04-06",
    }

    train_json = {
        "info": info,
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories,
    }
    val_json = {
        "info": info,
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories,
    }

    with (output_root / "annotations" / "train.json").open("w", encoding="utf-8") as handle:
        json.dump(train_json, handle, indent=2)
    with (output_root / "annotations" / "val.json").open("w", encoding="utf-8") as handle:
        json.dump(val_json, handle, indent=2)
    if preview_root is not None:
        write_preview_annotations(preview_root, train_json, val_json)

    annotation_summary = summarize_annotations(output_root)
    print(f"Dataset generation complete ({num_charts} charts).")
    print(f"Output directory: {output_root}")
    if preview_root is not None:
        print(f"Preview directory: {preview_root}")
    print("Generation summary:")
    for key in sorted(generation_stats):
        print(f"  {key}: {generation_stats[key]}")
    print("Annotation summary:")
    for split, stats in annotation_summary.items():
        print(f"  {split}: {stats}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic clinical box plot dataset for SAM 3 fine-tuning.")
    parser.add_argument("--num-charts", type=int, default=20, help="Number of box plot charts to generate.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of charts to place in the train split.")
    parser.add_argument("--output-dir", type=str, default="boxplot_dataset", help="Output directory for the generated dataset.")
    parser.add_argument(
        "--preview-dir",
        type=str,
        default=None,
        help="Optional separate preview directory for RGB copies, class masks, and masked previews.",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip generating the separate preview directory with mask inspection assets.",
    )
    parser.add_argument("--keep-existing", action="store_true", help="Do not delete the output directory before generating files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_full_dataset(
        num_charts=args.num_charts,
        train_ratio=args.train_ratio,
        output_dir=args.output_dir,
        clean_output=not args.keep_existing,
        preview_dir=args.preview_dir,
        export_preview=not args.skip_preview,
    )
