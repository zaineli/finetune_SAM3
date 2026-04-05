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
from matplotlib.lines import Line2D
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
    1: {"name": "dot", "category_name": "dots", "text_input": "dots"},
    2: {"name": "line", "category_name": "lines", "text_input": "lines"},
    3: {"name": "annotation", "category_name": "annotations", "text_input": "annotations"},
    4: {"name": "legend", "category_name": "legend", "text_input": "legend"},
}

DPI = 180
MASK_THRESHOLD = 10

GENE_LABELS = [
    "TP53", "EGFR", "MYC", "BRCA1", "BRCA2", "PTEN", "STAT1", "STAT3", "PIK3CA", "TNF",
    "CXCL8", "IL6", "BCL2", "CDK1", "CDK2", "RPLP0", "RPS7", "MYBL1", "ZNF117", "HNRNPF",
    "NDUFA4", "POLR2K", "COX7C", "MIR16", "CARD8", "PHC2", "KMT2C", "STIM2", "GVINP1", "SVIL",
    "MALAT1", "PLK1", "CD52", "LCP2", "RPL36", "RPS27", "FAM126B", "PLSCR1", "IFIT3", "IFI44L",
    "ZNF621", "RPL24", "HSP90AA1", "VIM", "MMP9", "GATA3", "FOXP3", "SOX9", "COL1A1", "COL3A1",
    "CXCL10", "ISG15", "NFKB1", "JUNB", "RPS18", "RPL13A", "B2M", "HLA-DRA", "CD74", "LYZ",
]

DRUG_NAMES = [
    "Heparin Sodium", "Lisinopril", "Ibuprofen", "Atenolol", "Metformin", "Amoxicillin",
    "Prednisone", "Warfarin", "Omeprazole", "Insulin", "Clopidogrel", "Amlodipine",
]

REACTION_NAMES = [
    "RESPIRATORY DISTRESS", "CARDIOGENIC SHOCK", "HEPATIC FAILURE", "WOUND DEHISCENCE",
    "MOTOR DYSFUNCTION", "VENOUS STENOSIS", "FEELING HOT", "RECTAL HAEMORRHAGE",
    "ANXIETY", "PLATELET DISORDER", "CHEST PAIN", "NECROSIS",
    "CHRONIC FATIGUE", "MUSCLE WEAKNESS", "DIZZINESS", "HYPOTENSION",
]

DIRECTION_LABELS = [
    ("Male", "Female"),
    ("Control", "Treatment"),
    ("Low dose", "High dose"),
    ("Group A", "Group B"),
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


def make_point_group(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    edgecolor: str,
    size: float,
    alpha: float,
    marker: str = "o",
    zorder: int = 3,
):
    return {
        "name": name,
        "x": x.astype(float),
        "y": y.astype(float),
        "color": color,
        "edgecolor": edgecolor,
        "size": size,
        "alpha": alpha,
        "marker": marker,
        "zorder": zorder,
    }


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


def select_spaced_indices(
    x: np.ndarray,
    y: np.ndarray,
    candidate_indices: np.ndarray,
    max_count: int,
    min_dx: float,
    min_dy: float,
):
    selected = []
    for idx in candidate_indices:
        if all(abs(x[idx] - x[prev]) > min_dx or abs(y[idx] - y[prev]) > min_dy for prev in selected):
            selected.append(int(idx))
        if len(selected) >= max_count:
            break
    return selected


def random_gene_labels(count: int):
    return random.sample(GENE_LABELS, k=min(count, len(GENE_LABELS)))


def random_pharma_labels(count: int):
    labels = []
    for _ in range(count):
        labels.append(f"{random.choice(DRUG_NAMES)}\n{random.choice(REACTION_NAMES)}")
    return labels


def data_text_position(x: float, y: float, xlim: tuple[float, float], ylim: tuple[float, float], side_sign: float):
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    x_offset = side_sign * random.uniform(0.04, 0.09) * x_range
    y_offset = random.uniform(0.02, 0.08) * y_range
    text_x = np.clip(x + x_offset, xlim[0] + 0.06 * x_range, xlim[1] - 0.06 * x_range)
    text_y = np.clip(y + y_offset, ylim[0] + 0.08 * y_range, ylim[1] - 0.05 * y_range)
    return float(text_x), float(text_y)


def build_legend_config(style_family: str, items: list[dict], enabled: bool):
    if not enabled or not items:
        return {"enabled": False}

    if style_family == "pharma":
        return {"enabled": True, "items": items, "loc": "upper right", "bbox_to_anchor": None}
    if style_family == "classic":
        return {
            "enabled": True,
            "items": items,
            "loc": "upper left",
            "bbox_to_anchor": (0.02, 0.98),
        }
    return {"enabled": True, "items": items, "loc": "upper right", "bbox_to_anchor": None}


def configure_axes(ax, data, mask_mode: bool):
    text_color = "black"
    line_color = "black"
    grid_color = "black" if mask_mode else data["grid_color"]

    ax.clear()
    ax.set_facecolor("black" if mask_mode else data["facecolor"])
    ax.set_xlim(*data["xlim"])
    ax.set_ylim(*data["ylim"])
    ax.set_xlabel(data["xlabel"], fontsize=data["label_fontsize"], color=text_color)
    ax.set_ylabel(data["ylabel"], fontsize=data["label_fontsize"], color=text_color)

    if data["title"]:
        ax.set_title(data["title"], fontsize=data["title_fontsize"], color=text_color, pad=8)

    if data["xticks"] is not None:
        ax.set_xticks(data["xticks"])
    if data["yticks"] is not None:
        ax.set_yticks(data["yticks"])

    ax.tick_params(colors=text_color, labelsize=data["tick_fontsize"], width=0.8, length=5)
    for spine_name, spine in ax.spines.items():
        spine.set_linewidth(0.8)
        spine.set_color(line_color)
        if data["hide_top_right"] and spine_name in {"top", "right"}:
            spine.set_visible(False)

    if data["show_grid"]:
        ax.grid(
            True,
            linestyle=data["grid_linestyle"],
            linewidth=data["grid_linewidth"],
            color=grid_color,
            alpha=1.0 if mask_mode else data["grid_alpha"],
            axis=data["grid_axis"],
        )
    else:
        ax.grid(False)

    ax.set_axisbelow(True)


def draw_points(ax, data, mask_mode: bool):
    for group in data["point_groups"]:
        if group["x"].size == 0:
            continue
        color = "white" if mask_mode else group["color"]
        edge = "white" if mask_mode else group["edgecolor"]
        ax.scatter(
            group["x"],
            group["y"],
            s=group["size"] + (7 if mask_mode else 0),
            c=color,
            edgecolors=edge,
            linewidths=0.7 if mask_mode else 0.55,
            alpha=1.0 if mask_mode else group["alpha"],
            marker=group["marker"],
            zorder=group["zorder"],
        )


def draw_line_items(ax, data, mask_mode: bool):
    for item in data["line_items"]:
        color = "white" if mask_mode else item["color"]
        linewidth = item["linewidth"] + (0.5 if mask_mode else 0.0)

        if item["kind"] == "vline":
            ax.axvline(
                item["x"],
                color=color,
                linestyle=item["linestyle"],
                linewidth=linewidth,
                alpha=1.0 if mask_mode else item["alpha"],
                zorder=item.get("zorder", 2),
            )
        elif item["kind"] == "hline":
            ax.axhline(
                item["y"],
                color=color,
                linestyle=item["linestyle"],
                linewidth=linewidth,
                alpha=1.0 if mask_mode else item["alpha"],
                zorder=item.get("zorder", 2),
            )
        elif item["kind"] == "axes_arrow":
            ax.annotate(
                "",
                xy=item["xy"],
                xytext=item["xytext"],
                xycoords="axes fraction",
                textcoords="axes fraction",
                annotation_clip=False,
                arrowprops={
                    "arrowstyle": item["arrowstyle"],
                    "lw": linewidth,
                    "color": color,
                    "linestyle": item["linestyle"],
                    "alpha": 1.0 if mask_mode else item["alpha"],
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                },
            )
        elif item["kind"] == "axes_line":
            ax.plot(
                item["xs"],
                item["ys"],
                color=color,
                linewidth=linewidth,
                linestyle=item["linestyle"],
                alpha=1.0 if mask_mode else item["alpha"],
                transform=ax.transAxes,
                clip_on=False,
                zorder=item.get("zorder", 2),
            )
        elif item["kind"] == "data_line":
            ax.plot(
                item["xs"],
                item["ys"],
                color=color,
                linewidth=linewidth,
                linestyle=item["linestyle"],
                alpha=1.0 if mask_mode else item["alpha"],
                zorder=item.get("zorder", 2),
            )


def draw_annotations(ax, data, mask_mode: bool):
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


def draw_legend(ax, data, mask_mode: bool):
    legend = data["legend"]
    if not legend["enabled"]:
        return

    handles = []
    for item in legend["items"]:
        color = "white" if mask_mode else item["color"]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=item["marker"],
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                linestyle="",
                markersize=item["markersize"],
                label=item["label"],
            )
        )

    leg = ax.legend(
        handles=handles,
        loc=legend["loc"],
        bbox_to_anchor=legend["bbox_to_anchor"],
        frameon=True,
        fontsize=data["legend_fontsize"],
        title=legend.get("title"),
        title_fontsize=data["legend_fontsize"] + 0.5,
    )
    frame = leg.get_frame()
    if mask_mode:
        frame.set_facecolor("white")
        frame.set_edgecolor("white")
        frame.set_alpha(1.0)
        if legend.get("title"):
            leg.get_title().set_color("white")
        for text in leg.get_texts():
            text.set_color("white")
    else:
        frame.set_facecolor("white")
        frame.set_edgecolor("#c7c7c7")
        frame.set_alpha(0.92)


def plot_volcano(ax, data, mode: str = "rgb", only_class: str | None = None):
    mask_mode = mode == "mask"
    configure_axes(ax, data, mask_mode=mask_mode)

    if only_class in (None, "dot"):
        draw_points(ax, data, mask_mode=mask_mode)
    if only_class in (None, "line"):
        draw_line_items(ax, data, mask_mode=mask_mode)
    if only_class in (None, "annotation"):
        draw_annotations(ax, data, mask_mode=mask_mode)
    if only_class in (None, "legend"):
        draw_legend(ax, data, mask_mode=mask_mode)


def apply_layout(fig, data):
    fig.subplots_adjust(
        left=data["margins"]["left"],
        right=data["margins"]["right"],
        top=data["margins"]["top"],
        bottom=data["margins"]["bottom"],
    )


def render_rgb_image(data):
    fig = plt.figure(figsize=data["figure_size"], dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    plot_volcano(ax, data, mode="rgb", only_class=None)
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
    plot_volcano(ax, data, mode="mask", only_class=class_name)
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
    single_data = dict(data)

    if class_name == "line":
        single_data["line_items"] = [data["line_items"][instance_index]]
        single_data["point_groups"] = []
        single_data["annotation_items"] = []
        single_data["legend"] = {"enabled": False}
    elif class_name == "annotation":
        single_data["line_items"] = []
        single_data["point_groups"] = []
        single_data["annotation_items"] = [data["annotation_items"][instance_index]]
        single_data["legend"] = {"enabled": False}
    elif class_name == "legend":
        single_data["line_items"] = []
        single_data["point_groups"] = []
        single_data["annotation_items"] = []
    else:
        raise ValueError(f"Unsupported explicit instance render for class '{class_name}'")

    return render_class_mask(single_data, class_name)


def extract_instance_masks(data, class_name: str):
    if class_name == "dot":
        return split_connected_components(render_class_mask(data, class_name), min_area=4)
    if class_name == "line":
        return [render_instance_mask(data, class_name, idx) for idx in range(len(data["line_items"]))]
    if class_name == "annotation":
        return [render_instance_mask(data, class_name, idx) for idx in range(len(data["annotation_items"]))]
    if class_name == "legend":
        if not data["legend"]["enabled"]:
            return []
        return [render_instance_mask(data, class_name)]
    return []


def build_label_annotations(
    x: np.ndarray,
    y: np.ndarray,
    candidates: np.ndarray,
    labels: list[str],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    color: str = "black",
    fontsize: float = 8.0,
):
    items = []
    x_range = xlim[1] - xlim[0]
    for idx, label in zip(candidates, labels):
        side_sign = 1.0 if x[idx] >= 0 else -1.0
        if x[idx] > xlim[1] - 0.16 * x_range:
            side_sign = -1.0
        elif x[idx] < xlim[0] + 0.16 * x_range:
            side_sign = 1.0
        text_x, text_y = data_text_position(x[idx], y[idx], xlim, ylim, side_sign=side_sign)
        items.append(
            make_text_item(
                label,
                text_x,
                text_y,
                coords="data",
                fontsize=fontsize,
                ha="left" if side_sign > 0 else "right",
                va="center",
                color=color,
            )
        )
    return items


def build_classic_spec(force_legend: bool = False):
    xlim = random.choice([(-6.2, 6.2), (-5.6, 5.6), (-4.8, 4.8)])
    ylim = (0.0, random.uniform(16.0, 22.0))
    x_thr = random.choice([0.8, 1.0, 1.2])
    y_thr = random.choice([1.15, 1.3, 1.5])
    n_points = random.randint(2200, 4200)

    core_n = int(n_points * random.uniform(0.68, 0.78))
    tail_n = n_points - core_n
    left_prob = random.uniform(0.44, 0.56)
    x_core = np.random.normal(0.0, random.uniform(0.45, 0.8), core_n)
    tail_sign = np.where(np.random.rand(tail_n) < left_prob, -1.0, 1.0)
    x_tail = tail_sign * (np.random.gamma(shape=2.0, scale=random.uniform(0.34, 0.55), size=tail_n) + 0.25)
    x = np.clip(np.concatenate([x_core, x_tail]), xlim[0] + 0.1, xlim[1] - 0.1)

    base = np.random.gamma(shape=1.4, scale=random.uniform(0.42, 0.6), size=n_points)
    effect = np.maximum(0.0, np.abs(x) - random.uniform(0.18, 0.35)) ** random.uniform(1.5, 1.9)
    y = base + effect * random.uniform(1.8, 2.5) + np.random.normal(0.0, 0.12, n_points)
    y += (np.random.rand(n_points) < random.uniform(0.04, 0.08)) * np.random.gamma(2.2, 1.6, size=n_points)
    y = np.clip(y, 0.0, ylim[1] - 0.1)

    left_sig = (x <= -x_thr) & (y >= y_thr)
    right_sig = (x >= x_thr) & (y >= y_thr)
    background = ~(left_sig | right_sig)

    point_groups = [
        make_point_group("background", x[background], y[background], "#c8c8c8", "#c8c8c8", random.uniform(8, 12), 0.72),
        make_point_group("left_sig", x[left_sig], y[left_sig], "#de2d26", "#de2d26", random.uniform(9, 13), 0.78),
        make_point_group("right_sig", x[right_sig], y[right_sig], "#4768b3", "#4768b3", random.uniform(9, 13), 0.78),
    ]

    line_items = [
        make_line_item("vline", x=0.0, color="#1f1f1f", linestyle="-", linewidth=2.0, alpha=0.95, zorder=2),
        make_line_item("vline", x=-x_thr, color="#9a9a9a", linestyle="--", linewidth=1.0, alpha=0.75, zorder=2),
        make_line_item("vline", x=x_thr, color="#9a9a9a", linestyle="--", linewidth=1.0, alpha=0.75, zorder=2),
        make_line_item("hline", y=y_thr, color="#9a9a9a", linestyle="--", linewidth=1.0, alpha=0.75, zorder=2),
    ]

    annotation_items = []
    margins = {"left": 0.12, "right": 0.96, "top": 0.93, "bottom": 0.13}

    annotation_mode = random.choices(
        ["none", "top_explainer", "threshold_note", "both"],
        weights=[0.28, 0.26, 0.22, 0.24],
        k=1,
    )[0]

    if annotation_mode in {"top_explainer", "both"}:
        annotation_items.extend(
            [
                make_text_item("Negative change\nvs control", 0.03, 0.95, coords="axes", fontsize=10.5, ha="left", va="top"),
                make_text_item("Zero point", 0.5, 0.96, coords="axes", fontsize=10.5, ha="center", va="top"),
                make_text_item("Positive change\nvs control", 0.97, 0.95, coords="axes", fontsize=10.5, ha="right", va="top"),
            ]
        )
        line_items.extend(
            [
                make_line_item("axes_arrow", xy=(0.28, 0.975), xytext=(0.49, 0.975), color="#6aa6df", linestyle="-", linewidth=1.6, alpha=0.95, arrowstyle="->"),
                make_line_item("axes_arrow", xy=(0.72, 0.975), xytext=(0.51, 0.975), color="#6aa6df", linestyle="-", linewidth=1.6, alpha=0.95, arrowstyle="->"),
            ]
        )

    if annotation_mode in {"threshold_note", "both"}:
        y_axes = float((y_thr - ylim[0]) / (ylim[1] - ylim[0]))
        annotation_items.extend(
            [
                make_text_item("Statistically significant\nchange", 0.03, min(0.86, y_axes + 0.07), coords="axes", fontsize=9.3, ha="left", va="bottom"),
                make_text_item("Statistically insignificant\nchange", 0.03, max(0.09, y_axes - 0.06), coords="axes", fontsize=9.3, ha="left", va="top"),
            ]
        )
        line_items.append(
            make_line_item(
                "axes_arrow",
                xy=(0.16, min(0.85, y_axes + 0.01)),
                xytext=(0.16, max(0.04, y_axes - 0.01)),
                color="#6aa6df",
                linestyle="-",
                linewidth=1.6,
                alpha=0.95,
                arrowstyle="<->",
            )
        )

    label_candidates = np.where(left_sig | right_sig)[0]
    label_order = label_candidates[np.argsort(y[label_candidates])[::-1]]
    selected = select_spaced_indices(
        x,
        y,
        label_order,
        max_count=random.randint(4, 8),
        min_dx=(xlim[1] - xlim[0]) * 0.08,
        min_dy=(ylim[1] - ylim[0]) * 0.06,
    )
    annotation_items.extend(
        build_label_annotations(x, y, selected, random_gene_labels(len(selected)), xlim, ylim, fontsize=7.8)
    )

    legend_enabled = (force_legend or random.random() < 0.22) and annotation_mode != "both"
    legend_items = [
        {"label": "Downregulated", "color": "#de2d26", "marker": "o", "markersize": 6.0},
        {"label": "Upregulated", "color": "#4768b3", "marker": "o", "markersize": 6.0},
        {"label": "Not significant", "color": "#bdbdbd", "marker": "o", "markersize": 6.0},
    ]

    return {
        "style_family": "classic",
        "figure_size": random.choice([(7.8, 6.8), (8.4, 7.2), (9.0, 7.0)]),
        "xlim": xlim,
        "ylim": ylim,
        "xlabel": random.choice(["Fold Change (log2)", "log2 Fold Change", "logFC"]),
        "ylabel": random.choice(["-log10 p-value", "-log10(P-value)", "-log10 Adjusted P"]),
        "title": random.choice([None, "Volcano Plot", "Differential Expression", "RNA-seq Volcano Plot"]),
        "title_fontsize": 13.5,
        "label_fontsize": 12.0,
        "tick_fontsize": 10.0,
        "xticks": None,
        "yticks": None,
        "show_grid": random.random() < 0.35,
        "grid_axis": "both",
        "grid_linestyle": ":",
        "grid_linewidth": 0.7,
        "grid_alpha": 0.55,
        "grid_color": "#dbdbdb",
        "hide_top_right": random.random() < 0.25,
        "facecolor": "white",
        "point_groups": point_groups,
        "line_items": line_items,
        "annotation_items": annotation_items,
        "legend": build_legend_config("classic", legend_items, legend_enabled),
        "legend_fontsize": 8.8,
        "margins": margins,
        "metadata": {
            "style": "classic",
            "x_threshold": x_thr,
            "y_threshold": y_thr,
            "legend_enabled": legend_enabled,
            "annotation_mode": annotation_mode,
        },
    }


def build_compact_spec():
    xlim = (-1.6, 1.6)
    ylim = (0.85, random.uniform(3.9, 4.25))
    x_thr = random.uniform(0.16, 0.28)
    y_thr = random.choice([1.25, 1.3, 1.35])
    n_points = random.randint(1800, 3200)

    core_n = int(n_points * random.uniform(0.72, 0.82))
    tail_n = n_points - core_n
    x_core = np.random.normal(0.0, random.uniform(0.12, 0.22), core_n)
    tail_sign = np.where(np.random.rand(tail_n) < 0.48, -1.0, 1.0)
    x_tail = tail_sign * (np.random.gamma(shape=2.2, scale=random.uniform(0.12, 0.18), size=tail_n) + 0.05)
    x = np.clip(np.concatenate([x_core, x_tail]), xlim[0] + 0.02, xlim[1] - 0.02)

    base = 0.86 + np.random.gamma(shape=1.35, scale=random.uniform(0.12, 0.18), size=n_points)
    effect = np.maximum(0.0, np.abs(x) - 0.05) ** random.uniform(1.15, 1.35)
    y = base + effect * random.uniform(1.9, 2.5) + np.random.normal(0.0, 0.05, n_points)
    y += (np.random.rand(n_points) < random.uniform(0.03, 0.06)) * np.random.gamma(1.7, 0.45, size=n_points)
    y = np.clip(y, ylim[0], ylim[1] - 0.02)

    sig_mask = (np.abs(x) >= x_thr) & (y >= y_thr)
    highlight_source = sig_mask & (y >= np.quantile(y[sig_mask], 0.92) if np.any(sig_mask) else False)
    highlight_candidates = np.where(highlight_source)[0]
    highlight_count = min(len(highlight_candidates), random.randint(10, 16))
    highlight_indices = set(
        select_spaced_indices(
            x,
            y,
            highlight_candidates[np.argsort(y[highlight_candidates])[::-1]] if len(highlight_candidates) else np.array([], dtype=int),
            max_count=highlight_count,
            min_dx=(xlim[1] - xlim[0]) * 0.06,
            min_dy=(ylim[1] - ylim[0]) * 0.05,
        )
    )
    lowlight_candidates = np.where((~sig_mask) & (np.abs(x) > 1.0) & (y < y_thr + 0.1))[0]
    lowlight_count = min(len(lowlight_candidates), random.randint(3, 8))
    lowlight_indices = set(
        random.sample(list(lowlight_candidates), k=lowlight_count) if lowlight_count > 0 else []
    )

    background_mask = (~sig_mask) & (~np.isin(np.arange(n_points), list(lowlight_indices)))
    significant_mask = sig_mask & (~np.isin(np.arange(n_points), list(highlight_indices)))
    highlight_mask = np.isin(np.arange(n_points), list(highlight_indices))
    lowlight_mask = np.isin(np.arange(n_points), list(lowlight_indices))

    point_groups = [
        make_point_group("background", x[background_mask], y[background_mask], "#111111", "#111111", random.uniform(5.8, 8.0), 0.88),
        make_point_group("significant", x[significant_mask], y[significant_mask], "#e41a1c", "#e41a1c", random.uniform(6.2, 8.5), 0.92),
        make_point_group("highlight", x[highlight_mask], y[highlight_mask], "#4daf4a", "#4daf4a", random.uniform(8.5, 11.0), 0.98),
        make_point_group("lowlight", x[lowlight_mask], y[lowlight_mask], "#d9a520", "#d9a520", random.uniform(6.8, 9.2), 0.95),
    ]

    line_items = [
        make_line_item("vline", x=0.0, color="#000000", linestyle="-", linewidth=2.6, alpha=0.98, zorder=2),
        make_line_item("vline", x=-1.0, color="#4d4d4d", linestyle=":", linewidth=1.0, alpha=0.85, zorder=1),
        make_line_item("vline", x=1.0, color="#4d4d4d", linestyle=":", linewidth=1.0, alpha=0.85, zorder=1),
        make_line_item("hline", y=y_thr, color="#4d4d4d", linestyle=":", linewidth=1.0, alpha=0.85, zorder=1),
    ]

    annotation_items = []
    selected_labels = select_spaced_indices(
        x,
        y,
        np.where(highlight_mask)[0][np.argsort(y[highlight_mask])[::-1]] if np.any(highlight_mask) else np.array([], dtype=int),
        max_count=min(random.randint(8, 14), int(np.sum(highlight_mask))),
        min_dx=(xlim[1] - xlim[0]) * 0.07,
        min_dy=(ylim[1] - ylim[0]) * 0.06,
    )
    annotation_items.extend(
        build_label_annotations(x, y, selected_labels, random_gene_labels(len(selected_labels)), xlim, ylim, color="#2f8f2f", fontsize=7.0)
    )

    if random.random() < 0.75:
        y_axes = float((y_thr - ylim[0]) / (ylim[1] - ylim[0]))
        line_items.extend(
            [
                make_line_item(
                    "axes_arrow",
                    xy=(1.03, 0.93),
                    xytext=(1.03, min(0.92, y_axes + 0.05)),
                    color="#000000",
                    linestyle="-",
                    linewidth=1.2,
                    alpha=0.95,
                    arrowstyle="<->",
                ),
                make_line_item(
                    "axes_arrow",
                    xy=(1.03, max(0.06, y_axes - 0.01)),
                    xytext=(1.03, 0.03),
                    color="#000000",
                    linestyle="-",
                    linewidth=1.2,
                    alpha=0.95,
                    arrowstyle="<->",
                ),
            ]
        )
        annotation_items.extend(
            [
                make_text_item("Significant\nexpression", 1.09, 0.62, coords="axes", fontsize=10.0, ha="left", va="center"),
                make_text_item("Background", 1.09, 0.11, coords="axes", fontsize=10.0, ha="left", va="center"),
            ]
        )
        margins = {"left": 0.12, "right": 0.81, "top": 0.92, "bottom": 0.14}
    else:
        margins = {"left": 0.12, "right": 0.96, "top": 0.92, "bottom": 0.14}

    title = random.choice(["Volcano plot", "Compact Volcano Plot", "Differential Expression"])
    if random.random() < 0.25:
        title = None

    return {
        "style_family": "compact",
        "figure_size": random.choice([(9.2, 5.8), (8.8, 6.0), (9.6, 6.1)]),
        "xlim": xlim,
        "ylim": ylim,
        "xlabel": random.choice(["logFC", "log2 Fold Change", "Fold Change"]),
        "ylabel": random.choice(["-log10(P-value)", "-Log10 (P-value)", "-log10 p-value"]),
        "title": title,
        "title_fontsize": 12.2,
        "label_fontsize": 10.8,
        "tick_fontsize": 8.5,
        "xticks": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        "yticks": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "show_grid": False,
        "grid_axis": "both",
        "grid_linestyle": ":",
        "grid_linewidth": 0.6,
        "grid_alpha": 0.4,
        "grid_color": "#dddddd",
        "hide_top_right": False,
        "facecolor": "white",
        "point_groups": point_groups,
        "line_items": line_items,
        "annotation_items": annotation_items,
        "legend": {"enabled": False},
        "legend_fontsize": 8.2,
        "margins": margins,
        "metadata": {
            "style": "compact",
            "x_threshold": x_thr,
            "y_threshold": y_thr,
            "legend_enabled": False,
            "annotation_mode": "compact_labels",
        },
    }


def build_pharma_spec(force_legend: bool = False):
    xlim = (-10.8, 10.8)
    ylim = (0.0, random.uniform(68.0, 92.0))
    x_thr = 1.0
    y_thr = random.uniform(4.8, 6.2)
    n_points = random.randint(650, 1200)

    core_n = int(n_points * random.uniform(0.62, 0.72))
    tail_n = n_points - core_n
    right_bias = random.uniform(0.56, 0.72)
    x_core = np.random.normal(0.0, random.uniform(0.55, 0.9), core_n)
    tail_sign = np.where(np.random.rand(tail_n) < right_bias, 1.0, -1.0)
    x_tail = tail_sign * (np.random.gamma(shape=1.9, scale=random.uniform(0.95, 1.4), size=tail_n) + 0.2)
    x = np.clip(np.concatenate([x_core, x_tail]), xlim[0] + 0.1, xlim[1] - 0.1)

    base = np.random.gamma(shape=1.45, scale=random.uniform(2.2, 3.0), size=n_points)
    effect = np.maximum(0.0, np.abs(x) - 0.1) ** random.uniform(1.18, 1.45)
    y = base + effect * random.uniform(2.2, 3.6) + np.random.normal(0.0, 0.45, n_points)
    y += (np.random.rand(n_points) < random.uniform(0.07, 0.1)) * np.random.gamma(2.4, 8.0, size=n_points)
    y = np.clip(y, 0.0, ylim[1] - 0.1)

    left_sig = (x <= -x_thr) & (y >= y_thr)
    right_sig = (x >= x_thr) & (y >= y_thr)
    background = ~(left_sig | right_sig)

    point_groups = [
        make_point_group("background", x[background], y[background], "#b7b7b7", "#555555", random.uniform(18, 24), 0.9),
        make_point_group("left_sig", x[left_sig], y[left_sig], "#e41a1c", "#111111", random.uniform(20, 28), 0.92),
        make_point_group("right_sig", x[right_sig], y[right_sig], "#2b2be8", "#111111", random.uniform(20, 28), 0.92),
    ]

    line_items = [
        make_line_item("vline", x=-x_thr, color="#9bbf49", linestyle=(0, (6, 4)), linewidth=1.8, alpha=0.95, zorder=2),
        make_line_item("vline", x=x_thr, color="#9bbf49", linestyle=(0, (6, 4)), linewidth=1.8, alpha=0.95, zorder=2),
        make_line_item("hline", y=y_thr, color="#9bbf49", linestyle=(0, (6, 4)), linewidth=1.6, alpha=0.95, zorder=2),
    ]

    annotation_items = []
    direction_left, direction_right = random.choice(DIRECTION_LABELS)
    annotation_items.extend(
        [
            make_text_item(direction_left, 0.24, -0.16, coords="axes", fontsize=22.0, ha="center", va="center"),
            make_text_item(direction_right, 0.76, -0.16, coords="axes", fontsize=22.0, ha="center", va="center"),
            make_text_item(random.choice(["Adjusted\nP = 0.05", "Adjusted\nP-value cutoff"]), -0.02, max(0.04, (y_thr - ylim[0]) / (ylim[1] - ylim[0]) + 0.03), coords="axes", fontsize=9.5, ha="right", va="center"),
        ]
    )
    line_items.extend(
        [
            make_line_item("axes_arrow", xy=(0.48, -0.12), xytext=(0.02, -0.12), color="#d6453b", linestyle="-", linewidth=1.8, alpha=0.95, arrowstyle="-|>"),
            make_line_item("axes_arrow", xy=(0.52, -0.12), xytext=(0.98, -0.12), color="#2d63d6", linestyle="-", linewidth=1.8, alpha=0.95, arrowstyle="-|>"),
        ]
    )

    label_candidates = np.where(right_sig | left_sig)[0]
    label_order = label_candidates[np.argsort(y[label_candidates])[::-1]]
    selected = select_spaced_indices(
        x,
        y,
        label_order,
        max_count=random.randint(7, 11),
        min_dx=(xlim[1] - xlim[0]) * 0.08,
        min_dy=(ylim[1] - ylim[0]) * 0.07,
    )
    annotation_items.extend(
        build_label_annotations(x, y, selected, random_pharma_labels(len(selected)), xlim, ylim, fontsize=7.0)
    )

    legend_enabled = force_legend or random.random() < 0.18
    legend_items = [
        {"label": direction_left + "-biased", "color": "#e41a1c", "marker": "o", "markersize": 6.4},
        {"label": direction_right + "-biased", "color": "#2b2be8", "marker": "o", "markersize": 6.4},
        {"label": "Neutral", "color": "#b7b7b7", "marker": "o", "markersize": 6.4},
    ]

    return {
        "style_family": "pharma",
        "figure_size": random.choice([(8.3, 7.2), (8.8, 7.0), (9.2, 7.3)]),
        "xlim": xlim,
        "ylim": ylim,
        "xlabel": random.choice(["Log2 ROR", "log2 Reporting Odds Ratio", "Log2 Odds Ratio"]),
        "ylabel": random.choice(["-Log10 Adjusted P Value", "-Log10 P Value", "-log10 Adjusted P-value"]),
        "title": random.choice([None, None, "Volcano Plot for Safety Signals"]),
        "title_fontsize": 13.0,
        "label_fontsize": 13.2,
        "tick_fontsize": 10.2,
        "xticks": [-10, -5, 0, 5, 10],
        "yticks": None,
        "show_grid": False,
        "grid_axis": "both",
        "grid_linestyle": ":",
        "grid_linewidth": 0.6,
        "grid_alpha": 0.35,
        "grid_color": "#dddddd",
        "hide_top_right": False,
        "facecolor": "white",
        "point_groups": point_groups,
        "line_items": line_items,
        "annotation_items": annotation_items,
        "legend": build_legend_config("pharma", legend_items, legend_enabled),
        "legend_fontsize": 8.0,
        "margins": {"left": 0.12, "right": 0.94, "top": 0.94, "bottom": 0.22},
        "metadata": {
            "style": "pharma",
            "x_threshold": x_thr,
            "y_threshold": y_thr,
            "legend_enabled": legend_enabled,
            "annotation_mode": "directional",
        },
    }


def generate_volcano_spec(seed: int, force_legend: bool = False):
    random.seed(seed)
    np.random.seed(seed)

    family = random.choices(
        ["classic", "compact", "pharma"],
        weights=[0.46, 0.28, 0.26],
        k=1,
    )[0]

    if force_legend and family == "compact":
        family = random.choice(["classic", "pharma"])

    if family == "classic":
        return build_classic_spec(force_legend=force_legend)
    if family == "compact":
        return build_compact_spec()
    return build_pharma_spec(force_legend=force_legend)


def create_full_dataset(
    num_charts: int = 20,
    train_ratio: float = 0.8,
    output_dir: str = "volcano_dataset",
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
        data = generate_volcano_spec(idx, force_legend=(idx % 7 == 0))
        rgb_image, width, height = render_rgb_image(data)
        image_id = idx + 1
        split_name = "train" if idx < train_cutoff else "val"
        image_filename = f"{image_id:06d}.png"
        mpimg.imsave(output_root / "images" / split_name / image_filename, rgb_image)

        generation_stats[f"style:{data['metadata']['style']}"] += 1
        generation_stats[f"legend:{'yes' if data['metadata']['legend_enabled'] else 'no'}"] += 1
        generation_stats[f"annotation:{data['metadata']['annotation_mode']}"] += 1

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

        preview_masks = {} if preview_root is not None else None
        for class_id, class_info in CLASS_INFO.items():
            class_name = class_info["name"]
            if preview_masks is not None:
                preview_masks[class_info["category_name"]] = render_class_mask(data, class_name)
            for instance_mask in extract_instance_masks(data, class_name):
                bbox, area = get_bbox_and_area(instance_mask)
                if area == 0 or bbox is None:
                    continue

                annotation_entry = {
                    "id": int(ann_id),
                    "image_id": int(image_id),
                    "category_id": int(class_id),
                    "bbox": bbox,
                    "segmentation": encode_rle(instance_mask),
                    "area": area,
                    "iscrowd": 0,
                    "text_input": class_info["text_input"],
                }
                ann_id += 1
                split_annotations.append(annotation_entry)

        if preview_root is not None and preview_masks is not None:
            save_preview_assets(preview_root, split_name, image_filename, rgb_image, preview_masks)

    categories = [
        {"id": int(class_id), "name": class_info["category_name"]}
        for class_id, class_info in CLASS_INFO.items()
    ]
    info = {
        "description": "Synthetic volcano plot dataset for SAM 3 fine-tuning",
        "version": "2.0",
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
    parser = argparse.ArgumentParser(description="Generate a synthetic volcano plot dataset for SAM 3 fine-tuning.")
    parser.add_argument("--num-charts", type=int, default=20, help="Number of volcano charts to generate.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of charts written into train.json. The remainder goes into val.json.",
    )
    parser.add_argument("--output-dir", type=str, default="volcano_dataset", help="Dataset output directory.")
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
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep existing files instead of cleaning the output directory before generation.",
    )
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
