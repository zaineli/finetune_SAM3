import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.image as mpimg
import os
import json
import random

# ──────────────────────────────────────────────────────────────────────────────
#  CLASS DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
CLASS_INFO = {
    1: {"name": "bars",   "color": np.array([255, 0,   0],   dtype=np.uint8)},
    2: {"name": "dots",   "color": np.array([0,   255, 0],   dtype=np.uint8)},
    3: {"name": "lines",  "color": np.array([0,   0,   255], dtype=np.uint8)},
    4: {"name": "legend", "color": np.array([255, 255, 0],   dtype=np.uint8)},
}

CLASS_COLOR_FLOAT = {
    "bars":   (1.0, 0.0, 0.0),
    "dots":   (0.0, 1.0, 0.0),
    "lines":  (0.0, 0.0, 1.0),
    "legend": (1.0, 1.0, 0.0),
}

# ──────────────────────────────────────────────────────────────────────────────
#  DATA GENERATION ── controls plot variation
# ──────────────────────────────────────────────────────────────────────────────
def generate_data(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    num_groups      = random.randint(3, 8)
    num_categories  = random.randint(2, 5)
    bar_width       = 0.75 / num_categories
    bar_offsets     = np.linspace(-((num_categories-1)/2)*bar_width,
                                  ((num_categories-1)/2)*bar_width,
                                  num_categories)
    x_base          = np.arange(num_groups, dtype=float)

    means           = np.random.uniform(3.0, 20.0, (num_groups, num_categories))
    has_errorbars   = random.random() < 0.65
    stds            = np.random.uniform(0.3, 2.8, (num_groups, num_categories)) if has_errorbars else np.zeros_like(means)

    num_points_per  = random.randint(7, 28)
    jitter_amount   = random.uniform(0.035, 0.24)

    # Jittered points (one array per category)
    jitter_xs = []
    jitter_ys = []
    for c in range(num_categories):
        xs_c, ys_c = [], []
        for g in range(num_groups):
            base_x = x_base[g] + bar_offsets[c]
            base_y = means[g, c]
            jx = base_x + np.random.uniform(-jitter_amount, jitter_amount, num_points_per)
            jy = base_y + np.random.normal(0.0, max(0.5, stds[g, c]*0.8), num_points_per)
            xs_c.append(jx); ys_c.append(jy)
        jitter_xs.append(np.concatenate(xs_c))
        jitter_ys.append(np.concatenate(ys_c))

    # Errorbar data
    error_xs    = [x_base + bar_offsets[c] for c in range(num_categories)]
    error_ys    = [means[:, c] for c in range(num_categories)]
    error_yerrs = [stds[:, c] for c in range(num_categories)] if has_errorbars else None

    # Visual / layout parameters
    cmap_names = ['tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'Pastel1', 'Paired']
    cmap = plt.get_cmap(random.choice(cmap_names))
    cat_labels = [f"Category {chr(65 + c)}" for c in range(num_categories)]
    legend_loc_options = ['upper right', 'upper left', 'lower right', 'lower left',
                          'center left', 'center right', 'lower center', 'upper center', 'center']
    legend_loc = random.choice(legend_loc_options)

    max_y = means.max() + (stds.max() if has_errorbars else 2.0) * 1.2 + 3.0

    return {
        'x_base': x_base,
        'bar_offsets': bar_offsets,
        'bar_width': bar_width,
        'means': means,
        'cat_labels': cat_labels,
        'jitter_xs': jitter_xs,
        'jitter_ys': jitter_ys,
        'has_errorbars': has_errorbars,
        'error_xs': error_xs,
        'error_ys': error_ys,
        'error_yerrs': error_yerrs,
        'legend_loc': legend_loc,
        'cmap': cmap,
        'max_y': max_y,
        'num_categories': num_categories
    }

# ──────────────────────────────────────────────────────────────────────────────
#  PLOTTING LOGIC (used for both RGB image and semantic mask)
# ──────────────────────────────────────────────────────────────────────────────
def plot_chart(ax, data, is_mask: bool = False):
    ax.clear()
    ax.set_xlabel("Groups")
    ax.set_ylabel("Measurement")
    title_options = ["Performance Metrics", "Experimental Results", "Sales Overview",
                     "Jittered Summary Plot", "Comparative Analysis", "Benchmark Data"]
    ax.set_title(random.choice(title_options))
    ax.set_xticks(data['x_base'])
    ax.set_xticklabels([f"G{g+1}" for g in range(len(data['x_base']))])
    ax.set_ylim(0, data['max_y'])

    num_cat = data['num_categories']
    cat_colors = [data['cmap'](float(c)/max(1, num_cat-1)) for c in range(num_cat)] if not is_mask else None

    # Bars
    for c in range(num_cat):
        xs = data['x_base'] + data['bar_offsets'][c]
        heights = data['means'][:, c]
        color = CLASS_COLOR_FLOAT["bars"] if is_mask else cat_colors[c]
        edge  = CLASS_COLOR_FLOAT["bars"] if is_mask else 'k'
        ax.bar(xs, heights, data['bar_width'], color=color, edgecolor=edge,
               label=data['cat_labels'][c] if not is_mask else None)

    # Jittered dots
    for c in range(num_cat):
        color     = CLASS_COLOR_FLOAT["dots"] if is_mask else cat_colors[c]
        edgecolor = CLASS_COLOR_FLOAT["dots"] if is_mask else 'k'
        alpha     = 1.0 if is_mask else 0.8
        s         = 22 if is_mask else 14
        ax.scatter(data['jitter_xs'][c], data['jitter_ys'][c], s=s,
                   color=color, edgecolor=edgecolor, alpha=alpha, zorder=3)

    # Error bars
    if data['has_errorbars']:
        for c in range(num_cat):
            ecolor = CLASS_COLOR_FLOAT["lines"] if is_mask else cat_colors[c]
            ax.errorbar(data['error_xs'][c], data['error_ys'][c],
                        yerr=data['error_yerrs'][c],
                        fmt='none', ecolor=ecolor, elinewidth=2.0,
                        capsize=5, capthick=1.5)

    # Legend
    ax.legend(loc=data['legend_loc'], title="Categories", fontsize=9)

def recolor_legend_and_lines(ax_mask):
    # Force error-bar lines to class color
    for line in ax_mask.get_lines():
        line.set_color(CLASS_COLOR_FLOAT["lines"])
        line.set_alpha(1.0)

    # Recolor legend children
    leg = ax_mask.get_legend()
    if leg is not None:
        for artist in list(leg.get_children()) + leg.get_texts():
            if hasattr(artist, "set_color"):
                artist.set_color(CLASS_COLOR_FLOAT["legend"])
            if hasattr(artist, "set_facecolor"):
                artist.set_facecolor(CLASS_COLOR_FLOAT["legend"])
            if hasattr(artist, "set_edgecolor"):
                artist.set_edgecolor(CLASS_COLOR_FLOAT["legend"])

# ──────────────────────────────────────────────────────────────────────────────
#  BBOX + POSITIVE POINTS EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────
def get_bbox_and_points(binary_mask: np.ndarray, num_points: int = 8):
    ys, xs = np.nonzero(binary_mask)
    if len(ys) == 0:
        return None, None

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    bbox = [xmin, ymin, xmax, ymax]  # [x_min, y_min, x_max, y_max] ── xyxy format

    # Sample positive points (label=1 means foreground)
    n = min(num_points, len(ys))
    if n == 0:
        return bbox, []
    idx = np.random.choice(len(ys), n, replace=False)
    points = [[float(xs[i]), float(ys[i])] for i in idx]

    return bbox, points

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN GENERATION FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def create_and_save_pair(idx: int):
    data = generate_data(idx)

    # ── IMAGE PASS ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 6), dpi=180)
    ax = fig.add_subplot(111)
    plot_chart(ax, data, is_mask=False)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    rgb_img = buf.reshape((h, w, 4))[:, :, :3].copy()

    # ── MASK PASS (semantic RGB mask) ───────────────────────────────────────
    fig_mask = plt.figure(figsize=(9, 6), dpi=180)
    ax_mask = fig_mask.add_subplot(111)
    plot_chart(ax_mask, data, is_mask=True)
    recolor_legend_and_lines(ax_mask)
    canvas_mask = FigureCanvasAgg(fig_mask)
    canvas_mask.draw()
    buf_mask = np.frombuffer(canvas_mask.buffer_rgba(), dtype=np.uint8)
    rgb_mask = buf_mask.reshape((h, w, 4))[:, :, :3].copy()

    # Build integer label mask
    label_mask = np.zeros((h, w), dtype=np.uint8)
    for class_id, info in CLASS_INFO.items():
        label_mask[np.all(rgb_mask == info["color"], axis=-1)] = class_id

    # ── PER-CLASS BINARY MASKS + ANNOTATIONS ────────────────────────────────
    annotations = []
    os.makedirs("dataset/masks", exist_ok=True)
    os.makedirs("dataset/annotations", exist_ok=True)

    for class_id, info in CLASS_INFO.items():
        bin_mask = (label_mask == class_id).astype(np.uint8) * 255
        if not np.any(bin_mask):
            continue  # skip absent classes (e.g. no error bars)

        bbox, points = get_bbox_and_points(bin_mask, num_points=8)

        mask_filename = f"{idx:06d}_{info['name']}.png"
        mask_path = f"dataset/masks/{mask_filename}"
        mpimg.imsave(mask_path, bin_mask, cmap="gray")

        annotations.append({
            "id": len(annotations),
            "class_name": info["name"],
            "mask_file": f"masks/{mask_filename}",
            "bbox": bbox,
            "point_coords": points,
            "point_labels": [1] * len(points)   # 1 = positive point
        })

    # Save RGB image
    os.makedirs("dataset/images", exist_ok=True)
    mpimg.imsave(f"dataset/images/{idx:06d}.png", rgb_img)

    # Save JSON annotation
    json_data = {
        "image_id": idx,
        "file_name": f"images/{idx:06d}.png",
        "height": h,
        "width": w,
        "annotations": annotations
    }
    with open(f"dataset/annotations/{idx:06d}.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    plt.close(fig)
    plt.close(fig_mask)

# ──────────────────────────────────────────────────────────────────────────────
#  EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from jittered_bar_script import create_full_dataset

    create_full_dataset(num_charts=10, output_dir="dataset")
