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
from matplotlib.patches import Patch
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
    1: {"name": "bar", "category_name": "bars", "text_input": "bars"},
    2: {"name": "dot", "category_name": "dots", "text_input": "dots"},
    3: {"name": "line", "category_name": "lines", "text_input": "lines"},
    4: {"name": "legend", "category_name": "legend", "text_input": "legend"},
}

FIGSIZE = (9, 6)
DPI = 180
MASK_THRESHOLD = 10

QUALITATIVE_CMAPS = ["tab10", "tab20", "Set2", "Set3", "Paired", "Accent"]
MONO_BASE_COLORS = [
    "#3E5C76",
    "#5C6F68",
    "#6D597A",
    "#A05A2C",
    "#2F6690",
    "#4F772D",
]
GRID_STYLES = ["--", ":", "-."]
OUTLIER_DENSITY_PROFILES = {
    "none": {
        "activation_prob_range": (0.0, 0.0),
        "count_choices": [0],
        "count_weights": [1.0],
        "burst_slots_range": (0, 0),
        "burst_bonus_range": (0, 0),
        "distance_range": (2.0, 3.0),
        "spread_scale_range": (0.15, 0.2),
    },
    "few": {
        "activation_prob_range": (0.58, 0.74),
        "count_choices": [2, 3, 4, 5],
        "count_weights": [0.18, 0.3, 0.3, 0.22],
        "burst_slots_range": (1, 2),
        "burst_bonus_range": (2, 4),
        "distance_range": (2.0, 3.1),
        "spread_scale_range": (0.14, 0.2),
    },
    "medium": {
        "activation_prob_range": (0.76, 0.9),
        "count_choices": [3, 4, 5, 6, 7, 8],
        "count_weights": [0.12, 0.18, 0.22, 0.2, 0.16, 0.12],
        "burst_slots_range": (2, 4),
        "burst_bonus_range": (3, 5),
        "distance_range": (2.0, 3.3),
        "spread_scale_range": (0.13, 0.19),
    },
    "many": {
        "activation_prob_range": (0.9, 1.0),
        "count_choices": [5, 6, 7, 8, 9, 10, 11],
        "count_weights": [0.1, 0.14, 0.18, 0.2, 0.16, 0.12, 0.1],
        "burst_slots_range": (3, 5),
        "burst_bonus_range": (4, 7),
        "distance_range": (1.9, 3.1),
        "spread_scale_range": (0.12, 0.18),
    },
    "dense": {
        "activation_prob_range": (0.98, 1.0),
        "count_choices": [8, 9, 10, 11, 12, 14, 16],
        "count_weights": [0.1, 0.12, 0.16, 0.18, 0.18, 0.14, 0.12],
        "burst_slots_range": (4, 7),
        "burst_bonus_range": (5, 10),
        "distance_range": (1.8, 3.0),
        "spread_scale_range": (0.11, 0.17),
    },
}


def month_labels(n: int):
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"][:n]


def quarter_labels(n: int):
    return ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"][:n]


def cohort_labels(n: int):
    return [f"Cohort {idx + 1}" for idx in range(n)]


def site_labels(n: int):
    return [f"Site {chr(65 + idx)}" for idx in range(n)]


def segment_labels(n: int):
    return [f"Segment {idx + 1}" for idx in range(n)]


def line_labels(n: int):
    return [f"Line {idx + 1}" for idx in range(n)]


def department_labels(n: int):
    labels = [
        "Support",
        "Sales",
        "Ops",
        "Finance",
        "HR",
        "R&D",
        "QA",
        "Field",
    ]
    return labels[:n]


CONTEXTS = [
    {
        "titles": [
            "Experiment Summary",
            "Benchmark Comparison",
            "Model Evaluation Overview",
        ],
        "group_axis_label": "Cohorts",
        "value_label": "Average score",
        "group_label_fn": cohort_labels,
        "category_labels": ["Baseline", "Method A", "Method B", "Method C", "Method D"],
        "value_range": (4.0, 24.0),
        "error_range": (0.4, 2.6),
    },
    {
        "titles": [
            "Quarterly Sales Comparison",
            "Regional Revenue Snapshot",
            "Sales Performance by Quarter",
        ],
        "group_axis_label": "Quarter",
        "value_label": "Revenue (k$)",
        "group_label_fn": quarter_labels,
        "category_labels": ["North", "South", "East", "West", "Central"],
        "value_range": (35.0, 160.0),
        "error_range": (4.0, 18.0),
    },
    {
        "titles": [
            "Processing Time by Cluster",
            "Latency Comparison",
            "Runtime by Deployment Group",
        ],
        "group_axis_label": "Cluster",
        "value_label": "Latency (ms)",
        "group_label_fn": site_labels,
        "category_labels": ["Pipeline A", "Pipeline B", "Pipeline C", "Pipeline D", "Pipeline E"],
        "value_range": (85.0, 420.0),
        "error_range": (7.0, 30.0),
    },
    {
        "titles": [
            "Customer Rating Summary",
            "Average Satisfaction Scores",
            "Service Rating Comparison",
        ],
        "group_axis_label": "Department",
        "value_label": "Rating",
        "group_label_fn": department_labels,
        "category_labels": ["Current", "Pilot", "Updated", "Premium", "Legacy"],
        "value_range": (2.5, 9.5),
        "error_range": (0.2, 1.1),
    },
    {
        "titles": [
            "Yield by Production Line",
            "Manufacturing Yield Comparison",
            "Output Quality Overview",
        ],
        "group_axis_label": "Production line",
        "value_label": "Yield (%)",
        "group_label_fn": line_labels,
        "category_labels": ["Shift A", "Shift B", "Shift C", "Shift D", "Shift E"],
        "value_range": (55.0, 97.0),
        "error_range": (1.0, 5.5),
    },
    {
        "titles": [
            "Conversion Rate by Segment",
            "Campaign Response Overview",
            "Segment Conversion Comparison",
        ],
        "group_axis_label": "Segment",
        "value_label": "Conversion rate (%)",
        "group_label_fn": segment_labels,
        "category_labels": ["Email", "Paid", "Organic", "Referral", "Partner"],
        "value_range": (8.0, 42.0),
        "error_range": (0.8, 4.5),
    },
    {
        "titles": [
            "Monthly Throughput Review",
            "Production Throughput Snapshot",
            "Output per Month",
        ],
        "group_axis_label": "Month",
        "value_label": "Units processed",
        "group_label_fn": month_labels,
        "category_labels": ["Line A", "Line B", "Line C", "Line D", "Line E"],
        "value_range": (120.0, 420.0),
        "error_range": (10.0, 35.0),
    },
]

SCIENTIFIC_METRICS = [
    {
        "value_label": "% of Live cells",
        "titles": ["Lymphocytes", "T Cells", "B Cells", "Neutrophils", "MHCII+ Cells"],
        "value_range": (0.15, 9.8),
    },
    {
        "value_label": "THS+ area fraction (%)",
        "titles": ["THS+ area fraction", "Plaque burden", "Signal fraction"],
        "value_range": (0.05, 0.32),
    },
    {
        "value_label": "6E10+ area fraction (%)",
        "titles": ["6E10+ area fraction", "Amyloid area fraction", "Cortical burden"],
        "value_range": (0.08, 1.25),
    },
    {
        "value_label": "6E10+ density (No./mm²)",
        "titles": ["6E10+ density", "Plaque density", "Deposit density"],
        "value_range": (55.0, 520.0),
    },
    {
        "value_label": "Aβ (% of control)",
        "titles": ["Aβ", "Aβ42", "Aβ load"],
        "value_range": (35.0, 180.0),
    },
    {
        "value_label": "Normalized intensity",
        "titles": ["GFAP", "IBA1", "Synaptophysin", "pTau"],
        "value_range": (0.35, 3.2),
    },
]

SCIENTIFIC_PAIR_LABELS = [
    ["Lab", "Rewild"],
    ["Control", "Treatment"],
    ["Vehicle", "Drug"],
    ["WT", "KO"],
    ["eGFP", "hPGC1α"],
    ["Sham", "Injury"],
    ["Male", "Female"],
    ["Sed", "Run"],
]

SCIENTIFIC_GROUP_LABELS = [
    ["Neocortex", "Hippocampus"],
    ["Cortex", "Hippocampus"],
    ["Day 7", "Day 28"],
    ["Male", "Female"],
    ["CA1", "DG"],
    ["APP23", "WT"],
]

SCIENTIFIC_CATEGORY_LABELS = [
    ["Ctrl", "Tx"],
    ["Vehicle", "Drug"],
    ["WT", "APP"],
    ["Sed", "Run"],
    ["Saline", "Compound"],
]

SCIENTIFIC_PAIR_PALETTES = [
    ["#ff8c82", "#c26bff"],
    ["#66d100", "#2b99f0"],
    ["#2a66e8", "#17387f"],
    ["#ff6b6b", "#9c27ff"],
    ["#6ecf5d", "#3096ea"],
]

SCIENTIFIC_OBJECT_TEST_METRICS = [
    {
        "titles": ["OLT testing", "Novel object recognition", "Object exploration"],
        "value_label": "% Exploration",
        "value_range": (0.0, 100.0),
    },
    {
        "titles": ["Recognition memory", "Displaced object exploration", "Object memory task"],
        "value_label": "% Exploration",
        "value_range": (0.0, 100.0),
    },
]

SCIENTIFIC_OBJECT_TEST_GROUPS = [
    ["WT", "WT\neGFP", "WT\nhPGC1α", "APP23\neGFP", "APP23\nhPGC1α"],
    ["WT", "WT\nVector", "WT\nhPGC1α", "APP23\nVector", "APP23\nhPGC1α"],
    ["WT", "WT\nVehicle", "WT\nDrug", "APP23\nVehicle", "APP23\nDrug"],
]

SCIENTIFIC_OBJECT_TEST_CATEGORY_LABELS = [
    ["Obj A", "Obj B\n(displaced)"],
    ["Object A", "Object B\n(displaced)"],
    ["Familiar", "Novel\n(displaced)"],
]

SCIENTIFIC_SEX_METRICS = [
    {
        "titles": ["Hippocampal volume", "Hippocampal size", "Dentate gyrus volume"],
        "value_label": "Hippocampal volume (mm³)",
        "value_range": (0.0, 15.0),
    },
    {
        "titles": ["Cortical thickness", "Cortical volume", "Parenchymal volume"],
        "value_label": "Normalized volume",
        "value_range": (0.0, 12.0),
    },
    {
        "titles": ["Area fraction", "Signal coverage", "Marker-positive fraction"],
        "value_label": "Area fraction (%)",
        "value_range": (0.0, 14.0),
    },
]

SCIENTIFIC_SEX_CATEGORY_LABELS = [
    ["Control", "Treatment"],
    ["Vehicle", "Drug"],
    ["WT", "APP23"],
]

SCIENTIFIC_TOXICANT_METRICS = [
    {
        "titles": ["hAbeta40", "hAbeta42", "Aβ signal"],
        "value_label": "Intensity (a.u.)",
        "value_range": (0.0, 8000.0),
    },
    {
        "titles": ["Amyloid burden", "Plaque-associated signal", "Aβ concentration"],
        "value_label": "Signal (a.u.)",
        "value_range": (0.0, 7000.0),
    },
]

SCIENTIFIC_TOXICANT_GROUP_LABELS = [
    ["B6", "hAhA"],
    ["WT", "Tg"],
    ["C57BL/6J", "Humanized"],
]

SCIENTIFIC_TOXICANT_CATEGORY_LABELS = [
    ["Untreated", "Arsenic", "Cadmium", "Lead", "Mix"],
    ["Control", "Arsenic", "Cadmium", "Lead", "Mixture"],
]

SCIENTIFIC_VARIANT_CYCLE = [
    "paired",
    "grouped_pair",
    "age_block",
    "sex_pair",
    "toxicant_multi",
]

PLOT_BLUEPRINT_CYCLE = [
    ("scientific", "paired"),
    ("scientific", "grouped_pair"),
    ("scientific", "age_block"),
    ("scientific", "sex_pair"),
    ("scientific", "toxicant_multi"),
    ("classic", None),
    ("scientific", "paired"),
    ("classic", None),
]


def darken_color(color, amount: float = 0.35):
    rgb = np.array(plt.matplotlib.colors.to_rgb(color), dtype=float)
    return tuple(np.clip(rgb * (1.0 - amount), 0.0, 1.0))


def lighten_color(color, amount: float = 0.45):
    rgb = np.array(plt.matplotlib.colors.to_rgb(color), dtype=float)
    return tuple(np.clip(rgb + (1.0 - rgb) * amount, 0.0, 1.0))


def choose_palette(num_categories: int, color_mode: str):
    if color_mode == "mono":
        base = random.choice(MONO_BASE_COLORS)
        base_rgb = plt.matplotlib.colors.to_rgb(base)
        colors = [base_rgb for _ in range(num_categories)]
    else:
        cmap = plt.get_cmap(random.choice(QUALITATIVE_CMAPS))
        colors = [cmap(idx / max(1, num_categories - 1)) for idx in range(num_categories)]
    return colors


def choose_legend_config(num_categories: int):
    family = random.choices(
        population=["top_outside", "bottom_outside", "upper_right", "upper_left"],
        weights=[0.45, 0.33, 0.14, 0.08],
        k=1,
    )[0]

    ncol = min(num_categories, 4) if "outside" in family else 1
    if family == "top_outside":
        return {
            "family": family,
            "loc": "lower center",
            "bbox_to_anchor": (0.5, 1.07),
            "ncol": ncol,
        }
    if family == "bottom_outside":
        return {
            "family": family,
            "loc": "upper center",
            "bbox_to_anchor": (0.5, -0.14),
            "ncol": ncol,
        }
    if family == "upper_left":
        return {"family": family, "loc": "upper left", "bbox_to_anchor": None, "ncol": 1}
    return {"family": family, "loc": "upper right", "bbox_to_anchor": None, "ncol": 1}


def choose_outlier_density():
    return random.choices(
        ["few", "medium", "many", "dense"],
        weights=[0.08, 0.24, 0.36, 0.32],
        k=1,
    )[0]


def choose_outlier_pattern():
    return random.choices(
        ["balanced", "group_hotspots", "category_hotspots", "mixed_hotspots"],
        weights=[0.34, 0.26, 0.18, 0.22],
        k=1,
    )[0]


def choose_outlier_layout():
    return random.choices(
        ["isolated", "mixed", "clustered"],
        weights=[0.2, 0.52, 0.28],
        k=1,
    )[0]


def choose_outlier_variance():
    return random.choices(
        ["tight", "mixed", "loose"],
        weights=[0.22, 0.56, 0.22],
        k=1,
    )[0]


def build_slot_scale_map(num_groups: int, num_categories: int, outlier_pattern: str):
    group_scale = np.random.lognormal(mean=0.0, sigma=0.3, size=num_groups)
    category_scale = np.random.lognormal(mean=0.0, sigma=0.24, size=num_categories)

    if outlier_pattern in {"group_hotspots", "mixed_hotspots"}:
        hot_group_count = random.randint(1, min(3, num_groups))
        hot_groups = random.sample(range(num_groups), k=hot_group_count)
        for group_idx in hot_groups:
            group_scale[group_idx] *= random.uniform(1.35, 1.9)

        remaining_groups = [idx for idx in range(num_groups) if idx not in hot_groups]
        if remaining_groups and random.random() < 0.68:
            cool_group_count = random.randint(1, min(2, len(remaining_groups)))
            for group_idx in random.sample(remaining_groups, k=cool_group_count):
                group_scale[group_idx] *= random.uniform(0.58, 0.82)

    if outlier_pattern in {"category_hotspots", "mixed_hotspots"}:
        hot_category_count = random.randint(1, min(2, num_categories))
        hot_categories = random.sample(range(num_categories), k=hot_category_count)
        for category_idx in hot_categories:
            category_scale[category_idx] *= random.uniform(1.25, 1.8)

        remaining_categories = [idx for idx in range(num_categories) if idx not in hot_categories]
        if remaining_categories and random.random() < 0.52:
            cool_category_count = random.randint(1, min(1, len(remaining_categories)))
            for category_idx in random.sample(remaining_categories, k=cool_category_count):
                category_scale[category_idx] *= random.uniform(0.65, 0.88)

    slot_scale = np.outer(group_scale, category_scale)
    slot_scale *= np.random.lognormal(mean=0.0, sigma=0.18, size=(num_groups, num_categories))
    slot_scale /= np.median(slot_scale)
    return np.clip(slot_scale, 0.45, 2.65)


def sample_cluster_count(outlier_count: int, outlier_layout: str):
    if outlier_count <= 2:
        return 1

    if outlier_layout == "isolated":
        cluster_count = random.choices([1, 2], weights=[0.76, 0.24], k=1)[0]
    elif outlier_layout == "clustered":
        cluster_count = random.choices([1, 2, 3], weights=[0.14, 0.5, 0.36], k=1)[0]
    else:
        cluster_count = random.choices([1, 2, 3], weights=[0.42, 0.36, 0.22], k=1)[0]

    return min(outlier_count, cluster_count)


def choose_plot_family():
    return random.choices(["scientific", "classic"], weights=[0.68, 0.32], k=1)[0]


def choose_plot_blueprint(seed: int):
    return PLOT_BLUEPRINT_CYCLE[seed % len(PLOT_BLUEPRINT_CYCLE)]


def choose_scientific_variant(seed: int):
    return SCIENTIFIC_VARIANT_CYCLE[seed % len(SCIENTIFIC_VARIANT_CYCLE)]


def generate_scientific_samples(target_mean: float, sample_count: int, value_span: float):
    dispersion = max(value_span * random.uniform(0.025, 0.08), target_mean * random.uniform(0.08, 0.22))
    distribution = random.choices(
        ["normal", "upper_tail", "mild_skew"],
        weights=[0.42, 0.34, 0.24],
        k=1,
    )[0]

    if distribution == "normal":
        values = np.random.normal(target_mean, dispersion, sample_count)
    elif distribution == "upper_tail":
        values = np.random.normal(target_mean * random.uniform(0.93, 0.99), dispersion * 0.72, sample_count)
        tail_mask = np.random.rand(sample_count) < random.uniform(0.18, 0.34)
        if tail_mask.any():
            values[tail_mask] += np.random.gamma(2.1, dispersion * 0.58, tail_mask.sum())
    else:
        values = np.random.normal(target_mean, dispersion * 0.78, sample_count)
        skew_mask = np.random.rand(sample_count) < random.uniform(0.14, 0.28)
        if skew_mask.any():
            values[skew_mask] += np.random.normal(dispersion * 0.6, dispersion * 0.22, skew_mask.sum())

    if sample_count >= 6 and random.random() < 0.24:
        extreme_idx = random.randrange(sample_count)
        values[extreme_idx] += dispersion * random.uniform(1.0, 2.2)

    return np.clip(values, 0.0, None)


def summarize_slot_samples(slot_samples, num_groups: int, num_categories: int, value_span: float):
    means = np.zeros((num_groups, num_categories), dtype=float)
    stds = np.zeros((num_groups, num_categories), dtype=float)

    slot_idx = 0
    for group_idx in range(num_groups):
        for category_idx in range(num_categories):
            samples = slot_samples[slot_idx]
            means[group_idx, category_idx] = float(samples.mean())
            stds[group_idx, category_idx] = (
                float(samples.std(ddof=1) / np.sqrt(len(samples)))
                if len(samples) > 1
                else value_span * 0.03
            )
            slot_idx += 1

    return means, stds


def build_scientific_dot_positions(slot_samples, group_positions, bar_offsets, bar_width, jitter_scale=(0.18, 0.28)):
    jitter_amount = min(0.19, bar_width * random.uniform(*jitter_scale))
    dot_xs = []
    dot_ys = []
    slot_idx = 0

    for group_idx in range(len(group_positions)):
        for category_idx in range(len(bar_offsets)):
            position = group_positions[group_idx] + bar_offsets[category_idx]
            samples = slot_samples[slot_idx]
            dot_xs.append(position + np.random.uniform(-jitter_amount, jitter_amount, len(samples)))
            dot_ys.append(samples)
            slot_idx += 1

    return dot_xs, dot_ys


def build_group_pair_targets(base_levels, deltas, value_min, value_max):
    targets = np.stack([base_levels, base_levels + deltas], axis=1)
    return np.clip(targets, value_min * 0.6, value_max * 0.98)


def build_slot_samples_from_targets(targets, sample_counts, value_span: float):
    slot_samples = []
    num_groups, num_categories = targets.shape
    for group_idx in range(num_groups):
        for category_idx in range(num_categories):
            samples = generate_scientific_samples(
                float(targets[group_idx, category_idx]),
                int(sample_counts[group_idx, category_idx]),
                value_span,
            )
            slot_samples.append(samples)
    return slot_samples


def choose_significance_label(effect_ratio: float):
    thresholds = [
        (0.6, "****"),
        (0.42, "***"),
        (0.26, "**"),
        (0.14, "*"),
    ]
    for threshold, label in thresholds:
        if effect_ratio >= threshold:
            return label
    return None


def no_legend_config():
    return {"family": "none", "loc": None, "bbox_to_anchor": None, "ncol": 0}


def build_scientific_slot_colors(num_groups: int, num_categories: int):
    palette = random.choice(SCIENTIFIC_PAIR_PALETTES)
    if num_categories == 1:
        colors = [plt.matplotlib.colors.to_rgb(color) for color in palette[:num_groups]]
        slot_facecolors = [[colors[group_idx]] for group_idx in range(num_groups)]
        slot_edgecolors = [[darken_color(colors[group_idx], 0.24)] for group_idx in range(num_groups)]
        category_colors = [colors[0]]
    else:
        category_colors = [plt.matplotlib.colors.to_rgb(color) for color in palette[:num_categories]]
        slot_facecolors = []
        slot_edgecolors = []
        for _ in range(num_groups):
            slot_facecolors.append([category_colors[category_idx] for category_idx in range(num_categories)])
            slot_edgecolors.append([darken_color(category_colors[category_idx], 0.22) for category_idx in range(num_categories)])
    return category_colors, slot_facecolors, slot_edgecolors


def build_scientific_dot_style(slot_facecolors, slot_edgecolors, num_groups: int, num_categories: int):
    dot_variant = random.choices(
        ["filled_color", "hollow_black", "hollow_color", "filled_black"],
        weights=[0.28, 0.34, 0.24, 0.14],
        k=1,
    )[0]

    dot_facecolors = []
    dot_edgecolors = []
    dot_linewidths = []

    for group_idx in range(num_groups):
        for category_idx in range(num_categories):
            slot_face = slot_facecolors[group_idx][category_idx]
            slot_edge = slot_edgecolors[group_idx][category_idx]
            if dot_variant == "filled_color":
                dot_facecolors.append(slot_face)
                dot_edgecolors.append(darken_color(slot_face, 0.42))
                dot_linewidths.append(0.9)
            elif dot_variant == "hollow_black":
                dot_facecolors.append("white")
                dot_edgecolors.append("#222222")
                dot_linewidths.append(1.2)
            elif dot_variant == "hollow_color":
                dot_facecolors.append("white")
                dot_edgecolors.append(slot_edge)
                dot_linewidths.append(1.15)
            else:
                dot_facecolors.append("#111111")
                dot_edgecolors.append("#111111")
                dot_linewidths.append(0.8)

    return dot_variant, dot_facecolors, dot_edgecolors, dot_linewidths


def build_scientific_brackets(
    means,
    stds,
    dot_values_by_slot,
    group_positions,
    bar_offsets,
    max_value,
    group_indices=None,
):
    if len(group_positions) < 1:
        return []

    value_span = max(max_value, 1.0)
    brackets = []

    if means.shape[1] == 1 and len(group_positions) == 2:
        slot_max = max(
            means[0, 0] + stds[0, 0],
            means[1, 0] + stds[1, 0],
            float(dot_values_by_slot[0].max()) if dot_values_by_slot[0].size > 0 else 0.0,
            float(dot_values_by_slot[1].max()) if dot_values_by_slot[1].size > 0 else 0.0,
        )
        baseline = max((means[0, 0] + means[1, 0]) / 2.0, value_span * 0.08, 1e-6)
        effect_ratio = abs(means[1, 0] - means[0, 0]) / baseline
        label = choose_significance_label(effect_ratio)
        if not label and abs(means[1, 0] - means[0, 0]) > value_span * 0.08 and random.random() < 0.7:
            label = "*"
        if label:
            brackets.append(
                {
                    "x1": group_positions[0] + bar_offsets[0],
                    "x2": group_positions[1] + bar_offsets[0],
                    "y": slot_max + value_span * random.uniform(0.08, 0.12),
                    "arm": value_span * random.uniform(0.035, 0.055),
                    "label": label,
                    "text_offset": value_span * 0.02,
                }
            )
        return brackets

    if means.shape[1] == 2:
        candidate_groups = list(group_indices) if group_indices is not None else list(range(len(group_positions)))
        for group_idx in candidate_groups:
            left_slot = group_idx * 2
            right_slot = left_slot + 1
            slot_max = max(
                means[group_idx, 0] + stds[group_idx, 0],
                means[group_idx, 1] + stds[group_idx, 1],
                float(dot_values_by_slot[left_slot].max()) if dot_values_by_slot[left_slot].size > 0 else 0.0,
                float(dot_values_by_slot[right_slot].max()) if dot_values_by_slot[right_slot].size > 0 else 0.0,
            )
            baseline = max((means[group_idx, 0] + means[group_idx, 1]) / 2.0, value_span * 0.08, 1e-6)
            effect_ratio = abs(means[group_idx, 1] - means[group_idx, 0]) / baseline
            label = choose_significance_label(effect_ratio)
            if not label and abs(means[group_idx, 1] - means[group_idx, 0]) > value_span * 0.07 and random.random() < 0.68:
                label = "*"
            if not label:
                continue
            brackets.append(
                {
                    "x1": group_positions[group_idx] + bar_offsets[0],
                    "x2": group_positions[group_idx] + bar_offsets[1],
                    "y": slot_max + value_span * random.uniform(0.08, 0.12),
                    "arm": value_span * random.uniform(0.03, 0.05),
                    "label": label,
                    "text_offset": value_span * 0.018,
                }
            )
    return brackets


def generate_scientific_data(seed: int, variant: str | None = None):
    random.seed(seed)
    np.random.seed(seed)

    scientific_variant = variant or choose_scientific_variant(seed)
    orientation = "vertical"
    has_errorbars = True
    color_mode = "scientific"
    group_axis_label = ""
    show_legend = False
    legend_config = no_legend_config()
    legend_title = None
    panel_label = None
    bottom_blocks = []
    title_enabled = True
    title_fontsize = random.uniform(13.0, 15.5)
    tick_labelsize = random.uniform(9.5, 11.5)
    label_fontsize = random.uniform(11.0, 13.0)
    spine_linewidth = random.uniform(1.55, 2.1)
    tick_length = random.uniform(5.0, 7.5)
    tick_width = random.uniform(1.2, 1.6)
    layout_left = random.uniform(0.15, 0.19)
    layout_bottom = random.uniform(0.17, 0.22)
    layout_top = random.uniform(0.84, 0.9)
    layout_right = 0.98
    marker_size = random.uniform(24.0, 44.0)
    marker_alpha = 0.95
    bar_alpha = random.uniform(0.72, 0.88)
    edge_linewidth = random.uniform(1.4, 1.9)
    x_tick_rotation = random.choice([0, 0, 0, 28, 40])
    explicit_max_value = None

    if scientific_variant == "paired":
        metric = random.choice(SCIENTIFIC_METRICS)
        title = random.choice(metric["titles"])
        value_min, value_max = metric["value_range"]
        value_span = value_max - value_min
        num_groups = 2
        num_categories = 1
        group_labels = random.choice(SCIENTIFIC_PAIR_LABELS)
        cat_labels = [""]
        colors, slot_facecolors, slot_edgecolors = build_scientific_slot_colors(num_groups, num_categories)
        dot_variant, dot_facecolors, dot_edgecolors, dot_linewidths = build_scientific_dot_style(
            slot_facecolors,
            slot_edgecolors,
            num_groups,
            num_categories,
        )

        sample_counts = np.random.randint(4, 11, size=(num_groups, num_categories))
        base_level = random.uniform(value_min + value_span * 0.12, value_max - value_span * 0.35)
        delta = value_span * random.uniform(0.08, 0.34)
        direction = random.choice([-1.0, 1.0])
        target_means = [base_level, np.clip(base_level + direction * delta, value_min * 0.6, value_max * 0.98)]
        if abs(target_means[1] - target_means[0]) < value_span * 0.08:
            target_means[1] = np.clip(
                target_means[0] + value_span * random.uniform(0.1, 0.22),
                value_min * 0.6,
                value_max * 0.98,
            )

        targets = np.array(target_means, dtype=float).reshape(num_groups, num_categories)
        slot_samples = build_slot_samples_from_targets(targets, sample_counts, value_span)
        means, stds = summarize_slot_samples(slot_samples, num_groups, num_categories, value_span)
        sig_brackets = build_scientific_brackets(
            means,
            stds,
            slot_samples,
            np.arange(num_groups, dtype=float),
            np.array([0.0], dtype=float),
            float(max(means.max(), value_max)),
        )
        show_legend = False

    elif scientific_variant == "grouped_pair":
        metric = random.choice(SCIENTIFIC_METRICS)
        title = random.choice(metric["titles"])
        value_min, value_max = metric["value_range"]
        value_span = value_max - value_min
        num_groups = 2
        num_categories = 2
        group_labels = random.choice(SCIENTIFIC_GROUP_LABELS)
        cat_labels = random.choice(SCIENTIFIC_CATEGORY_LABELS)
        colors, slot_facecolors, slot_edgecolors = build_scientific_slot_colors(num_groups, num_categories)
        dot_variant, dot_facecolors, dot_edgecolors, dot_linewidths = build_scientific_dot_style(
            slot_facecolors,
            slot_edgecolors,
            num_groups,
            num_categories,
        )

        sample_counts = np.random.randint(4, 11, size=(num_groups, num_categories))
        base_levels = np.random.uniform(value_min + value_span * 0.12, value_max - value_span * 0.45, size=num_groups)
        treatment_shift = value_span * random.uniform(0.06, 0.22)
        if random.random() < 0.25:
            treatment_shift *= -1.0

        targets = np.zeros((num_groups, num_categories), dtype=float)
        for group_idx in range(num_groups):
            group_shift = value_span * random.uniform(-0.08, 0.08)
            targets[group_idx, 0] = np.clip(base_levels[group_idx] + group_shift, value_min * 0.6, value_max * 0.95)
            targets[group_idx, 1] = np.clip(
                base_levels[group_idx] + group_shift + treatment_shift * random.uniform(0.75, 1.2),
                value_min * 0.6,
                value_max * 0.98,
            )

        slot_samples = build_slot_samples_from_targets(targets, sample_counts, value_span)
        means, stds = summarize_slot_samples(slot_samples, num_groups, num_categories, value_span)
        sig_brackets = build_scientific_brackets(
            means,
            stds,
            slot_samples,
            np.arange(num_groups, dtype=float),
            np.array([-0.185, 0.185], dtype=float),
            float(max(means.max(), value_max)),
        )
        show_legend = num_categories > 1 and random.random() < 0.1
        legend_config = choose_legend_config(num_categories) if show_legend else no_legend_config()

    elif scientific_variant == "age_block":
        metric = random.choice(SCIENTIFIC_OBJECT_TEST_METRICS)
        title = random.choice(metric["titles"])
        value_min, value_max = metric["value_range"]
        value_span = value_max - value_min
        num_groups = 5
        num_categories = 2
        group_labels = random.choice(SCIENTIFIC_OBJECT_TEST_GROUPS)
        cat_labels = random.choice(SCIENTIFIC_OBJECT_TEST_CATEGORY_LABELS)
        colors = [
            plt.matplotlib.colors.to_rgb("#ffffff"),
            plt.matplotlib.colors.to_rgb(random.choice(["#4051d4", "#3046d7", "#4a57d8", "#3048c8"])),
        ]
        slot_facecolors = [[colors[0], colors[1]] for _ in range(num_groups)]
        slot_edgecolors = [[(0.08, 0.08, 0.08), darken_color(colors[1], 0.22)] for _ in range(num_groups)]
        dot_variant = "filled_black"
        dot_facecolors = ["#111111" for _ in range(num_groups * num_categories)]
        dot_edgecolors = ["#111111" for _ in range(num_groups * num_categories)]
        dot_linewidths = [0.75 for _ in range(num_groups * num_categories)]
        sample_counts = np.random.randint(6, 12, size=(num_groups, num_categories))

        base_levels = np.array(
            [
                random.uniform(30.0, 44.0),
                random.uniform(39.0, 50.0),
                random.uniform(36.0, 48.0),
                random.uniform(40.0, 53.0),
                random.uniform(42.0, 55.0),
            ],
            dtype=float,
        )
        displaced_deltas = np.array(
            [
                random.uniform(16.0, 28.0),
                random.uniform(5.0, 12.0),
                random.uniform(10.0, 22.0),
                random.uniform(8.0, 18.0),
                random.uniform(3.0, 11.0),
            ],
            dtype=float,
        )
        if random.random() < 0.5:
            displaced_deltas[3] += random.uniform(3.0, 7.0)
        if random.random() < 0.35:
            displaced_deltas[4] *= random.uniform(0.55, 0.9)

        targets = build_group_pair_targets(base_levels, displaced_deltas, value_min, value_max)
        slot_samples = build_slot_samples_from_targets(targets, sample_counts, value_span)
        means, stds = summarize_slot_samples(slot_samples, num_groups, num_categories, value_span)
        sig_group_count = random.choice([1, 2, 2])
        candidate_groups = [0, 2, 3, 4]
        sig_groups = sorted(random.sample(candidate_groups, k=min(sig_group_count, len(candidate_groups))))
        sig_brackets = build_scientific_brackets(
            means,
            stds,
            slot_samples,
            np.arange(num_groups, dtype=float),
            np.array([-0.185, 0.185], dtype=float),
            float(max(means.max(), value_max)),
            group_indices=sig_groups,
        )
        show_legend = True
        legend_config = {
            "family": "right_outside",
            "loc": "upper left",
            "bbox_to_anchor": (1.01, 1.0),
            "ncol": 1,
        }
        panel_label = {
            "text": random.choice(["A", "B", "C"]),
            "x": -0.16,
            "y": 1.05,
            "fontsize": 26,
            "fontweight": "bold",
        }
        bottom_blocks = [
            {
                "start_idx": 0,
                "end_idx": 0,
                "label": random.choice(["12 months\nold", "12-month\ncohort"]),
                "line_y": -0.2,
                "text_y": -0.26,
            },
            {
                "start_idx": 1,
                "end_idx": 4,
                "label": random.choice(["15 months old", "15-month cohort"]),
                "line_y": -0.2,
                "text_y": -0.26,
            },
        ]
        title_fontsize = random.uniform(18.0, 20.5)
        tick_labelsize = random.uniform(10.5, 12.0)
        label_fontsize = random.uniform(12.0, 13.6)
        marker_size = random.uniform(28.0, 38.0)
        bar_alpha = random.uniform(0.93, 0.98)
        edge_linewidth = random.uniform(1.7, 2.05)
        layout_left = random.uniform(0.11, 0.14)
        layout_bottom = random.uniform(0.28, 0.32)
        layout_top = random.uniform(0.82, 0.87)
        layout_right = random.uniform(0.8, 0.85)
        x_tick_rotation = 0
        explicit_max_value = max(90.0, float(means.max() + value_span * 0.15))

    elif scientific_variant == "sex_pair":
        metric = random.choice(SCIENTIFIC_SEX_METRICS)
        title = random.choice(metric["titles"])
        value_min, value_max = metric["value_range"]
        value_span = value_max - value_min
        num_groups = 2
        num_categories = 2
        group_labels = ["Male", "Female"]
        cat_labels = random.choice(SCIENTIFIC_SEX_CATEGORY_LABELS)
        colors = [
            plt.matplotlib.colors.to_rgb(random.choice(["#9a9a9a", "#a6a6a6", "#8d8d8d"])),
            plt.matplotlib.colors.to_rgb(random.choice(["#f1b36b", "#e8a54b", "#edb76f"])),
        ]
        slot_facecolors = [[colors[0], colors[1]] for _ in range(num_groups)]
        slot_edgecolors = [[darken_color(colors[0], 0.24), darken_color(colors[1], 0.2)] for _ in range(num_groups)]
        dot_variant = "hollow_color"
        dot_facecolors = ["white" for _ in range(num_groups * num_categories)]
        dot_edgecolors = [slot_edgecolors[group_idx][category_idx] for group_idx in range(num_groups) for category_idx in range(num_categories)]
        dot_linewidths = [1.0 for _ in range(num_groups * num_categories)]
        sample_counts = np.random.randint(7, 14, size=(num_groups, num_categories))

        base_levels = np.random.uniform(value_max * 0.42, value_max * 0.6, size=num_groups)
        treatment_drop = np.random.uniform(value_span * 0.1, value_span * 0.24, size=num_groups)
        targets = build_group_pair_targets(base_levels, -treatment_drop, value_min, value_max)
        slot_samples = build_slot_samples_from_targets(targets, sample_counts, value_span)
        means, stds = summarize_slot_samples(slot_samples, num_groups, num_categories, value_span)
        sig_brackets = build_scientific_brackets(
            means,
            stds,
            slot_samples,
            np.arange(num_groups, dtype=float),
            np.array([-0.185, 0.185], dtype=float),
            float(max(means.max(), value_max)),
            group_indices=[0, 1],
        )
        show_legend = False
        title_enabled = random.random() < 0.45
        title_fontsize = random.uniform(14.0, 16.5)
        marker_size = random.uniform(22.0, 30.0)
        bar_alpha = random.uniform(0.9, 0.96)
        edge_linewidth = random.uniform(1.55, 1.9)
        x_tick_rotation = 0
        explicit_max_value = max(value_max * 0.9, float(means.max() + value_span * 0.18))

    else:
        metric = random.choice(SCIENTIFIC_TOXICANT_METRICS)
        title = random.choice(metric["titles"])
        value_min, value_max = metric["value_range"]
        value_span = value_max - value_min
        num_groups = 2
        num_categories = 5
        group_labels = random.choice(SCIENTIFIC_TOXICANT_GROUP_LABELS)
        cat_labels = random.choice(SCIENTIFIC_TOXICANT_CATEGORY_LABELS)
        colors = [
            plt.matplotlib.colors.to_rgb("#ffffff"),
            plt.matplotlib.colors.to_rgb("#ff1f1f"),
            plt.matplotlib.colors.to_rgb("#1d1df1"),
            plt.matplotlib.colors.to_rgb("#1d8d0a"),
            plt.matplotlib.colors.to_rgb("#6f6f6f"),
        ]
        slot_facecolors = [[colors[category_idx] for category_idx in range(num_categories)] for _ in range(num_groups)]
        slot_edgecolors = [
            [
                (0.08, 0.08, 0.08) if category_idx == 0 else darken_color(colors[category_idx], 0.18)
                for category_idx in range(num_categories)
            ]
            for _ in range(num_groups)
        ]
        dot_variant = "filled_black"
        dot_facecolors = ["#111111" for _ in range(num_groups * num_categories)]
        dot_edgecolors = ["#111111" for _ in range(num_groups * num_categories)]
        dot_linewidths = [0.7 for _ in range(num_groups * num_categories)]
        sample_counts = np.random.randint(4, 9, size=(num_groups, num_categories))

        low_signal = np.random.uniform(0.0, value_max * 0.004, size=num_categories)
        high_signal = np.random.uniform(value_max * 0.35, value_max * 0.62, size=num_categories)
        high_signal += np.linspace(-0.08, 0.08, num_categories) * value_max * 0.18
        if random.random() < 0.45:
            high_signal = np.sort(high_signal)
        else:
            np.random.shuffle(high_signal)

        targets = np.vstack([low_signal, high_signal])
        targets = np.clip(targets, value_min, value_max * 0.92)
        slot_samples = build_slot_samples_from_targets(targets, sample_counts, value_span)
        means, stds = summarize_slot_samples(slot_samples, num_groups, num_categories, value_span)
        sig_brackets = []
        show_legend = True
        legend_config = {
            "family": "left_outside",
            "loc": "upper left",
            "bbox_to_anchor": (-0.42, 1.0),
            "ncol": 1,
        }
        title_fontsize = random.uniform(16.5, 19.0)
        tick_labelsize = random.uniform(10.5, 12.0)
        label_fontsize = random.uniform(11.5, 13.0)
        marker_size = random.uniform(18.0, 28.0)
        bar_alpha = random.uniform(0.94, 0.99)
        edge_linewidth = random.uniform(1.55, 1.9)
        layout_left = random.uniform(0.3, 0.36)
        layout_bottom = random.uniform(0.18, 0.23)
        layout_top = random.uniform(0.83, 0.88)
        layout_right = 0.98
        x_tick_rotation = random.choice([35, 40, 45])
        explicit_max_value = float(max(random.choice([6000.0, 7000.0, 8000.0]), means.max() * 1.12))

    group_positions = np.arange(num_groups, dtype=float)
    bar_span = 0.74
    bar_width = bar_span / num_categories
    bar_offsets = np.linspace(
        -((num_categories - 1) / 2) * bar_width,
        ((num_categories - 1) / 2) * bar_width,
        num_categories,
    )

    dot_xs, dot_ys = build_scientific_dot_positions(
        slot_samples,
        group_positions,
        bar_offsets,
        bar_width,
        jitter_scale=(0.15, 0.24) if scientific_variant == "toxicant_multi" else (0.18, 0.28),
    )
    max_dot_value = max((float(values.max()) for values in dot_ys if values.size > 0), default=0.0)
    sig_top = max(
        (bracket["y"] + bracket["text_offset"] + value_span * 0.03 for bracket in sig_brackets),
        default=0.0,
    )
    computed_max_value = float(max(means.max(), max_dot_value, sig_top) + value_span * random.uniform(0.05, 0.1))
    max_value = explicit_max_value if explicit_max_value is not None else computed_max_value

    return {
        "seed": seed,
        "plot_family": "scientific",
        "scientific_variant": scientific_variant,
        "orientation": orientation,
        "color_mode": color_mode,
        "has_errorbars": has_errorbars,
        "num_groups": num_groups,
        "num_categories": num_categories,
        "group_positions": group_positions,
        "bar_offsets": bar_offsets,
        "bar_width": bar_width,
        "means": means,
        "stds": stds,
        "dot_xs": dot_xs,
        "dot_ys": dot_ys,
        "colors": colors,
        "edgecolors": [
            (0.08, 0.08, 0.08) if plt.matplotlib.colors.to_rgb(color) == plt.matplotlib.colors.to_rgb("#ffffff") else darken_color(color, 0.32)
            for color in colors
        ],
        "slot_facecolors": slot_facecolors,
        "slot_edgecolors": slot_edgecolors,
        "dot_facecolors": dot_facecolors,
        "dot_edgecolors": dot_edgecolors,
        "dot_linewidths": dot_linewidths,
        "title": title,
        "group_labels": group_labels,
        "cat_labels": cat_labels,
        "legend": legend_config,
        "legend_title": legend_title,
        "show_legend": show_legend,
        "value_label": metric["value_label"],
        "group_axis_label": group_axis_label,
        "max_value": max_value,
        "marker_size": marker_size,
        "marker_alpha": marker_alpha,
        "bar_alpha": bar_alpha,
        "edge_linewidth": edge_linewidth,
        "grid_enabled": False,
        "grid_style": "--",
        "hide_top_right": True,
        "title_enabled": title_enabled,
        "has_outliers": True,
        "outlier_density": "sample_points",
        "outlier_side_mode": "mixed_in_bar",
        "outlier_pattern": scientific_variant,
        "outlier_layout": dot_variant,
        "outlier_variance": random.choices(["tight", "mixed", "loose"], weights=[0.26, 0.48, 0.26], k=1)[0],
        "sig_brackets": sig_brackets,
        "x_tick_rotation": x_tick_rotation,
        "axis_theme": "scientific",
        "title_fontsize": title_fontsize,
        "tick_labelsize": tick_labelsize,
        "label_fontsize": label_fontsize,
        "spine_linewidth": spine_linewidth,
        "tick_length": tick_length,
        "tick_width": tick_width,
        "layout_left": layout_left,
        "layout_bottom": layout_bottom,
        "layout_top": layout_top,
        "layout_right": layout_right,
        "panel_label": panel_label,
        "bottom_blocks": bottom_blocks,
        "block_linewidth": 1.4 if scientific_variant == "age_block" else 1.2,
        "block_textsize": 11.0 if scientific_variant == "age_block" else 10.0,
    }


def generate_classic_data(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    orientation = random.choices(["vertical", "horizontal"], weights=[0.65, 0.35], k=1)[0]
    num_groups = random.randint(3, 8)
    num_categories = random.randint(2, 5)
    context = random.choice(CONTEXTS)
    color_mode = random.choices(["multi", "mono"], weights=[0.65, 0.35], k=1)[0]
    has_errorbars = random.random() < 0.78

    value_min, value_max = context["value_range"]
    error_min, error_max = context["error_range"]
    means = np.random.uniform(value_min, value_max, (num_groups, num_categories))
    stds = (
        np.random.uniform(error_min, error_max, (num_groups, num_categories))
        if has_errorbars
        else np.zeros((num_groups, num_categories), dtype=float)
    )

    group_positions = np.arange(num_groups, dtype=float)
    bar_span = 0.74
    bar_width = bar_span / num_categories
    bar_offsets = np.linspace(
        -((num_categories - 1) / 2) * bar_width,
        ((num_categories - 1) / 2) * bar_width,
        num_categories,
    )
    jitter_amount = min(0.25, bar_width * random.uniform(0.32, 0.55))
    colors = choose_palette(num_categories, color_mode)
    title = random.choice(context["titles"])
    group_labels = context["group_label_fn"](num_groups)
    cat_labels = context["category_labels"][:num_categories]
    legend = choose_legend_config(num_categories)
    outlier_density = choose_outlier_density()
    outlier_profile = OUTLIER_DENSITY_PROFILES[outlier_density]
    has_outliers = outlier_density != "none"
    base_activation_prob = random.uniform(*outlier_profile["activation_prob_range"])
    outlier_side_mode = "outside_bar"
    outlier_pattern = choose_outlier_pattern()
    outlier_layout = choose_outlier_layout()
    outlier_variance = choose_outlier_variance()
    slot_scale_map = build_slot_scale_map(num_groups, num_categories, outlier_pattern)

    variance_scale_ranges = {
        "tight": (0.06, 0.11),
        "mixed": (0.08, 0.16),
        "loose": (0.11, 0.24),
    }
    band_ranges = {
        "near": (0.9, 1.55),
        "mid": (1.45, 2.35),
        "far": (2.2, 3.45),
    }
    band_weights = {
        "few": [0.58, 0.29, 0.13],
        "medium": [0.52, 0.32, 0.16],
        "many": [0.46, 0.35, 0.19],
        "dense": [0.4, 0.37, 0.23],
    }
    slot_count_caps = {"few": 7, "medium": 11, "many": 16, "dense": 22}

    total_slots = num_groups * num_categories
    burst_count = random.randint(*outlier_profile["burst_slots_range"])
    burst_count = min(total_slots, burst_count)
    burst_slots = set(random.sample(range(total_slots), k=burst_count)) if burst_count > 0 else set()

    dot_xs = []
    dot_ys = []
    for category_idx in range(num_categories):
        xs_c, ys_c = [], []
        for group_idx in range(num_groups):
            position = group_positions[group_idx] + bar_offsets[category_idx]
            center_value = means[group_idx, category_idx]
            if not has_outliers:
                continue

            slot_idx = category_idx * num_groups + group_idx
            is_burst_slot = slot_idx in burst_slots
            slot_scale = float(slot_scale_map[group_idx, category_idx])
            activation_scale = np.interp(slot_scale, [0.45, 2.65], [0.62, 1.18])
            slot_activation_prob = float(np.clip(base_activation_prob * activation_scale, 0.18, 1.0))

            if not is_burst_slot and random.random() > slot_activation_prob:
                continue

            outlier_count = int(
                np.random.choice(
                    outlier_profile["count_choices"],
                    p=outlier_profile["count_weights"],
                )
            )
            outlier_count = max(1, int(round(outlier_count * np.interp(slot_scale, [0.45, 2.65], [0.7, 1.5]))))
            if is_burst_slot:
                outlier_count += random.randint(*outlier_profile["burst_bonus_range"])
            outlier_count = min(slot_count_caps[outlier_density], outlier_count)
            if outlier_count <= 0:
                continue

            spread = max(error_min * 0.9, stds[group_idx, category_idx] * 0.85, value_max * 0.015)
            bar_clearance = max(stds[group_idx, category_idx] * 0.8, error_min * 0.9, value_max * 0.02)
            outlier_jitter = min(bar_width * 0.45, max(jitter_amount * 1.4, bar_width * 0.18))
            cluster_count = sample_cluster_count(outlier_count, outlier_layout)

            if cluster_count == 1:
                cluster_sizes = np.array([outlier_count], dtype=int)
            else:
                cluster_weights = np.random.dirichlet(np.full(cluster_count, random.uniform(0.8, 1.35)))
                cluster_sizes = np.random.multinomial(outlier_count - cluster_count, cluster_weights) + 1

            slot_xs = []
            slot_ys = []
            band_choice_weights = band_weights[outlier_density]
            if is_burst_slot:
                band_choice_weights = [0.28, 0.42, 0.3]

            for cluster_size in cluster_sizes:
                band_name = random.choices(["near", "mid", "far"], weights=band_choice_weights, k=1)[0]
                distance_multiplier = random.uniform(*band_ranges[band_name])
                distance_multiplier = min(
                    distance_multiplier,
                    outlier_profile["distance_range"][1] + (0.25 if band_name == "far" else 0.0),
                )
                distance = max(outlier_profile["distance_range"][0], distance_multiplier) * spread

                cluster_sd = spread * random.uniform(*variance_scale_ranges[outlier_variance])
                if band_name == "far":
                    cluster_sd *= random.uniform(1.05, 1.28)
                elif band_name == "near":
                    cluster_sd *= random.uniform(0.86, 1.0)

                cluster_center = center_value + bar_clearance + distance
                outlier_values = np.random.normal(cluster_center, cluster_sd, cluster_size)
                if cluster_size >= 4 and random.random() < 0.22:
                    extreme_idx = random.randrange(cluster_size)
                    outlier_values[extreme_idx] += spread * random.uniform(0.75, 1.85)
                outlier_values = np.clip(outlier_values, 0.0, None)

                cluster_shift = random.uniform(-outlier_jitter * 0.55, outlier_jitter * 0.55)
                if outlier_layout == "isolated":
                    jitter_scale = random.uniform(0.18, 0.3)
                elif outlier_layout == "clustered":
                    jitter_scale = random.uniform(0.12, 0.24)
                else:
                    jitter_scale = random.uniform(0.16, 0.34)
                jitter = np.random.normal(cluster_shift, outlier_jitter * jitter_scale, cluster_size)
                jitter = np.clip(jitter, -outlier_jitter, outlier_jitter)

                if orientation == "vertical":
                    slot_xs.append(position + jitter)
                    slot_ys.append(outlier_values)
                else:
                    slot_xs.append(outlier_values)
                    slot_ys.append(position + jitter)

            xs_c.append(np.concatenate(slot_xs))
            ys_c.append(np.concatenate(slot_ys))

        if xs_c:
            dot_xs.append(np.concatenate(xs_c))
            dot_ys.append(np.concatenate(ys_c))
        else:
            dot_xs.append(np.array([], dtype=float))
            dot_ys.append(np.array([], dtype=float))

    max_dot_value = max((float(values.max()) for values in dot_xs if values.size > 0), default=0.0)
    if orientation == "vertical":
        max_dot_value = max((float(values.max()) for values in dot_ys if values.size > 0), default=0.0)
    max_value = float(max(means.max(), max_dot_value) + max(1.5 * error_max, 0.08 * value_max))
    edgecolors = [darken_color(color, 0.38) for color in colors]
    slot_facecolors = [[colors[category_idx] for category_idx in range(num_categories)] for _ in range(num_groups)]
    slot_edgecolors = [[edgecolors[category_idx] for category_idx in range(num_categories)] for _ in range(num_groups)]

    return {
        "seed": seed,
        "plot_family": "classic",
        "orientation": orientation,
        "color_mode": color_mode,
        "has_errorbars": has_errorbars,
        "num_groups": num_groups,
        "num_categories": num_categories,
        "group_positions": group_positions,
        "bar_offsets": bar_offsets,
        "bar_width": bar_width,
        "means": means,
        "stds": stds,
        "dot_xs": dot_xs,
        "dot_ys": dot_ys,
        "colors": colors,
        "edgecolors": edgecolors,
        "slot_facecolors": slot_facecolors,
        "slot_edgecolors": slot_edgecolors,
        "dot_facecolors": [colors[category_idx] for category_idx in range(num_categories)],
        "dot_edgecolors": [edgecolors[category_idx] for category_idx in range(num_categories)],
        "dot_linewidths": [1.0 for _ in range(num_categories)],
        "title": title,
        "group_labels": group_labels,
        "cat_labels": cat_labels,
        "legend": legend,
        "show_legend": True,
        "value_label": context["value_label"],
        "group_axis_label": context["group_axis_label"],
        "max_value": max_value,
        "marker_size": random.uniform(16.0, 22.0),
        "marker_alpha": random.uniform(0.82, 0.92),
        "bar_alpha": random.uniform(0.84, 0.95),
        "edge_linewidth": random.uniform(0.9, 1.3),
        "grid_enabled": random.random() < 0.75,
        "grid_style": random.choice(GRID_STYLES),
        "hide_top_right": random.random() < 0.72,
        "title_enabled": random.random() < 0.9,
        "has_outliers": has_outliers,
        "outlier_density": outlier_density,
        "outlier_side_mode": outlier_side_mode,
        "outlier_pattern": outlier_pattern,
        "outlier_layout": outlier_layout,
        "outlier_variance": outlier_variance,
        "sig_brackets": [],
        "x_tick_rotation": 0,
        "axis_theme": "classic",
        "title_fontsize": 14.0,
        "tick_labelsize": 10.0,
        "label_fontsize": 10.5,
        "spine_linewidth": 1.0,
        "tick_length": 4.0,
        "tick_width": 1.0,
    }


def generate_data(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    plot_family, scientific_variant = choose_plot_blueprint(seed)
    if plot_family == "scientific":
        return generate_scientific_data(seed, variant=scientific_variant)
    return generate_classic_data(seed)


def apply_layout(fig, data):
    left = data.get("layout_left", 0.12 if data["orientation"] == "vertical" else 0.18)
    right = data.get("layout_right", 0.98)
    top = data.get("layout_top", 0.90)
    bottom = data.get("layout_bottom", 0.14)

    if data.get("show_legend", True) and data["legend"]["family"] == "top_outside":
        top = 0.74
    elif data.get("show_legend", True) and data["legend"]["family"] == "bottom_outside":
        bottom = 0.24
    elif data.get("show_legend", True) and data["legend"]["family"] == "left_outside":
        left = max(left, 0.28)
    elif data.get("show_legend", True) and data["legend"]["family"] == "right_outside":
        right = min(right, 0.86)

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)


def configure_axes(ax, data, mask_mode: bool):
    text_color = "black"
    grid_color = "#d0d7de"
    axis_theme = data.get("axis_theme", "classic")
    label_fontsize = data.get("label_fontsize", 10.5)
    tick_labelsize = data.get("tick_labelsize", 10.0)
    title_fontsize = data.get("title_fontsize", 14.0)
    spine_linewidth = data.get("spine_linewidth", 1.0)

    if mask_mode:
        text_color = "black"
        grid_color = "black"

    ax.clear()

    if data["orientation"] == "vertical":
        ax.set_xticks(data["group_positions"])
        ax.set_xticklabels(data["group_labels"])
        if data["group_axis_label"]:
            ax.set_xlabel(data["group_axis_label"], color=text_color, fontsize=label_fontsize)
        ax.set_ylabel(data["value_label"], color=text_color, fontsize=label_fontsize)
        ax.set_ylim(0, data["max_value"])
        ax.set_xlim(data["group_positions"][0] - 0.8, data["group_positions"][-1] + 0.8)
        if data["grid_enabled"]:
            ax.yaxis.grid(
                True,
                linestyle=data["grid_style"],
                linewidth=0.7,
                alpha=0.65 if not mask_mode else 1.0,
                color=grid_color,
            )
    else:
        ax.set_yticks(data["group_positions"])
        ax.set_yticklabels(data["group_labels"])
        if data["group_axis_label"]:
            ax.set_ylabel(data["group_axis_label"], color=text_color, fontsize=label_fontsize)
        ax.set_xlabel(data["value_label"], color=text_color, fontsize=label_fontsize)
        ax.set_xlim(0, data["max_value"])
        ax.set_ylim(data["group_positions"][0] - 0.8, data["group_positions"][-1] + 0.8)
        if data["grid_enabled"]:
            ax.xaxis.grid(
                True,
                linestyle=data["grid_style"],
                linewidth=0.7,
                alpha=0.65 if not mask_mode else 1.0,
                color=grid_color,
            )

    if data["title_enabled"]:
        ax.set_title(
            data["title"],
            color=text_color,
            fontsize=title_fontsize,
            pad=10,
            fontweight="bold" if axis_theme == "scientific" else "normal",
        )

    if data["hide_top_right"] or axis_theme == "scientific":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for spine_name, spine in ax.spines.items():
        if mask_mode:
            spine.set_color("black")
            spine.set_linewidth(spine_linewidth)
        elif spine_name in {"left", "bottom"}:
            spine.set_color("#111111" if axis_theme == "scientific" else "#4a4a4a")
            spine.set_linewidth(spine_linewidth)

    ax.tick_params(
        colors=text_color,
        labelsize=tick_labelsize,
        width=data.get("tick_width", 1.0),
        length=data.get("tick_length", 4.0),
    )
    if data["orientation"] == "vertical" and data.get("x_tick_rotation", 0):
        for tick in ax.get_xticklabels():
            tick.set_rotation(data["x_tick_rotation"])
            tick.set_ha("right")
    ax.set_axisbelow(True)
    ax.set_facecolor("black" if mask_mode else "white")


def build_legend_handles(data, mask_mode: bool):
    handles = []
    for idx, label in enumerate(data["cat_labels"]):
        color = "white" if mask_mode else data["colors"][idx]
        edge = "white" if mask_mode else data["edgecolors"][idx]
        handles.append(
            Patch(
                facecolor=color,
                edgecolor=edge,
                linewidth=1.0,
                label=label,
                alpha=1.0 if mask_mode else data["bar_alpha"],
            )
        )
    return handles


def add_legend(ax, data, mask_mode: bool):
    if not data.get("show_legend", True):
        return None

    legend = data["legend"]
    handles = build_legend_handles(data, mask_mode=mask_mode)
    legend_title = data.get("legend_title")

    leg = ax.legend(
        handles=handles,
        labels=data["cat_labels"],
        title=legend_title,
        fontsize=data.get("legend_fontsize", 9),
        title_fontsize=data.get("legend_title_fontsize", 10),
        loc=legend["loc"],
        bbox_to_anchor=legend["bbox_to_anchor"],
        ncol=legend["ncol"],
        frameon=True,
        borderaxespad=0.6,
        handlelength=1.8,
        columnspacing=1.2,
    )

    frame = leg.get_frame()
    if mask_mode:
        frame.set_facecolor("white")
        frame.set_edgecolor("white")
        frame.set_alpha(1.0)
        if leg.get_title() is not None:
            leg.get_title().set_color("white")
        for text in leg.get_texts():
            text.set_color("white")
    else:
        frame.set_facecolor("white")
        frame.set_edgecolor("#c8ced6")
        frame.set_alpha(0.92)

    return leg


def draw_bars(ax, data, mask_mode: bool):
    for group_idx in range(data["num_groups"]):
        for category_idx in range(data["num_categories"]):
            position = data["group_positions"][group_idx] + data["bar_offsets"][category_idx]
            value = data["means"][group_idx, category_idx]
            color = "white" if mask_mode else data["slot_facecolors"][group_idx][category_idx]
            edge = "white" if mask_mode else data["slot_edgecolors"][group_idx][category_idx]

            if data["orientation"] == "vertical":
                ax.bar(
                    position,
                    value,
                    width=data["bar_width"],
                    color=color,
                    edgecolor=edge,
                    linewidth=data["edge_linewidth"],
                    alpha=1.0 if mask_mode else data["bar_alpha"],
                )
            else:
                ax.barh(
                    position,
                    value,
                    height=data["bar_width"],
                    color=color,
                    edgecolor=edge,
                    linewidth=data["edge_linewidth"],
                    alpha=1.0 if mask_mode else data["bar_alpha"],
                )


def draw_dots(ax, data, mask_mode: bool):
    for series_idx in range(len(data["dot_xs"])):
        face = "white" if mask_mode else data["dot_facecolors"][series_idx]
        edge = "white" if mask_mode else data["dot_edgecolors"][series_idx]
        linewidth = data["dot_linewidths"][series_idx] if not mask_mode else 1.1
        ax.scatter(
            data["dot_xs"][series_idx],
            data["dot_ys"][series_idx],
            s=(data["marker_size"] + 8) if mask_mode else data["marker_size"],
            facecolors=face,
            edgecolors=edge,
            linewidths=linewidth,
            marker="o",
            alpha=1.0 if mask_mode else data["marker_alpha"],
            zorder=3,
        )


def draw_errorbars(ax, data, mask_mode: bool):
    if not data["has_errorbars"]:
        return

    for group_idx in range(data["num_groups"]):
        for category_idx in range(data["num_categories"]):
            position = data["group_positions"][group_idx] + data["bar_offsets"][category_idx]
            value = data["means"][group_idx, category_idx]
            error = data["stds"][group_idx, category_idx]
            ecolor = "white" if mask_mode else data["slot_edgecolors"][group_idx][category_idx]
            kwargs = {
                "fmt": "none",
                "ecolor": ecolor,
                "elinewidth": 2.2 if mask_mode else (1.5 if data.get("axis_theme") == "scientific" else 1.8),
                "capsize": 4.5,
                "capthick": 1.5,
                "zorder": 4,
            }

            if data["orientation"] == "vertical":
                ax.errorbar(position, value, yerr=error, **kwargs)
            else:
                ax.errorbar(value, position, xerr=error, **kwargs)


def draw_significance_brackets(ax, data, mask_mode: bool):
    if data["orientation"] != "vertical":
        return

    for bracket in data.get("sig_brackets", []):
        color = "white" if mask_mode else "#111111"
        linewidth = 2.0 if mask_mode else 1.7
        x1 = bracket["x1"]
        x2 = bracket["x2"]
        y = bracket["y"]
        arm = bracket["arm"]
        ax.plot([x1, x1, x2, x2], [y - arm, y, y, y - arm], color=color, linewidth=linewidth, zorder=5)
        if not mask_mode:
            ax.text(
                (x1 + x2) / 2.0,
                y + bracket["text_offset"],
                bracket["label"],
                ha="center",
                va="bottom",
                fontsize=max(11, int(data.get("title_fontsize", 14) * 0.9)),
                fontweight="bold",
                color="#111111",
            )


def draw_bottom_blocks(ax, data, mask_mode: bool):
    if data["orientation"] != "vertical":
        return

    blocks = data.get("bottom_blocks", [])
    if not blocks:
        return

    min_offset = float(np.min(data["bar_offsets"])) if len(data["bar_offsets"]) else 0.0
    max_offset = float(np.max(data["bar_offsets"])) if len(data["bar_offsets"]) else 0.0
    line_width = data.get("block_linewidth", 1.2)
    text_size = data.get("block_textsize", 10.0)
    text_color = "white" if mask_mode else "#111111"
    line_color = "white" if mask_mode else "#111111"

    for block in blocks:
        start_idx = block["start_idx"]
        end_idx = block["end_idx"]
        line_y = block.get("line_y", -0.13)
        text_y = block.get("text_y", line_y - 0.05)
        left = data["group_positions"][start_idx] + min_offset - data["bar_width"] * 0.62
        right = data["group_positions"][end_idx] + max_offset + data["bar_width"] * 0.62
        ax.plot(
            [left, right],
            [line_y, line_y],
            transform=ax.get_xaxis_transform(),
            color=line_color,
            linewidth=2.0 if mask_mode else line_width,
            clip_on=False,
            zorder=6,
        )
        if not mask_mode:
            ax.text(
                (left + right) / 2.0,
                text_y,
                block["label"],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=text_size,
                color=text_color,
            )


def draw_panel_label(ax, data, mask_mode: bool):
    panel_label = data.get("panel_label")
    if mask_mode or not panel_label:
        return

    ax.text(
        panel_label.get("x", -0.14),
        panel_label.get("y", 1.04),
        panel_label["text"],
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=panel_label.get("fontsize", 24),
        fontweight=panel_label.get("fontweight", "bold"),
        color=panel_label.get("color", "#111111"),
    )


def plot_chart(ax, data, mode: str = "rgb", only_class: str | None = None):
    mask_mode = mode == "mask"
    configure_axes(ax, data, mask_mode=mask_mode)

    if only_class in (None, "bar"):
        draw_bars(ax, data, mask_mode=mask_mode)
    if only_class in (None, "dot"):
        draw_dots(ax, data, mask_mode=mask_mode)
    if only_class in (None, "line"):
        draw_errorbars(ax, data, mask_mode=mask_mode)
        draw_significance_brackets(ax, data, mask_mode=mask_mode)
        draw_bottom_blocks(ax, data, mask_mode=mask_mode)
    if only_class in (None, "legend"):
        add_legend(ax, data, mask_mode=mask_mode)
    if only_class is None:
        draw_panel_label(ax, data, mask_mode=mask_mode)


def render_rgb_image(data):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    plot_chart(ax, data, mode="rgb", only_class=None)
    apply_layout(fig, data)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    buffer = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    rgb = buffer.reshape((height, width, 4))[:, :, :3].copy()
    plt.close(fig)
    return rgb, width, height


def render_class_mask(data, class_name: str):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111)
    plot_chart(ax, data, mode="mask", only_class=class_name)
    apply_layout(fig, data)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    buffer = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    rgb = buffer.reshape((height, width, 4))[:, :, :3]
    binary_mask = (rgb.max(axis=-1) > MASK_THRESHOLD).astype(np.uint8)
    plt.close(fig)
    return binary_mask


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


def split_mask_instances(binary_mask: np.ndarray, class_name: str):
    if class_name == "legend":
        return [binary_mask.astype(np.uint8)] if binary_mask.sum() > 0 else []

    min_area = {
        "bar": 24,
        "dot": 10,
        "line": 8,
    }.get(class_name, 1)
    return split_connected_components(binary_mask, min_area=min_area)


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
    num_charts: int = 500,
    train_ratio: float = 0.8,
    output_dir: str = "chart_dataset",
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
        data = generate_data(idx)
        image_rgb, width, height = render_rgb_image(data)
        image_id = idx + 1
        split_name = "train" if idx < train_cutoff else "val"
        image_filename = f"{image_id:06d}.png"
        mpimg.imsave(output_root / "images" / split_name / image_filename, image_rgb)

        generation_stats[f"orientation:{data['orientation']}"] += 1
        generation_stats[f"family:{data['plot_family']}"] += 1
        if data["plot_family"] == "scientific":
            generation_stats[f"scientific_variant:{data.get('scientific_variant', 'base')}"] += 1
        generation_stats[f"color_mode:{data['color_mode']}"] += 1
        generation_stats[f"legend:{data['legend']['family']}"] += 1
        generation_stats[f"errorbars:{'yes' if data['has_errorbars'] else 'no'}"] += 1
        generation_stats[f"outliers:{data['outlier_density']}"] += 1
        generation_stats[f"outlier_pattern:{data['outlier_pattern']}"] += 1
        generation_stats[f"outlier_layout:{data['outlier_layout']}"] += 1

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
            binary_mask = render_class_mask(data, class_name)
            if preview_masks is not None:
                preview_masks[class_info["category_name"]] = binary_mask
            for instance_mask in split_mask_instances(binary_mask, class_name):
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
            save_preview_assets(preview_root, split_name, image_filename, image_rgb, preview_masks)

    categories = [
        {"id": int(class_id), "name": class_info["category_name"]}
        for class_id, class_info in CLASS_INFO.items()
    ]
    info = {
        "description": "Synthetic jittered bar chart dataset for SAM 3 fine-tuning",
        "version": "3.2",
        "year": 2026,
        "contributor": "Generated dataset",
        "date_created": "2026-04-07",
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
    parser = argparse.ArgumentParser(description="Generate the chart_dataset synthetic jittered bar chart dataset.")
    parser.add_argument("--num-charts", type=int, default=20, help="Number of chart images to generate.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of charts written into train.json. The remainder goes into val.json.",
    )
    parser.add_argument("--output-dir", type=str, default="chart_dataset", help="Dataset output directory.")
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
