"""Cloud pair manifest construction and filtering helpers."""

from datetime import date

import numpy as np
import pandas as pd


PAIR_MANIFEST_COLUMNS = [
    "pair_id",
    "pair_label",
    "group",
    "tags",
    "pairing_group",
    "pair_index",
    "date",
    "paired_overlap_pct",
    "test_etag",
    "test_filename",
    "test_s3_key",
    "ref_etag",
    "ref_filename",
    "ref_s3_key",
]
DEVICE_TAGS = {"pacer", "h10", "eq02"}

PAIR_SELECTION_TABLE_COLUMNS = [
    "Group",
    "Date",
    "Test File",
    "Ref File",
    "Overlap (%)",
]


def _coerce_manifest_date(value):
    """Coerce manifest dates to date objects for filtering."""
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def coerce_input_date(value):
    """Coerce Shiny date inputs to date objects."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def parse_tags(tags_value) -> list[str]:
    """Return normalized pipe-delimited catalogue tags."""
    if tags_value is None or pd.isna(tags_value):
        return []
    tags = []
    seen = set()
    for tag in str(tags_value).split("|"):
        normalized = tag.strip()
        if normalized and normalized not in seen:
            tags.append(normalized)
            seen.add(normalized)
    return tags


def _non_device_tags(*tag_values) -> list[str]:
    tags = []
    seen = set()
    for tag_value in tag_values:
        for tag in parse_tags(tag_value):
            if tag in DEVICE_TAGS or tag in seen:
                continue
            tags.append(tag)
            seen.add(tag)
    return tags


def _tag_filter_values(row) -> set[str]:
    return set(parse_tags(getattr(row, "tags", None))) - DEVICE_TAGS


def _group_column(pair_df: pd.DataFrame) -> pd.Series:
    return pair_df["group"]


def get_cloud_manifest_groups(pair_df: pd.DataFrame) -> list[str]:
    """Return available non-device tag filters from the manifest."""
    if not isinstance(pair_df, pd.DataFrame) or pair_df.empty:
        return []
    groups = set()
    for row in pair_df.itertuples(index=False):
        groups.update(_tag_filter_values(row))
    return sorted(group for group in groups if group)


def filter_cloud_pair_manifest(
    pair_df: pd.DataFrame,
    selected_groups=None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Filter manifest pairs by group and inclusive date range."""
    if not isinstance(pair_df, pd.DataFrame) or pair_df.empty:
        return pd.DataFrame(columns=PAIR_MANIFEST_COLUMNS)
    if not {"group", "tags"}.issubset(pair_df.columns):
        return pd.DataFrame(columns=PAIR_MANIFEST_COLUMNS)

    filtered_df = pair_df.copy()
    filtered_df["_pair_date"] = filtered_df["date"].apply(_coerce_manifest_date)

    if selected_groups is not None:
        groups = [str(group) for group in selected_groups if group]
        if not groups:
            return pd.DataFrame(columns=PAIR_MANIFEST_COLUMNS)
        selected_group_set = set(groups)
        filter_mask = filtered_df.apply(
            lambda row: bool(_tag_filter_values(row).intersection(selected_group_set)),
            axis=1,
        )
        filtered_df = filtered_df[filter_mask]

    start = coerce_input_date(start_date)
    if start is not None:
        filtered_df = filtered_df[
            filtered_df["_pair_date"].notna() & (filtered_df["_pair_date"] >= start)
        ]

    end = coerce_input_date(end_date)
    if end is not None:
        filtered_df = filtered_df[
            filtered_df["_pair_date"].notna() & (filtered_df["_pair_date"] <= end)
        ]

    return (
        filtered_df.drop(columns="_pair_date")
        .sort_values(
            ["group", "date", "pair_index", "test_filename", "ref_filename"]
        )
        .reset_index(drop=True)
    )


def get_cloud_manifest_date_bounds(pair_df: pd.DataFrame):
    """Return inclusive date bounds for the filtered manifest."""
    if not isinstance(pair_df, pd.DataFrame) or pair_df.empty:
        return None, None

    pair_dates = pair_df["date"].apply(_coerce_manifest_date).dropna()
    if pair_dates.empty:
        return None, None

    return pair_dates.min(), pair_dates.max()


def build_cloud_pair_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Build one manifest row per test/reference pair."""
    if not isinstance(manifest_df, pd.DataFrame) or manifest_df.empty:
        return pd.DataFrame(columns=PAIR_MANIFEST_COLUMNS)

    required_columns = {
        "etag",
        "paired_etag",
        "pairing_group",
        "pair_index",
        "date",
        "filename",
        "device_type",
        "tags",
        "s3_key",
    }
    if not required_columns.issubset(manifest_df.columns):
        return pd.DataFrame(columns=PAIR_MANIFEST_COLUMNS)

    manifest_df = manifest_df.copy()

    test_rows = manifest_df[
        (manifest_df["device_type"] == "test") & manifest_df["paired_etag"].notna()
    ].copy()
    if test_rows.empty:
        return pd.DataFrame(columns=PAIR_MANIFEST_COLUMNS)

    ref_rows = (
        manifest_df[manifest_df["device_type"] == "ref"][
            ["etag", "filename", "s3_key", "tags"]
        ]
        .rename(
            columns={
                "etag": "ref_etag",
                "filename": "ref_filename",
                "s3_key": "ref_s3_key",
                "tags": "ref_tags",
            }
        )
        .copy()
    )

    pair_df = test_rows.merge(
        ref_rows, left_on="paired_etag", right_on="ref_etag", how="inner"
    )
    if pair_df.empty:
        return pd.DataFrame(columns=PAIR_MANIFEST_COLUMNS)

    pair_df = pair_df.rename(
        columns={
            "etag": "test_etag",
            "filename": "test_filename",
            "s3_key": "test_s3_key",
        }
    )
    if "tags" not in pair_df.columns:
        pair_df["tags"] = None
    if "ref_tags" not in pair_df.columns:
        pair_df["ref_tags"] = None
    pair_df["_non_device_tags"] = pair_df.apply(
        lambda row: _non_device_tags(row["tags"], row["ref_tags"]),
        axis=1,
    )
    pair_df["tags"] = pair_df["_non_device_tags"].apply(
        lambda tags: "|".join(tags) if tags else None
    )
    pair_df["group"] = pair_df["tags"]
    pair_df["pair_id"] = pair_df.apply(
        lambda row: (
            f"{row['group']}:{int(row['pair_index']) if pd.notna(row['pair_index']) else 'na'}:"
            f"{row['test_etag']}:{row['ref_etag']}"
        ),
        axis=1,
    )
    pair_df["pair_label"] = pair_df.apply(
        lambda row: (
            f"{row['group']} | {row['date']} | "
            f"{row['test_filename']} <> {row['ref_filename']}"
        ),
        axis=1,
    )
    if "paired_overlap_pct" not in pair_df.columns:
        pair_df["paired_overlap_pct"] = np.nan
    pair_df = pair_df.loc[:, PAIR_MANIFEST_COLUMNS].sort_values(
        ["group", "date", "pair_index", "test_filename", "ref_filename"]
    )
    pair_df = pair_df.reset_index(drop=True)
    return pair_df


def build_cloud_pair_selection_table(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Build a sortable display table for cloud pair selection."""
    if not isinstance(pair_df, pd.DataFrame) or pair_df.empty:
        return pd.DataFrame(columns=PAIR_SELECTION_TABLE_COLUMNS)

    overlap_pct = (
        (pair_df["paired_overlap_pct"] * 100).round(2)
        if "paired_overlap_pct" in pair_df.columns
        else np.nan
    )
    selection_df = pd.DataFrame(
        {
            "Group": _group_column(pair_df),
            "Date": pair_df["date"],
            "Test File": pair_df["test_filename"],
            "Ref File": pair_df["ref_filename"],
            "Overlap (%)": overlap_pct,
        }
    )
    return selection_df.reset_index(drop=True)


def get_selected_cloud_pair_ids(pair_df: pd.DataFrame, selected_rows) -> list[str]:
    """Map selected grid row indices back to pair ids."""
    if (
        not isinstance(pair_df, pd.DataFrame)
        or pair_df.empty
        or not selected_rows
        or "pair_id" not in pair_df.columns
    ):
        return []

    selected_indices = []
    for row_index in selected_rows:
        try:
            selected_indices.append(int(row_index))
        except (TypeError, ValueError):
            continue

    valid_indices = [
        row_index for row_index in selected_indices if 0 <= row_index < len(pair_df)
    ]
    if not valid_indices:
        return []

    return pair_df.iloc[valid_indices]["pair_id"].tolist()
