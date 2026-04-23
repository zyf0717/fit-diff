"""Shared FIT overlap helpers for local and S3 catalogue builders."""

from functools import lru_cache
from itertools import combinations


EMPTY_TIME_BOUNDS = {
    "start_epoch": None,
    "end_epoch": None,
    "duration_seconds": None,
}


def build_time_bounds(timestamps):
    """Build time-bound metadata from a collection of epoch timestamps."""
    if timestamps is None:
        return EMPTY_TIME_BOUNDS.copy()

    timestamp_set = {int(timestamp) for timestamp in timestamps}
    if not timestamp_set:
        return EMPTY_TIME_BOUNDS.copy()

    start_epoch = min(timestamp_set)
    end_epoch = max(timestamp_set)
    return {
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "duration_seconds": max(end_epoch - start_epoch, 0),
    }


def format_duration_seconds(total_seconds):
    """Format a duration in seconds as HH:MM:SS."""
    normalized_seconds = int(max(total_seconds or 0, 0))
    hours, remainder = divmod(normalized_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def calculate_interval_overlap_seconds(item_a, item_b):
    """Calculate overlap duration between two time intervals."""
    if (
        item_a.get("start_epoch") is None
        or item_a.get("end_epoch") is None
        or item_b.get("start_epoch") is None
        or item_b.get("end_epoch") is None
    ):
        return None

    overlap_seconds = min(item_a["end_epoch"], item_b["end_epoch"]) - max(
        item_a["start_epoch"], item_b["start_epoch"]
    )
    return max(overlap_seconds, 0)


def calculate_overlap_pct(overlap_seconds, duration_a, duration_b):
    """Normalize overlap duration by the longer file duration."""
    if overlap_seconds is None or duration_a is None or duration_b is None:
        return None

    denominator = max(duration_a, duration_b)
    if denominator == 0:
        return 1.0 if overlap_seconds == 0 else None
    return overlap_seconds / denominator


def build_pair_candidate(item_a, item_b, id_key="id"):
    """Build overlap metrics for a candidate pair."""
    overlap_seconds = calculate_interval_overlap_seconds(item_a, item_b)
    if overlap_seconds is None or overlap_seconds <= 0:
        return None

    timestamps_a = item_a.get("timestamps")
    timestamps_b = item_b.get("timestamps")
    overlap_datapoints = None
    overlap_duration_seconds = None
    if timestamps_a is not None and timestamps_b is not None:
        common_timestamps = set(timestamps_a).intersection(timestamps_b)
        overlap_datapoints = len(common_timestamps)
        if common_timestamps:
            overlap_duration_seconds = max(common_timestamps) - min(common_timestamps)
        else:
            overlap_duration_seconds = 0

    return {
        "item_a_id": item_a[id_key],
        "item_b_id": item_b[id_key],
        "overlap_seconds": overlap_seconds,
        "overlap_pct": calculate_overlap_pct(
            overlap_seconds,
            item_a.get("duration_seconds"),
            item_b.get("duration_seconds"),
        ),
        "overlap_datapoints": overlap_datapoints,
        "overlap_duration_seconds": overlap_duration_seconds,
        "overlap_start_epoch": max(item_a["start_epoch"], item_b["start_epoch"]),
        "overlap_end_epoch": min(item_a["end_epoch"], item_b["end_epoch"]),
    }


def build_group_pairings(items, group_key="group", id_key="id", role_key=None):
    """
    Build best-effort one-to-one pairings for grouped items based on interval overlap.

    If ``role_key`` is provided, candidates are only formed across different roles.
    """
    pairings = []

    groups = {}
    for item in items:
        groups.setdefault(item[group_key], []).append(item)

    for group_items in groups.values():
        candidate_map = {}
        adjacency = {item[id_key]: set() for item in group_items}

        for item_a, item_b in combinations(group_items, 2):
            if role_key and item_a.get(role_key) == item_b.get(role_key):
                continue

            candidate = build_pair_candidate(item_a, item_b, id_key=id_key)
            if candidate is None:
                continue

            pair_key = tuple(sorted((item_a[id_key], item_b[id_key])))
            candidate_map[pair_key] = candidate
            adjacency[item_a[id_key]].add(item_b[id_key])
            adjacency[item_b[id_key]].add(item_a[id_key])

        for component_ids in _build_components(adjacency):
            selected_pairs = _select_best_pairs_for_component(
                component_ids, candidate_map
            )
            for pair_ids in selected_pairs:
                pairings.append(candidate_map[pair_ids])

    return pairings


def _build_components(adjacency):
    components = []
    visited = set()

    for node_id, neighbors in adjacency.items():
        if node_id in visited or not neighbors:
            continue

        stack = [node_id]
        component = set()
        while stack:
            current_id = stack.pop()
            if current_id in visited:
                continue
            visited.add(current_id)
            component.add(current_id)
            stack.extend(adjacency[current_id] - visited)

        components.append(tuple(sorted(component)))

    return components


def _select_best_pairs_for_component(component_ids, candidate_map):
    component_ids = tuple(sorted(component_ids))

    def is_better(proposal, incumbent):
        if proposal[:3] != incumbent[:3]:
            return proposal[:3] > incumbent[:3]
        return proposal[3] < incumbent[3]

    @lru_cache(maxsize=None)
    def solve(remaining_ids):
        if not remaining_ids:
            return (0, 0.0, 0, tuple())

        first_id = remaining_ids[0]
        best = solve(remaining_ids[1:])

        for partner_id in remaining_ids[1:]:
            pair_key = tuple(sorted((first_id, partner_id)))
            candidate = candidate_map.get(pair_key)
            if candidate is None:
                continue

            next_ids = tuple(
                item_id
                for item_id in remaining_ids[1:]
                if item_id != partner_id
            )
            tail = solve(next_ids)
            overlap_pct = candidate.get("overlap_pct") or 0.0
            proposal = (
                tail[0] + 1,
                tail[1] + overlap_pct,
                tail[2] + candidate["overlap_seconds"],
                tuple(sorted((pair_key,) + tail[3])),
            )
            if is_better(proposal, best):
                best = proposal

        return best

    return list(solve(component_ids)[3])
