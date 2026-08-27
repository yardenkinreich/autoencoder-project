"""
review_labels.py — resumable ipywidgets UI to review configs/new_test_set_labels.csv
against the crater crops in data/raw/new_test_set_crops_wide/.

Not a labeling tool — the labels already exist (manual_labels.csv's expert
calls + the lat/lon-matched additional set). This is a REVIEW tool: for each
crater, show the image next to its current (area, degree) label and its
Robbins match quality, and record whether you agree or want to override it.

State lives in a sibling CSV (default: <labels>_review.csv) so the original
new_test_set_labels.csv is never mutated in place, and review progress is
autosaved on every action so you can stop and resume anytime.

Usage (in a notebook):
    from eval import review_labels as R
    store = R.ReviewStore("configs/new_test_set_labels.csv")
    _ = R.launch_reviewer(store, imgs_dir="data/raw/new_test_set_crops_wide")
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

AREA_CLASSES = ["mare", "highland"]
DEGREE_CLASSES = ["v-fr", "r-fr", "v-dr", "r-dr"]


# --------------------------------------------------------------------------
# Review store (resumable)
# --------------------------------------------------------------------------
class ReviewStore:
    """CSV-backed, resumable review state keyed by CRATER_ID, joined against
    the original labels CSV (which is never modified)."""

    COLUMNS = ["CRATER_ID", "reviewed", "agrees", "corrected_area",
               "corrected_degree", "flag_bad_match", "note"]

    def __init__(self, labels_csv: str | Path, review_csv: str | Path | None = None):
        self.labels_csv = Path(labels_csv)
        self.labels = pd.read_csv(self.labels_csv, dtype={"CRATER_ID": str})
        self.review_csv = Path(review_csv) if review_csv else (
            self.labels_csv.with_name(self.labels_csv.stem + "_review.csv")
        )
        self._df = self._init_or_load()

    def _init_or_load(self) -> pd.DataFrame:
        if self.review_csv.exists():
            df = pd.read_csv(self.review_csv, dtype={"CRATER_ID": str})
            df["agrees"] = df["agrees"].astype("boolean")
            new = self.labels[~self.labels["CRATER_ID"].isin(df["CRATER_ID"])]
            if len(new):
                df = pd.concat([df, self._blank_rows(new)], ignore_index=True)
            return df
        return self._blank_rows(self.labels)

    def _blank_rows(self, labels: pd.DataFrame) -> pd.DataFrame:
        n = len(labels)
        return pd.DataFrame({
            "CRATER_ID": labels["CRATER_ID"].astype(str).to_numpy(),
            "reviewed": False,
            "agrees": pd.array([None] * n, dtype="boolean"),
            "corrected_area": "",
            "corrected_degree": "",
            "flag_bad_match": False,
            "note": "",
        })[self.COLUMNS]

    # --- mutation ---
    def set_review(self, crater_id: str, agrees: bool, corrected_area: str = "",
                   corrected_degree: str = "", flag_bad_match: bool = False,
                   note: str = "") -> None:
        m = self._df["CRATER_ID"] == str(crater_id)
        self._df.loc[m, ["reviewed", "agrees", "corrected_area", "corrected_degree",
                         "flag_bad_match", "note"]] = [
            True, agrees, corrected_area, corrected_degree, flag_bad_match, note
        ]
        self.save()

    def save(self) -> None:
        self._df.to_csv(self.review_csv, index=False)

    # --- queries ---
    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def merged(self) -> pd.DataFrame:
        """Original labels joined with review state, one row per crater."""
        return self.labels.merge(self._df, on="CRATER_ID", how="left")

    def progress(self) -> tuple[int, int]:
        return int(self._df["reviewed"].sum()), len(self._df)

    def agreement_counts(self) -> dict:
        done = self._df[self._df["reviewed"]]
        return {
            "agree": int((done["agrees"] == True).sum()),
            "disagree": int((done["agrees"] == False).sum()),
            "flagged_bad_match": int(done["flag_bad_match"].sum()),
        }

    def review_order(self) -> pd.Index:
        """Flagged-low-confidence-match craters first (need the most scrutiny),
        then everything else in labels-file order."""
        if "flagged_low_confidence_match" in self.labels.columns:
            key = self.labels.set_index("CRATER_ID")["flagged_low_confidence_match"]
            key = key.reindex(self._df["CRATER_ID"]).fillna(False).to_numpy()
            return self._df.index[pd.Series(~key, index=self._df.index).argsort(kind="stable")]
        return self._df.index

    def next_unreviewed(self, after_pos: int = -1):
        order = self.review_order()
        todo_pos = [p for p, idx in enumerate(order) if p > after_pos and not self._df.loc[idx, "reviewed"]]
        if not todo_pos:
            todo_pos = [p for p, idx in enumerate(order) if not self._df.loc[idx, "reviewed"]]
        if not todo_pos:
            return None, -1
        pos = todo_pos[0]
        return self._df.loc[order[pos]], pos


def export_reviewed(store: ReviewStore, name: str = "new_test_set_labels_final.csv") -> Path:
    """Apply corrections/overrides and drop bad-match flags, for wiring into
    eval_suite.yaml once review is done."""
    m = store.merged()
    final = m[~m["flag_bad_match"].fillna(False)].copy()
    final["area"] = final["corrected_area"].where(final["corrected_area"].fillna("") != "", final["area"])
    final["degree"] = final["corrected_degree"].where(final["corrected_degree"].fillna("") != "", final["degree"])
    keep = ["CRATER_ID", "lat", "lon", "diam_km", "area", "degree", "source", "reviewed", "agrees"]
    final = final[keep]
    out = store.labels_csv.with_name(name)
    final.to_csv(out, index=False)
    return out


# --------------------------------------------------------------------------
# ipywidgets UI
# --------------------------------------------------------------------------
def launch_reviewer(store: ReviewStore, imgs_dir: str | Path):
    """Build and display the review widget. Returns the top-level widget."""
    import ipywidgets as W
    from IPython.display import display

    imgs_dir = Path(imgs_dir)
    state = {"pos": -1, "shown_id": None}

    # --- widgets ---
    img = W.Image(format="png", width=420, height=420)
    title = W.HTML()
    current_lbl = W.HTML()
    match_lbl = W.HTML()

    area_btn = W.ToggleButtons(options=AREA_CLASSES, description="Area:")
    degree_btn = W.ToggleButtons(options=DEGREE_CLASSES, description="Degree:")
    bad_match_chk = W.Checkbox(value=False, description="bad match (wrong crater entirely)")
    note_txt = W.Text(placeholder="optional note", description="Note:")

    agree_btn = W.Button(description="Agree → Next", button_style="success", icon="check")
    disagree_btn = W.Button(description="Save correction → Next", button_style="warning", icon="pencil")
    skip_btn = W.Button(description="Skip", icon="forward")

    progress_bar = W.IntProgress(min=0, max=len(store.df), description="Progress:")
    progress_lbl = W.HTML()
    counts_lbl = W.HTML()
    msg = W.HTML()

    # --- rendering ---
    def render_progress():
        done, total = store.progress()
        progress_bar.value = done
        progress_lbl.value = f"<b>{done}/{total}</b> reviewed"
        c = store.agreement_counts()
        counts_lbl.value = (f"agree: <b>{c['agree']}</b> &nbsp; "
                            f"disagree: <b>{c['disagree']}</b> &nbsp; "
                            f"flagged bad match: <b>{c['flagged_bad_match']}</b>")

    def load_image(crater_id: str):
        p = imgs_dir / f"{crater_id}.png"
        if p.exists():
            img.value = p.read_bytes()
        else:
            img.value = b""
            msg.value = f"<span style='color:#b00'>missing crop: {p.name}</span>"

    def show(row: pd.Series | None):
        state["shown_id"] = row["CRATER_ID"] if row is not None else None
        if row is None:
            title.value = "<h3>All craters reviewed.</h3>"
            img.value = b""
            return
        cid = row["CRATER_ID"]
        label_row = store.labels.loc[store.labels["CRATER_ID"] == cid].iloc[0]
        title.value = f"<h3>Crater {cid}</h3>"
        current_lbl.value = (f"current label: area=<b>{label_row['area']}</b> "
                             f"&nbsp;|&nbsp; degree=<b>{label_row['degree']}</b> "
                             f"&nbsp;|&nbsp; diam={label_row.get('diam_km', '?'):.2f} km "
                             f"&nbsp;|&nbsp; source={label_row.get('source', '?')}<br>"
                             f"Robbins lat/lon: <b>{label_row['lat']:.5f}, {label_row['lon']:.5f}</b>")
        dist = label_row.get("match_dist_km", 0.0)
        flagged = bool(label_row.get("flagged_low_confidence_match", False))
        color = "#b00" if flagged else "#555"
        match_lbl.value = (f"<span style='color:{color}'>match distance: {dist:.3f} km"
                           f"{'  ⚠️ low-confidence match — extra scrutiny' if flagged else ''}"
                           f"</span>")
        area_btn.value = label_row["area"] if label_row["area"] in AREA_CLASSES else None
        degree_btn.value = label_row["degree"] if label_row["degree"] in DEGREE_CLASSES else None
        bad_match_chk.value = False
        note_txt.value = ""
        msg.value = ""
        load_image(cid)

    def advance():
        row, pos = store.next_unreviewed(after_pos=state["pos"])
        state["pos"] = pos
        show(row)
        render_progress()

    # --- handlers ---
    def on_agree(_=None):
        cid = state["shown_id"]
        if cid is None:
            return
        store.set_review(cid, agrees=True)
        advance()

    def on_disagree(_=None):
        cid = state["shown_id"]
        if cid is None:
            return
        if area_btn.value is None or degree_btn.value is None:
            msg.value = "<span style='color:#b00'>pick both area and degree.</span>"
            return
        label_row = store.labels.loc[store.labels["CRATER_ID"] == cid].iloc[0]
        agrees = (not bad_match_chk.value and area_btn.value == label_row["area"]
                 and degree_btn.value == label_row["degree"])
        store.set_review(
            cid, agrees=agrees,
            corrected_area=area_btn.value, corrected_degree=degree_btn.value,
            flag_bad_match=bad_match_chk.value, note=note_txt.value,
        )
        advance()

    def on_skip(_=None):
        advance()

    agree_btn.on_click(on_agree)
    disagree_btn.on_click(on_disagree)
    skip_btn.on_click(on_skip)

    # --- layout ---
    left = W.VBox([title, img, current_lbl, match_lbl])
    controls = W.VBox([
        area_btn, degree_btn, bad_match_chk, note_txt,
        W.HBox([agree_btn, disagree_btn, skip_btn]), msg,
    ])
    right = W.VBox([progress_bar, progress_lbl, counts_lbl])
    ui = W.HBox([left, controls, right])

    render_progress()
    row, pos = store.next_unreviewed(after_pos=-1)
    state["pos"] = pos
    show(row)
    display(ui)
    return ui
