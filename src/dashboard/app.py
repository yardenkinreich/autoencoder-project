"""
Eval dashboard entrypoint.

    PYTHONPATH=src streamlit run src/dashboard/app.py

One app, one browser tab - "Compare Runs" and "Run Deep Dive" (in pages/)
are sidebar-navigable pages of this same running process, not separate
scripts to launch.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from dashboard import data as D

st.set_page_config(page_title="Eval Dashboard", layout="wide")

st.title("Autoencoder eval dashboard")
st.markdown(
    "Browse the full results of one model run (**Run Deep Dive**), or "
    "compare several runs by metric and by training parameters "
    "(**Compare Runs**) - both in the sidebar."
)

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

history_df = D.load_history()
run_dirs = D.list_run_dirs()

col1, col2 = st.columns(2)
# Unique checkpoints, not raw eval_history.csv rows - a checkpoint
# re-evaluated multiple times (e.g. after a bug fix) is one MODEL, not
# several - matches Compare Runs' own drop_duplicates("checkpoint",
# keep="last") convention and history.best_runs()'s dedup.
n_checkpoints = history_df["checkpoint"].nunique() if not history_df.empty else 0
col1.metric("Models evaluated (unique checkpoints)", n_checkpoints)
col2.metric("Eval run directories on disk (logs/**/eval_results/)", len(run_dirs))

if history_df.empty and not run_dirs:
    st.info(
        "No eval runs found yet. Run `PYTHONPATH=src python -m eval.evaluate "
        "--checkpoint ... --autoencoder-model ...` to produce one, then "
        "refresh."
    )
