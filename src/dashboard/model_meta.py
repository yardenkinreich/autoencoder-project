"""
model_meta.py — recover training parameters (family, scratch-vs-pretrained,
epochs, mask ratio, ...) for a checkpoint, from whatever structured metadata
already exists on disk. No training-code changes: this only reads files
train.py/train_dino.py/the Snakefile already write.

Sources, in order of reliability:
    1. stock pretrained checkpoints (mae_pretrained/dino_pretrained at eval
       time) - always a fixed, hardcoded checkpoint outside logs/, identified
       by the --autoencoder-model value alone.
    2. run_manifest.json, when present - written directly by
       train.py/train_dino.py from the in-memory training config at save
       time, so nothing here is re-derived or guessed. Older runs that
       predate this get one retroactively from backfill_manifests.py, which
       just freezes source #3/#4's own output - so picking this first is a
       no-op for those runs and free precedence for new ones.
    3. DINO run directories - always have a full structured config.yaml
       (OmegaConf dump written by DINOv2's own write_config()).
    4. MAE/CAE run directories - the Snakefile copies itself into
       Snakefile.snapshot at train time (present for most, not all, runs);
       its module-level parameter assignments are plain literals.
    5. Fallback: regex over the run directory name itself, which always
       exists and encodes epochs/mask_ratio (or latent_dim)/num_samples/
       diameter range, but never pretrained-vs-scratch.
"""
from __future__ import annotations
import ast
import json
import os
import re
import yaml

_STOCK_SOURCES = {"mae_pretrained", "dino_pretrained"}

# Only these module-level Snakefile variables are ever extracted - anything
# else (RUN_NAME, RUN_DIR, ...) is an f-string/expression, not a literal, and
# is intentionally left alone. PRETRAINED_MODEL/FREEZE_UNTIL are from an
# older Snakefile era (pre-DATA_TAG) that fine-tuned from a named HuggingFace
# checkpoint (e.g. "facebook/vit-mae-large") with the last N transformer
# blocks unfrozen, instead of today's PRETRAINED_WEIGHTS-path scheme.
_SNAPSHOT_VARS = {
    "AUTOENCODER_MODEL", "LATENT_DIM", "EPOCHS", "MASK_RATIO",
    "MIN_DIAMETER", "MAX_DIAMETER", "NUM_SAMPLES", "PRETRAINED_WEIGHTS",
    "DATA_TAG", "PRETRAINED_MODEL", "FREEZE_UNTIL",
}

# An even older Snakefile era used AUTOENCODER_MODEL="cnn" for what's now
# called "cae" - same architecture, renamed since.
_LEGACY_FAMILY_ALIASES = {"cnn": "cae"}

# The current Snakefile always passes --num_samples explicitly (NUM_SAMPLES
# is a required top-level global - see Snakefile), so a real MAE/CAE run
# with no recorded num_samples predates that flag existing at all, not a
# gap - it trained on the entire crater set available at the time, which
# the person who ran these confirmed was 50,000 craters (2026-08-23).
# Never applies to DINO (a different, un-subsampled pipeline) or stock
# checkpoints (never trained on our data at all).
_LEGACY_FULL_DATASET_SIZE = 50000
_ASSIGN_RE = re.compile(r"^([A-Z_][A-Z0-9_]*)\s*=\s*(.+)$")

_DIRNAME_RE = re.compile(
    r"d(?P<dmin>-?[\d.]+)-(?P<dmax>-?[\d.]+)_ep(?P<epochs>\d+)"
    r"(?:_mask(?P<mask>[\d.]+)|_lat(?P<latent>\d+))?"
    r"(?:_n(?P<n>\d+))?"
)

_RUN_DIR_MARKERS = ("config.yaml", "Snakefile.snapshot", "run_manifest.json")


def _empty_record(param_source: str = "unknown") -> dict:
    return {
        "family": None, "source": "unknown", "pretrained_weights": None,
        "epochs": None, "mask_ratio": None, "latent_dim": None,
        "num_samples": None, "diameter_range": None, "data_tag": None,
        "freeze_until": None, "param_source": param_source,
        # only ever populated for dino - see parse_dino_config()'s docstring
        # for why "epochs" alone is misleading for that family.
        "iterations": None,
        # Crop radius as a multiple of crater diameter (see preprocess_2.py's
        # --clean_offset) - MAE/CAE and DINO cut craters out of the SAME
        # underlying mosaic/sigma source at DIFFERENT crop radii, so two runs
        # can share a data_tag (e.g. "wac100m_sigma100") while actually
        # training on differently-framed crops. Confirmed by the person who
        # ran these (2026-08-23): MAE/CAE is always 0.5, DINO is always 1.0 -
        # neither has ever been swept, so this is a fixed family-level fact,
        # not a per-run guess.
        "fov": None,
        # Prototypical-loss auxiliary supervision during DINO finetuning
        # (src/train/prototypical_loss.py) - only ever populated for a
        # finetune run whose config had a "proto: enabled: true" section
        # (see train_dino.py's _write_run_manifest()); None/False for every
        # other run, including plain (non-prototypical) finetunes.
        "proto_loss_enabled": False,
        "proto_loss_weight": None,
        "proto_n_support_per_class": None,
        "proto_split_csv": None,
    }


_FOV_BY_FAMILY = {"mae": 0.5, "cae": 0.5, "dino": 1.0}


def _infer_data_tag_from_path(run_dir: str) -> str | None:
    """When no snapshot/config data_tag is recorded, but the run directory
    already follows the current logs/{family}/{data_tag}/{run_name}
    convention (exactly one directory level between family and the run's own
    leaf dir), that level IS the data tag. Deliberately exact-length-matched
    so it doesn't misfire on deeper legacy paths like
    logs/mae/facebook/vit-mae-large/{run_name}, where "facebook" is not a
    data tag."""
    parts = run_dir.replace(os.sep, "/").rstrip("/").split("/")
    if "logs" not in parts:
        return None
    idx = parts.index("logs")
    if len(parts) == idx + 4:
        return parts[idx + 2]
    return None


def _infer_sigma_data_tag_from_path(run_dir: str) -> str | None:
    """Some older orphan-checkpoint runs encode their data source as a
    .../sigma/{N} or .../processed_sigma/{N} path segment pair instead of a
    clean DATA_TAG - confirmed by the person who ran them: this means the
    same thing as the "wac100m_sigma{N}" data tag used elsewhere (WAC
    100m/px, sigma-{N} gaussian filter)."""
    parts = run_dir.replace(os.sep, "/").rstrip("/").split("/")
    for i, part in enumerate(parts[:-1]):
        if part == "sigma" or part.endswith("_sigma"):
            nxt = parts[i + 1]
            if nxt.isdigit():
                return f"wac100m_sigma{nxt}"
    return None


_QUOTED_SIGMA_PATH_RE = re.compile(r'"([^"]*/sigma/[^"]*)"')


def _infer_data_tag_from_snapshot_text(text: str) -> str | None:
    """Older Snakefile.snapshot eras never recorded a DATA_TAG variable at
    all (that's a newer convention) - the data source is only findable as a
    literal path string inside the preprocess_craters rule's own
    output_dir/np_output/metadata_output params (e.g.
    "data/processed_wac_100m_new/sigma/100/crater_crops"), which the
    UPPERCASE-only module-level scanner above never looks at (these are
    lowercase Snakemake rule parameters, not top-level assignments).
    Confirmed by the person who ran these: same "wac100m_sigma{N}"
    convention _infer_sigma_data_tag_from_path already applies to run
    directory paths - reused here against the first matching quoted path
    found anywhere in the snapshot text instead."""
    m = _QUOTED_SIGMA_PATH_RE.search(text)
    if not m:
        return None
    return _infer_sigma_data_tag_from_path(m.group(1))


# preprocess_craters' own map_file input (which raw/filtered mosaic a run's
# data was cut from) directly identifies the data source even when nothing
# else on disk records it - but ONLY for filenames a human has explicitly
# confirmed (2026-08-23 session, one run at a time); an unrecognized
# map_file must stay unresolved rather than guess at what it means.
_KNOWN_MAP_FILE_TAGS = {
    "Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif": "wac100m_raw",
    "highpass_filtered_lunar_mosaic_sigma_50.tif": "wac100m_sigma50",
}
_MAP_FILE_RE = re.compile(r'map_file\s*=\s*"([^"]+)"')


def _infer_data_tag_from_map_file(text: str) -> str | None:
    """See _KNOWN_MAP_FILE_TAGS - resolves a run's data_tag from the mosaic
    file its own preprocess_craters rule invocation actually read, for the
    (growing) set of filenames someone has confirmed the meaning of."""
    m = _MAP_FILE_RE.search(text)
    if not m:
        return None
    return _KNOWN_MAP_FILE_TAGS.get(os.path.basename(m.group(1)))


_MIN_MAX_DIAMETER_RE = re.compile(
    r'min_diameter\s*=\s*([\d.]+).*?max_diameter\s*=\s*([\d.]+)', re.DOTALL)


def _infer_diameter_range_from_snapshot_text(text: str) -> list[float] | None:
    """Some Snakefile eras hardcode min_diameter/max_diameter as plain
    numbers directly in the preprocess_craters rule's own params: block
    (e.g. `min_diameter=1.0,`) rather than referencing a top-level
    MIN_DIAMETER/MAX_DIAMETER variable - the UPPERCASE-only module-level
    scanner above never sees these (lowercase rule parameters, not
    assignments), so a run using this convention shows drange_unknown even
    though the values are sitting right there in the snapshot."""
    m = _MIN_MAX_DIAMETER_RE.search(text)
    if not m:
        return None
    return [float(m.group(1)), float(m.group(2))]


# Same era gap as diameter above: some Snakefile eras (e.g. the legacy
# PRETRAINED_MODEL/FREEZE_UNTIL fine-tuning convention) never had a
# top-level MASK_RATIO variable at all - the value was only ever a literal
# rule param (`masked_ratio=0.75,` or `mask_ratio = 0.75,`) passed straight
# through to --mask_ratio in the shell command. Deliberately requires
# digits right after "=" so it never matches a per-wildcard
# `mask_ratio = lambda wildcards: ...` edge-case rule, which has no fixed
# value to report.
_MASK_RATIO_RULE_RE = re.compile(r'masked?_ratio\s*=\s*([\d.]+)')


def _infer_mask_ratio_from_snapshot_text(text: str) -> float | None:
    m = _MASK_RATIO_RULE_RE.search(text)
    if not m:
        return None
    return float(m.group(1))


def _find_run_dir(checkpoint_path: str, markers=_RUN_DIR_MARKERS) -> str | None:
    """Walk upward from the checkpoint's directory looking for a run-level
    marker file (config.yaml, Snakefile.snapshot, or run_manifest.json),
    stopping at logs/."""
    d = os.path.dirname(checkpoint_path)
    for _ in range(6):
        if not d or d in (".", os.sep):
            break
        if any(os.path.exists(os.path.join(d, m)) for m in markers):
            return d
        if os.path.basename(d.rstrip(os.sep)) == "logs":
            break
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def parse_manifest(run_dir: str) -> dict:
    """Structured metadata from run_manifest.json, when one exists in the run
    dir. Written directly by train.py/train_dino.py's own in-memory config at
    save time (param_source="run_manifest" in the file), or retroactively for
    older runs by backfill_manifests.py, which stamps in whatever
    extract_model_params() already returned (original param_source
    preserved) - either way the file's own param_source is trusted as-is."""
    path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(path):
        return {}
    manifest = json.load(open(path))
    record = _empty_record()
    record.update({k: manifest[k] for k in record if k in manifest})
    return record


def parse_dino_config(run_dir: str) -> dict:
    """Structured metadata from a DINO run's config.yaml (always written by
    DINOv2's write_config()).

    "epochs" here is NOT the MAE/CAE sense of one full pass over the
    dataset - DINOv2 trains in ITERATIONS, sampling the dataset infinitely,
    and "epoch" is just a config-defined scaling label
    (OFFICIAL_EPOCH_LENGTH iterations per "epoch", see configs/dino_craters.
    yaml). Two runs both reading "epochs: 2" could represent wildly
    different training volume if OFFICIAL_EPOCH_LENGTH differs between them
    - "iterations" (epochs * OFFICIAL_EPOCH_LENGTH) is the actual comparable
    number, computed here so the dashboard can show both rather than "epochs"
    alone, which reads as directly comparable to an MAE/CAE run's epochs
    when it isn't."""
    path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(path):
        return {}
    cfg = yaml.safe_load(open(path)) or {}
    pretrained = (cfg.get("student", {}) or {}).get("pretrained_weights") \
        or (cfg.get("MODEL", {}) or {}).get("WEIGHTS")
    pretrained = pretrained or None  # empty string -> from scratch
    # dataset_path looks like "Craters:root=data/.../sigma/100/dino_wide/craters_wide.dat" -
    # not a filesystem-safe or even short string on its own. Prefer the same
    # sigma/{N} source tag MAE/CAE runs get (e.g. "wac100m_sigma100") when
    # present, since DINO crops are cut from the exact same mosaic/filter
    # source as those runs - just at a wider FOV (see "fov" below) - so the
    # underlying data source, not the DINO-specific wrapper directory name,
    # is what makes two runs actually comparable. Falls back to the
    # containing directory's name (e.g. "dino_wide") for older paths that
    # predate the sigma/{N} convention.
    dataset_path = (cfg.get("train", {}) or {}).get("dataset_path")
    data_tag = None
    if dataset_path:
        raw = dataset_path.split("root=", 1)[-1]
        data_tag = _infer_sigma_data_tag_from_path(raw) or os.path.basename(os.path.dirname(raw)) or None
    epochs = (cfg.get("optim", {}) or {}).get("epochs")
    epoch_length = (cfg.get("train", {}) or {}).get("OFFICIAL_EPOCH_LENGTH")
    return {
        **_empty_record(),
        "family": "dino",
        "source": "finetune" if pretrained else "scratch",
        "pretrained_weights": pretrained,
        "epochs": epochs,
        "iterations": (epochs * epoch_length) if epochs is not None and epoch_length is not None else None,
        "data_tag": data_tag,
        "fov": _FOV_BY_FAMILY["dino"],
        "param_source": "dino_config",
    }


def parse_snakefile_snapshot(run_dir: str, family_hint: str | None = None) -> dict:
    """Structured-ish metadata from an MAE/CAE run's Snakefile.snapshot, when
    one exists - a full copy of the Snakefile as it was at train time, whose
    module-level parameter assignments are plain literals.

    family_hint overrides the file's own recorded AUTOENCODER_MODEL - used
    for the one confirmed historical mislabeling (see
    legacy_run_overrides.py) where the file itself is simply wrong. Trusted
    over the file because it's a direct correction from the person who ran
    it, not a guess; the mask_ratio/latent_dim field-inclusion logic below
    keys off the (possibly-hinted) family too, so the correction is applied
    consistently rather than leaving stale fields from the wrong family."""
    path = os.path.join(run_dir, "Snakefile.snapshot")
    if not os.path.exists(path):
        return {}
    text = open(path).read()
    values = {}
    for line in text.splitlines():
        m = _ASSIGN_RE.match(line.strip())
        if not m or m.group(1) not in _SNAPSHOT_VARS:
            continue
        rhs = m.group(2).split("#", 1)[0].strip()
        try:
            values[m.group(1)] = ast.literal_eval(rhs)
        except (ValueError, SyntaxError):
            continue

    if not values:
        return {}

    raw_family = values.get("AUTOENCODER_MODEL")
    raw_family = _LEGACY_FAMILY_ALIASES.get(raw_family, raw_family)
    family = family_hint or raw_family

    # Both PRETRAINED_WEIGHTS (current convention, a checkpoint path) and
    # PRETRAINED_MODEL (older convention, a HuggingFace model id used with
    # FREEZE_UNTIL-based partial fine-tuning) are MAE-only concepts - every
    # Snakefile version that has ever existed only wires pretrained init
    # into CAE's plain conv autoencoder never (confirmed: the current
    # Snakefile's train_autoencoder rule hardcodes
    # `PRETRAINED_WEIGHTS if AUTOENCODER_MODEL == "mae" else ""`). A CAE
    # run's snapshot can still have a PRETRAINED_MODEL variable sitting
    # around (copy-pasted template, never actually consumed), so this must
    # be family-gated the same way mask_ratio already is below, or CAE runs
    # misreport as "finetune".
    pretrained = ((values.get("PRETRAINED_WEIGHTS") or values.get("PRETRAINED_MODEL") or None)
                 if family == "mae" else None)
    had_pretrained_var = family == "mae" and (
        "PRETRAINED_WEIGHTS" in values or "PRETRAINED_MODEL" in values)
    dmin, dmax = values.get("MIN_DIAMETER"), values.get("MAX_DIAMETER")
    return {
        **_empty_record(),
        "family": family,
        # cae is unconditionally "scratch", not a guess: the Snakefile's
        # train_autoencoder rule hardcodes
        # `PRETRAINED_WEIGHTS if AUTOENCODER_MODEL == "mae" else ""` - CAE
        # has never had a pretrained-init path in this codebase, so there's
        # nothing ambiguous to leave as "unknown" here, unlike MAE without a
        # recorded pretrained-related variable (genuinely could go either
        # way, from an even older snapshot era).
        "source": ("finetune" if pretrained else
                   "scratch" if had_pretrained_var or family == "cae" else "unknown"),
        "pretrained_weights": pretrained,
        "epochs": values.get("EPOCHS"),
        "mask_ratio": ((values.get("MASK_RATIO") or _infer_mask_ratio_from_snapshot_text(text))
                      if family == "mae" else None),
        "latent_dim": values.get("LATENT_DIM") if family == "cae" else None,
        "num_samples": (values.get("NUM_SAMPLES") or
                       (_LEGACY_FULL_DATASET_SIZE if family in ("mae", "cae") else None)),
        "diameter_range": ([dmin, dmax] if dmin is not None and dmax is not None
                          else _infer_diameter_range_from_snapshot_text(text)),
        "data_tag": (values.get("DATA_TAG") or _infer_sigma_data_tag_from_path(run_dir)
                    or _infer_data_tag_from_path(run_dir)
                    or _infer_data_tag_from_snapshot_text(text)
                    or _infer_data_tag_from_map_file(text)),
        # MAE-only, same reasoning as pretrained_weights above: a CAE run's
        # snapshot can still have a stray FREEZE_UNTIL=0 left over from a
        # copy-pasted MAE template (the variable's own comment literally
        # says "For MAE") - CAE has never had a freeze-partial-layers
        # concept, so an unqualified read here would misreport a meaningless
        # leftover 0 as if it were a real recorded fact.
        "freeze_until": values.get("FREEZE_UNTIL") if family == "mae" else None,
        "fov": _FOV_BY_FAMILY.get(family),
        "param_source": "snakefile_snapshot",
    }


def parse_run_dirname(checkpoint_path: str, autoencoder_model: str | None) -> dict:
    """Best-effort fallback: regex over the run directory name, which always
    exists but never encodes pretrained-vs-scratch."""
    # The run dir is the checkpoint's own parent - either directly (orphan
    # checkpoints, sitting right in their run dir with no wrapper) or one
    # level up if it's in the conventional models/ subdir.
    parent = os.path.dirname(checkpoint_path)
    run_dir = os.path.dirname(parent) if os.path.basename(parent) == "models" else parent
    # sigma-pattern checked first: it's the more specific match - the generic
    # 3-level heuristic below would otherwise misread "processed_sigma/100"
    # as data_tag="processed_sigma" (wrong) with "100" as an unrelated run
    # name, instead of recognizing the pair together means sigma=100.
    data_tag = _infer_sigma_data_tag_from_path(run_dir) or _infer_data_tag_from_path(run_dir)

    m = _DIRNAME_RE.search(checkpoint_path)
    if not m:
        # Even when the hyperparameter tokens can't be parsed at all, still
        # record family if the caller already knows it (e.g. inferred from
        # the logs/{family}/... path segment) - "no epoch/mask/etc signal"
        # isn't the same as "no idea what architecture this even is" - and
        # still try the path-based data_tag inferences above.
        return {**_empty_record(param_source="dirname_regex"), "family": autoencoder_model,
                "data_tag": data_tag,
                "num_samples": _LEGACY_FULL_DATASET_SIZE if autoencoder_model in ("mae", "cae") else None,
                "fov": _FOV_BY_FAMILY.get(autoencoder_model)}
    g = m.groupdict()
    return {
        **_empty_record(param_source="dirname_regex"),
        "family": autoencoder_model,
        "source": "unknown",
        "epochs": int(g["epochs"]) if g["epochs"] else None,
        "mask_ratio": float(g["mask"]) if g["mask"] else None,
        "latent_dim": int(g["latent"]) if g["latent"] else None,
        "num_samples": (int(g["n"]) if g["n"] else
                       (_LEGACY_FULL_DATASET_SIZE if autoencoder_model in ("mae", "cae") else None)),
        "diameter_range": [float(g["dmin"]), float(g["dmax"])]
                          if g["dmin"] and g["dmax"] else None,
        "data_tag": data_tag,
        "fov": _FOV_BY_FAMILY.get(autoencoder_model),
    }


def extract_model_params(checkpoint_path: str, autoencoder_model: str | None,
                         family_hint: str | None = None) -> dict:
    """Best available training-parameter record for a checkpoint, trying
    each source in reliability order and falling back as needed.

    family_hint: see parse_snakefile_snapshot() - a manual correction for the
    rare case where a run's own recorded family is confirmed wrong."""
    if autoencoder_model in _STOCK_SOURCES:
        family = "dino" if autoencoder_model == "dino_pretrained" else "mae"
        return {
            **_empty_record(param_source="stock"),
            "family": family, "source": "stock_pretrained",
            "pretrained_weights": checkpoint_path,
        }

    run_dir = _find_run_dir(checkpoint_path)
    if run_dir:
        record = parse_manifest(run_dir)
        if record:
            return record
        if autoencoder_model == "dino":
            record = parse_dino_config(run_dir)
        else:
            record = parse_snakefile_snapshot(run_dir, family_hint=family_hint)
        if record:
            return record

    record = parse_run_dirname(checkpoint_path, family_hint or autoencoder_model)
    return record
