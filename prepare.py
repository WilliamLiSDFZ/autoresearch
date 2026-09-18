"""Fixed public-data preparation and evaluation for Jigsaw Unintended Bias.

Run `python prepare.py --help` for preparation, verification and scoring commands.
Only NumPy and pandas are required: preparation and evaluation work without a GPU.
This task adapter replaces the upstream language-model/BPE preparation API.
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

TASK_ID = "jigsaw-unintended-bias-in-toxicity-classification"
METRIC_VERSION = "jubias-continuous-auc-v1"
SPLIT_METHOD = "autoresearch-numpy-stratified-target-v1"
IDENTITIES = [
    "male", "female", "homosexual_gay_or_lesbian", "christian", "jewish",
    "muslim", "black", "white", "psychiatric_or_mental_illness",
]
LABEL_COLUMNS = ["target", *IDENTITIES]
TRAIN_COLUMNS = ["id", "comment_text", *LABEL_COLUMNS]
TEST_COLUMNS = ["id", "comment_text"]
ARTIFACT_FILES = {"train.csv", "validation.csv", "test.csv", "split.npz"}
DEFAULT_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.05


def prepared_directory(directory=None):
    """Default stays with the checkout, so /workspace checkouts persist on a PVC."""
    return Path(directory or os.environ.get("AUTORESEARCH_JIGSAW_DIR") or
                Path(__file__).resolve().parent / "results" / "jigsaw-data").expanduser().resolve()


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path, value):
    """Publish a complete JSON file, without exposing a partially written result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _check_hashes(directory, hashes, required):
    directory = Path(directory).resolve()
    if not isinstance(hashes, dict) or set(hashes) != set(required):
        raise ValueError("Manifest must cover exactly the required artifact files")
    for name, expected in hashes.items():
        artifact = directory / name
        if (artifact.is_symlink() or not artifact.resolve().is_relative_to(directory)
                or not artifact.is_file() or digest(artifact) != expected):
            raise ValueError(f"Missing or changed artifact: {artifact}")


def _check_ids(frame, name):
    ids = frame["id"]
    if frame.empty or ids.isna().any() or not ids.is_unique or ids.str.strip().eq("").any():
        raise ValueError(f"{name}: IDs must be nonempty, unique strings; data must not be empty")


def _check_labels(answers):
    missing = set(LABEL_COLUMNS) - set(answers.columns)
    if missing:
        raise ValueError(f"Missing Jigsaw label columns: {sorted(missing)}")
    target = answers["target"].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError("Targets must be finite values in [0, 1]")
    for identity in IDENTITIES:
        values = answers[identity].to_numpy(dtype=np.float64)
        # Missing identity annotations are background, matching the MLEvolve metric.
        observed = values[~np.isnan(values)]
        if not np.isfinite(observed).all() or ((observed < 0) | (observed > 1)).any():
            raise ValueError(f"Invalid identity annotations: {identity}")


def _read_frame(path, columns):
    # Comment strings such as "NA" and "null" are text, not missing annotations.
    converters = {"comment_text": str} if "comment_text" in columns else None
    frame = pd.read_csv(path, dtype={"id": str}, usecols=columns,
                        converters=converters)[columns]
    _check_ids(frame, str(path))
    if "comment_text" in columns:
        frame["comment_text"] = frame["comment_text"].fillna("").astype(str)
    if "target" in columns:
        _check_labels(frame)
    return frame


def _probabilities(values, rows):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (rows,) or not np.isfinite(values).all():
        raise ValueError(f"Expected {rows} finite scalar probabilities; got {values.shape}")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Predictions must be probabilities in [0, 1]")
    return values


def _auc(labels, values):
    """Mann-Whitney ROC-AUC with average ranks for ties; no sklearn dependency."""
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("ROC-AUC requires both positive and negative examples")
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, len(ordered)]
    starts = np.r_[0, ends[:-1]]
    ranks = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    rank_sum = ranks[labels[order]].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def score_components(answers, values):
    """Official composite metric on a public holdout, higher is better.

    Continuous predictions are NEVER thresholded. Ground-truth target and identity
    annotations use >= 0.5. Reject undefined components instead of dropping groups.
    This is not a Kaggle hidden-test or MLE-bench private-test score.
    """
    _check_labels(answers)
    values = _probabilities(values, len(answers))
    labels = answers["target"].to_numpy(dtype=np.float64) >= 0.5

    def component(mask, name):
        positive = int(labels[mask].sum())
        negative = int(mask.sum()) - positive
        if not positive or not negative:
            raise ValueError(f"Undefined AUC for {name}; do not drop metric terms")
        return {"auc": _auc(labels[mask], values[mask]), "rows": positive + negative,
                "positive_count": positive, "negative_count": negative}

    overall = component(np.ones(len(labels), dtype=bool), "overall")
    parts = {"subgroup": [], "bpsn": [], "bnsp": []}
    identities = {}
    for identity in IDENTITIES:
        group = answers[identity].fillna(0).to_numpy(dtype=np.float64) >= 0.5
        masks = (group, (group & ~labels) | (~group & labels),
                 (group & labels) | (~group & ~labels))
        identities[identity] = {}
        for kind, mask in zip(parts, masks):
            term = component(mask, f"{identity}/{kind}")
            parts[kind].append(term["auc"])
            identities[identity][kind] = term

    def power_mean(values):
        return 0.0 if min(values) == 0 else float(np.mean(np.power(values, -5.0)) ** -0.2)

    means = {kind: power_mean(values) for kind, values in parts.items()}
    score = 0.25 * overall["auc"] + 0.25 * sum(means.values())
    return {"metric_version": METRIC_VERSION, "score": score, "maximize": True,
            "overall_auc": overall["auc"], "power_means": means, "identities": identities}


def _check_split(fit, validation, rows):
    for indices in (fit, validation):
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) == 0:
            raise ValueError("Split indices must be nonempty one-dimensional integer arrays")
    combined = np.concatenate([fit, validation])
    if len(combined) != rows or not np.array_equal(np.sort(combined), np.arange(rows)):
        raise ValueError("Train/validation split must cover every source row exactly once")


def _make_split(train, seed, fraction):
    """Own versioned stratified split; import a contract for exact MLEvolve parity."""
    labels = train["target"].to_numpy() >= 0.5
    groups = [np.flatnonzero(~labels), np.flatnonzero(labels)]
    counts = np.array([len(group) for group in groups])
    validation_rows = math.ceil(len(train) * fraction)
    if min(counts) < 2 or not 2 <= validation_rows <= len(train) - 2:
        raise ValueError("Too few examples for a stratified training/validation split")
    desired = counts * validation_rows / len(train)
    allocated = np.clip(np.floor(desired).astype(int), 1, counts - 1)
    while allocated.sum() != validation_rows:
        if allocated.sum() < validation_rows:
            priorities = np.where(allocated < counts - 1, desired - allocated, -np.inf)
            allocated[np.argmax(priorities)] += 1
        else:
            priorities = np.where(allocated > 1, allocated - desired, -np.inf)
            allocated[np.argmax(priorities)] -= 1
    for attempt in range(20):
        rng = np.random.default_rng(seed + attempt)
        shuffled = [rng.permutation(group) for group in groups]
        validation = np.sort(np.concatenate([g[:n] for g, n in zip(shuffled, allocated)]))
        fit = np.sort(np.concatenate([g[n:] for g, n in zip(shuffled, allocated)]))
        try:
            score_components(train.iloc[validation], np.full(len(validation), 0.5))
            return fit, validation, attempt
        except ValueError:
            # Retry only for defined AUC terms, never to select a better model score.
            continue
    raise ValueError("Cannot obtain all 27 bias AUC terms in 20 splits. Before starting "
                     "comparisons, use a larger validation fraction or a valid shared contract.")


def _import_contract(directory, source_hashes, seed, fraction):
    directory = Path(directory).expanduser().resolve()
    manifest = _read_json(directory / "manifest.json")
    body = {key: value for key, value in manifest.items() if key != "contract_id"}
    if _object_digest(body) != manifest.get("contract_id"):
        raise ValueError("Changed MLEvolve contract manifest")
    if (manifest.get("version") != 1 or manifest.get("task_id") != TASK_ID
            or manifest.get("metric_version") != METRIC_VERSION or manifest.get("maximize") is not True
            or manifest.get("split_method") != "stratified-target-v1"):
        raise ValueError("Unsupported MLEvolve task/metric/split contract")
    _check_hashes(directory, manifest.get("files"),
                  {"split.npz", "train_ids.csv", "validation.csv", "test_ids.csv"})
    if (manifest.get("public_train_sha256") != source_hashes["train.csv"]
            or manifest.get("public_test_sha256") != source_hashes["test.csv"]):
        raise ValueError("Public source data does not match the MLEvolve contract")
    if seed is not None and seed != manifest["seed"]:
        raise ValueError("--seed differs from the imported contract")
    if fraction is not None and fraction != manifest["validation_fraction"]:
        raise ValueError("--validation-fraction differs from the imported contract")
    return directory, manifest


def _load_manifest(directory):
    directory = prepared_directory(directory)
    manifest = _read_json(directory / "manifest.json")
    body = {key: value for key, value in manifest.items() if key != "prepared_id"}
    if _object_digest(body) != manifest.get("prepared_id"):
        raise ValueError("Changed preparation manifest")
    if (manifest.get("version") != 1 or manifest.get("task_id") != TASK_ID
            or manifest.get("metric_version") != METRIC_VERSION or manifest.get("maximize") is not True):
        raise ValueError("Unsupported prepared task/metric")
    if manifest.get("prepare_sha256") != digest(__file__):
        raise ValueError("prepare.py differs from the frozen preparation. Use the original task "
                         "base or deliberately prepare a new protocol in a new directory.")
    _check_hashes(directory, manifest.get("files"), ARTIFACT_FILES)
    return manifest


def load_data(prepared_dir=None):
    """Return fit, validation and public test frames, preserving prepared row order.

    Fit all learned transforms only on fit. Validation labels/identities are for
    scoring; they must never be training examples or inference features.
    """
    directory = prepared_directory(prepared_dir)
    manifest = _load_manifest(directory)
    fit = _read_frame(directory / "train.csv", TRAIN_COLUMNS)
    validation = _read_frame(directory / "validation.csv", TRAIN_COLUMNS)
    test = _read_frame(directory / "test.csv", TEST_COLUMNS)
    for name, frame in (("train", fit), ("validation", validation), ("test", test)):
        if len(frame) != manifest[f"{name}_rows"]:
            raise ValueError(f"Changed {name} row count")
    ids = pd.concat([frame["id"] for frame in (fit, validation, test)], ignore_index=True)
    if not ids.is_unique:
        raise ValueError("Train, validation and test IDs must be disjoint")
    with np.load(directory / "split.npz", allow_pickle=False) as split:
        _check_split(split["train"], split["validation"], len(fit) + len(validation))
        if len(split["train"]) != len(fit) or len(split["validation"]) != len(validation):
            raise ValueError("Split lengths differ from prepared CSVs")
    return fit, validation, test


def verify_prepared(prepared_dir=None):
    """Verify frozen code, file hashes, schemas, disjoint IDs and all metric terms."""
    _, validation, _ = load_data(prepared_dir)
    score_components(validation, np.full(len(validation), 0.5))
    return _load_manifest(prepared_directory(prepared_dir))


def prepare(data_dir, output_dir=None, *, seed=None, validation_fraction=None, mlevolve_contract=None):
    """Prepare once or verify an identical existing preparation; never overwrite it."""
    data_dir = Path(data_dir).expanduser().resolve()
    directory = prepared_directory(output_dir)
    if data_dir == directory or data_dir.is_relative_to(directory):
        raise ValueError("Output directory must be separate from the public source directory")
    if data_dir.name == "private":
        raise ValueError("Use prepared/public; private evaluation data is not an input")
    source_hashes = {name: digest(data_dir / name) for name in ("train.csv", "test.csv")}
    imported_dir, imported = (None, None)
    if mlevolve_contract is not None:
        imported_dir, imported = _import_contract(mlevolve_contract, source_hashes, seed, validation_fraction)
        seed, validation_fraction = imported["seed"], imported["validation_fraction"]
    seed = DEFAULT_SEED if seed is None else seed
    fraction = DEFAULT_VALIDATION_FRACTION if validation_fraction is None else validation_fraction
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError("Validation fraction must be strictly between 0 and 1")
    contract_id = imported["contract_id"] if imported else None
    if directory.exists():
        manifest = verify_prepared(directory)
        if (manifest["source_sha256"] != source_hashes or manifest["seed"] != seed
                or manifest["validation_fraction"] != fraction
                or manifest["mlevolve_contract_id"] != contract_id):
            raise ValueError("Existing preparation has different data/split settings. "
                             "Use a new output directory; never overwrite a comparison's data.")
        return manifest

    train = _read_frame(data_dir / "train.csv", TRAIN_COLUMNS)
    test = _read_frame(data_dir / "test.csv", TEST_COLUMNS)
    if train["id"].isin(test["id"]).any():
        raise ValueError("Public train and test IDs overlap")
    if imported:
        with np.load(imported_dir / "split.npz", allow_pickle=False) as split:
            fit_idx, val_idx = split["train"], split["validation"]
        _check_split(fit_idx, val_idx, len(train))
        for artifact, expected_ids in (("train_ids.csv", train["id"]),
                                       ("test_ids.csv", test["id"]),
                                       ("validation.csv", train.iloc[val_idx]["id"])):
            recorded = pd.read_csv(imported_dir / artifact, dtype={"id": str})
            if recorded["id"].tolist() != expected_ids.tolist():
                raise ValueError(f"Source row order differs from imported {artifact}")
        recorded_labels = pd.read_csv(imported_dir / "validation.csv", usecols=LABEL_COLUMNS)[LABEL_COLUMNS]
        actual_labels = train.iloc[val_idx][LABEL_COLUMNS]
        if (not np.allclose(recorded_labels, actual_labels, rtol=1e-12, atol=1e-12, equal_nan=True)
                or not np.array_equal(recorded_labels.fillna(0).to_numpy() >= 0.5,
                                      actual_labels.fillna(0).to_numpy() >= 0.5)):
            raise ValueError("Validation labels differ from imported contract")
        if (len(train) != imported["train_rows"] or len(test) != imported["test_rows"]
                or len(val_idx) != imported["validation_rows"]):
            raise ValueError("Source row counts differ from imported contract")
        attempt, split_method = imported["split_attempt"], imported["split_method"]
    else:
        fit_idx, val_idx, attempt = _make_split(train, seed, fraction)
        split_method = SPLIT_METHOD
    _check_split(fit_idx, val_idx, len(train))
    score_components(train.iloc[val_idx], np.full(len(val_idx), 0.5))

    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".jigsaw-prepare-", dir=directory.parent) as temporary:
        staging = Path(temporary) / "prepared"
        staging.mkdir()
        train.iloc[fit_idx].to_csv(staging / "train.csv", index=False)
        train.iloc[val_idx].to_csv(staging / "validation.csv", index=False)
        test.to_csv(staging / "test.csv", index=False)
        np.savez_compressed(staging / "split.npz", train=fit_idx, validation=val_idx)
        if any(digest(data_dir / name) != expected for name, expected in source_hashes.items()):
            raise ValueError("Source CSV changed during preparation; rerun against stable input")
        manifest = {
            "version": 1, "task_id": TASK_ID, "metric_version": METRIC_VERSION, "maximize": True,
            "seed": seed, "validation_fraction": fraction, "split_attempt": attempt,
            "split_method": split_method, "mlevolve_contract_id": contract_id,
            "train_rows": len(fit_idx), "validation_rows": len(val_idx), "test_rows": len(test),
            "source_sha256": source_hashes, "prepare_sha256": digest(__file__),
            "files": {name: digest(staging / name) for name in sorted(ARTIFACT_FILES)},
        }
        manifest["prepared_id"] = _object_digest(manifest)
        _write_json(staging / "manifest.json", manifest)
        verify_prepared(staging)
        if directory.exists():
            raise FileExistsError(f"Preparation appeared concurrently: {directory}")
        staging.rename(directory)
    return manifest


def evaluate_predictions(predictions, prepared_dir=None):
    """Score a strict id,prediction CSV/DataFrame, or an array in validation order."""
    directory = prepared_directory(prepared_dir)
    manifest = _load_manifest(directory)
    validation = _read_frame(directory / "validation.csv", TRAIN_COLUMNS)
    if isinstance(predictions, (str, os.PathLike)):
        predictions = pd.read_csv(predictions, dtype={"id": str})
    if isinstance(predictions, pd.DataFrame):
        if (list(predictions.columns) != ["id", "prediction"]
                or predictions["id"].tolist() != validation["id"].tolist()):
            raise ValueError("Prediction CSV must have exactly id,prediction columns and "
                             "all validation IDs in their prepared order")
        predictions = predictions["prediction"].to_numpy()
    result = score_components(validation, predictions)
    return {**result, "prepared_id": manifest["prepared_id"], "validation_rows": len(validation)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare", help="Prepare fixed public-data splits once (CPU)")
    setup.add_argument("--data-dir", required=True, help="Public directory containing train.csv and test.csv")
    setup.add_argument("--output-dir", help="Persistent output; defaults to results/jigsaw-data beside this script")
    setup.add_argument("--seed", type=int, help="Default 42, or inherit the imported contract")
    setup.add_argument("--validation-fraction", type=float, help="Default 0.05, or inherit imported contract")
    setup.add_argument("--mlevolve-contract", help="Existing candidate_results/contract for exact split parity")
    verify = commands.add_parser("verify", help="Verify an existing frozen preparation")
    verify.add_argument("--prepared-dir", help="Default AUTORESEARCH_JIGSAW_DIR or results/jigsaw-data")
    evaluate = commands.add_parser("evaluate", help="Score continuous validation probabilities (CPU)")
    evaluate.add_argument("--prepared-dir", help="Default AUTORESEARCH_JIGSAW_DIR or results/jigsaw-data")
    evaluate.add_argument("--predictions", required=True, help="CSV with exactly id,prediction columns")
    evaluate.add_argument("--output", help="Optional metrics JSON outside the frozen prepared directory")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            manifest = prepare(args.data_dir, args.output_dir, seed=args.seed,
                               validation_fraction=args.validation_fraction, mlevolve_contract=args.mlevolve_contract)
            print(f"prepared_dir: {prepared_directory(args.output_dir)}")
            print(f"prepared_id: {manifest['prepared_id']}")
            print(f"rows: train={manifest['train_rows']} validation={manifest['validation_rows']} test={manifest['test_rows']}")
            print("Data ready. The upstream train.py must be adapted to Jigsaw before training; see program.md.")
        elif args.command == "verify":
            manifest = verify_prepared(args.prepared_dir)
            print(f"Verified {TASK_ID}: {manifest['prepared_id']}")
        else:
            if args.output and Path(args.output).resolve().is_relative_to(prepared_directory(args.prepared_dir)):
                raise ValueError("Metrics output must be outside the frozen prepared directory")
            result = evaluate_predictions(args.predictions, args.prepared_dir)
            if args.output:
                _write_json(args.output, result)
            print(f"val_score: {result['score']:.10f}")
            print(f"overall_auc: {result['overall_auc']:.10f}")
            print(f"prepared_id: {result['prepared_id']}")
            print(f"metric_version: {result['metric_version']}")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
