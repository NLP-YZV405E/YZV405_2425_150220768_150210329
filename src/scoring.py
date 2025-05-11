import json
import os
import ast
import numpy as np
import pandas as pd


def scoring_program(
    truth_file: str,                 # ✱ EDIT – now a required function argument
    prediction_file: str,            # ✱ EDIT – now a required function argument
    score_output: str | None = None  # ✱ EDIT – optional .json path (writes only if supplied)
) -> dict:
    """
    Compute F1 scores (Turkish, Italian, and average) for a submission.

    Parameters
    ----------
    truth_file : str
        Path to the ground-truth CSV.
    prediction_file : str
        Path to the participant's prediction CSV.
    score_output : str | None, default None
        If given, writes a JSON file containing the scores; otherwise just returns them.

    Returns
    -------
    dict
        {"f1-score-tr": float, "f1-score-it": float, "f1-score-avg": float}
    """

    # ✱ EDIT – load the two CSVs via the function arguments
    prediction_df = pd.read_csv(prediction_file)
    truth_df      = pd.read_csv(truth_file)

    # --- sanity checks (unchanged except for message tweaks) -------------------
    required_cols = ["id", "indices", "language"]
    for col in required_cols:
        if col not in prediction_df.columns or col not in truth_df.columns:
            raise ValueError(f"Both files must contain a '{col}' column.")

    if len(prediction_df) != len(truth_df):
        raise ValueError(
            f"Row count mismatch: predictions={len(prediction_df)} vs ground truth={len(truth_df)}"
        )

    if prediction_df["indices"].isnull().any() or truth_df["indices"].isnull().any():
        raise ValueError("Found missing values in the 'indices' column.")

    if prediction_df["language"].isnull().any() or truth_df["language"].isnull().any():
        raise ValueError("Found missing values in the 'language' column.")

    if (prediction_df["indices"] == "").any() or (truth_df["indices"] == "").any():
        raise ValueError("Empty strings detected in 'indices' column.")

    # --- utility ---------------------------------------------------------------
    def parse_indices(x):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid indices format: {x}") from exc

    prediction_df["indices"] = prediction_df["indices"].apply(parse_indices)
    truth_df["indices"]      = truth_df["indices"].apply(parse_indices)

    # --- split by language -----------------------------------------------------
    pred_tr  = prediction_df[prediction_df["language"] == "tr"]
    gold_tr  = truth_df[truth_df["language"] == "tr"]
    pred_it  = prediction_df[prediction_df["language"] == "it"]
    gold_it  = truth_df[truth_df["language"] == "it"]

    # --- F1 helper -------------------------------------------------------------
    def f1_for_dfs(pred_df, gold_df) -> float:
        f1s = []
        for pred, gold in zip(pred_df["indices"], gold_df["indices"]):
            if gold == [-1]:
                f1s.append(1.0 if pred == [-1] else 0.0)
            else:
                pred_set, gold_set = set(pred), set(gold)
                inter = len(pred_set & gold_set)
                prec  = inter / len(pred_set)  if pred_set else 0
                rec   = inter / len(gold_set)  if gold_set else 0
                f1s.append((2 * prec * rec) / (prec + rec) if prec + rec else 0)
        return float(np.mean(f1s))

    # --- compute scores --------------------------------------------------------
    f1_tr  = f1_for_dfs(pred_tr, gold_tr)
    f1_it  = f1_for_dfs(pred_it, gold_it)
    f1_avg = (f1_tr + f1_it) / 2

    scores = {"f1-score-tr": f1_tr, "f1-score-it": f1_it, "f1-score-avg": f1_avg}

    # ✱ EDIT – optionally write to disk
    if score_output:
        os.makedirs(os.path.dirname(os.path.abspath(score_output)), exist_ok=True)  # ✱ EDIT
        with open(score_output, "w", encoding="utf-8") as fp:                       # ✱ EDIT
            json.dump(scores, fp, ensure_ascii=False, indent=2)                     # ✱ EDIT

    return scores
