"""Deterministic global UTC cutoffs; equal timestamps always stay together."""

import numpy as np
import pandas as pd


def choose_cutoffs(days, ratios=(.8, .1, .1), explicit=None):
    if len(ratios) != 3 or any(x <= 0 for x in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError("Provide three positive ratios summing to one")
    if explicit is not None:
        stamps = [pd.Timestamp(value) for value in explicit]
        stamps = [s.tz_localize("UTC") if s.tzinfo is None else s.tz_convert("UTC") for s in stamps]
        if any(s.value % 10**9 for s in stamps):
            raise ValueError("Explicit cutoffs must use whole seconds")
        result = tuple(int(s.timestamp()) for s in stamps)
    else:
        ordered = sorted(days)
        cumulative = np.cumsum([days[d] for d in ordered])
        result = tuple((int(ordered[np.searchsorted(cumulative, cumulative[-1]*q)]) + 1) * 86400
                       for q in (ratios[0], ratios[0]+ratios[1]))
    if len(result) != 2 or result[0] >= result[1]:
        raise ValueError("Cutoffs must be distinct and increasing; supply explicit dates for short datasets")
    return result


def masks(timestamps, cutoffs):
    validation, test = cutoffs
    return {"train": timestamps < validation,
            "validation": (timestamps >= validation) & (timestamps < test),
            "test": timestamps >= test}
