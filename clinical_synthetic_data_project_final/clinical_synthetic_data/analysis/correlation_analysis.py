from __future__ import annotations
import numpy as np
import pandas as pd

from ..config.correlations import BASE_CORRELATION_MATRIX, COPULA_VARIABLES
from ..core.patient_schema import CONTINUOUS_FIELDS


def pearson_correlation_matrix(
    df: pd.DataFrame,
    variables: tuple[str, ...] = CONTINUOUS_FIELDS,
) -> pd.DataFrame:
    cols = [v for v in variables if v in df.columns]
    return df[cols].corr(method="pearson")


def spearman_correlation_matrix(
    df: pd.DataFrame,
    variables: tuple[str, ...] = CONTINUOUS_FIELDS,
) -> pd.DataFrame:
    cols = [v for v in variables if v in df.columns]
    return df[cols].corr(method="spearman")


def correlation_matrix_by_class(
    df: pd.DataFrame,
    method: str = "pearson",
    variables: tuple[str, ...] = CONTINUOUS_FIELDS,
) -> dict[str, pd.DataFrame]:
   #voir si certaines corrélations s'expriment plus fortement dans une classe 
    cols = [v for v in variables if v in df.columns]
    return {
        cls: g[cols].corr(method=method)
        for cls, g in df.groupby("class_label")
    }


def correlation_difference(
    matrix_a: pd.DataFrame,
    matrix_b: pd.DataFrame,
) -> pd.DataFrame:
    common = matrix_a.columns.intersection(matrix_b.columns)
    return matrix_a.loc[common, common] - matrix_b.loc[common, common]


def frobenius_distance(
    matrix_a: pd.DataFrame,
    matrix_b: pd.DataFrame,
) -> float:
    diff = correlation_difference(matrix_a, matrix_b)
    return float(np.linalg.norm(diff.values, ord="fro"))


def empirical_vs_target_correlation(df: pd.DataFrame) -> pd.DataFrame:
    target = pd.DataFrame(
        BASE_CORRELATION_MATRIX,
        index=COPULA_VARIABLES,
        columns=COPULA_VARIABLES,
    )
    empirical = pearson_correlation_matrix(df, variables=COPULA_VARIABLES)

    rows = []
    for i, var1 in enumerate(COPULA_VARIABLES):
        for j, var2 in enumerate(COPULA_VARIABLES):
            if i < j:   # triangle supérieur strict
                rows.append({
                    "var1": var1,
                    "var2": var2,
                    "target": float(target.loc[var1, var2]),
                    "empirical": float(empirical.loc[var1, var2]),
                    "delta": float(empirical.loc[var1, var2] - target.loc[var1, var2]),
                })

    result = pd.DataFrame(rows)
    return result.reindex(result["delta"].abs().sort_values(ascending=False).index)
