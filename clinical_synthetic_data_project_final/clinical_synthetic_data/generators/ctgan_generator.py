from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from ..io_utils import read_dataset_csv

from ..core.patient_schema import (
    CATEGORICAL_FIELDS,
    CONTINUOUS_FIELDS,
    ClassLabel,
    Patient,
)
from ..validation import RejectionStatistics, validate
from .base_generator import BaseGenerator


FRIEDEWALD_NOISE_STD: float = 3.0

class CTGANGenerator(BaseGenerator):

    def __init__(
        self,
        epochs: int = 300,
        batch_size: int = 500,
        enforce_derived_variables: bool = True,
        seed: Optional[int] = None,
        max_total_attempts_factor: int = 500,
        verbose: bool = True,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.enforce_derived_variables = enforce_derived_variables
        self.seed = seed
        self.max_total_attempts_factor = max_total_attempts_factor
        self.verbose = verbose

        self.stats = RejectionStatistics()
        self._synthesizer = None
        self._metadata = None
        self._is_fitted = False

        if seed is not None:
            self._set_seeds(seed)

    @staticmethod
    def _set_seeds(seed: int) -> None:
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _build_metadata(self, df: pd.DataFrame):
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()

        for col in CONTINUOUS_FIELDS:
            if col in df.columns:
                metadata.add_column(col, sdtype="numerical")

        for col in CATEGORICAL_FIELDS:
            if col in df.columns:
                metadata.add_column(col, sdtype="categorical")

        if "class_label" in df.columns:
            metadata.add_column("class_label", sdtype="categorical")

        return metadata

    def fit(self, training_data: Optional[pd.DataFrame] = None) -> None:
        from sdv.single_table import CTGANSynthesizer

        if training_data is None or len(training_data) == 0:
            raise ValueError(
                "CTGAN requires a non-empty dataset for training. "
                "Generate a dataset with the copula (Method 1) first."
            )

        train_df = training_data.drop(columns=["patient_id"], errors="ignore")

        self._metadata = self._build_metadata(train_df)
        self._synthesizer = CTGANSynthesizer(
            metadata=self._metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        self._synthesizer.fit(train_df)
        self._is_fitted = True

    def _sample_raw_batch(
        self,
        class_label: ClassLabel,
        n_rows: int,
    ) -> pd.DataFrame:
        """Samples `n_rows` raw rows conditionally to the class."""
        from sdv.sampling import Condition

        condition = Condition(
            num_rows=n_rows,
            column_values={"class_label": class_label.value},
        )
        return self._synthesizer.sample_from_conditions(conditions=[condition])

    def _enforce_constructive_constraints(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Recalculate the derived variables to ensure R1 and R2 are exactly satisfied."""
        if not self.enforce_derived_variables:
            return df

        df = df.copy()

        df["weight"] = df["bmi"] * (df["height"] / 100.0) ** 2

        noise = np.random.normal(0.0, FRIEDEWALD_NOISE_STD, size=len(df))
        df["total_chol"] = (
            df["ldl"] + df["hdl"] + df["triglycerides"] / 5.0 + noise
        )
        return df

    def sample_class(
        self,
        class_label: ClassLabel,
        n: int,
    ) -> list[Patient]:
        """Samples exactly `n` valid patients from the requested class."""
        if not self._is_fitted:
            raise RuntimeError(
                "CTGAN is not yet fitted: call `fit()` before "
                "`sample_class()`."
            )
        if n <= 0:
            return []

        accepted: list[Patient] = []
        total_attempts = 0
        max_attempts = max(n * self.max_total_attempts_factor, 1000)

        while len(accepted) < n:
            remaining = n - len(accepted)
            batch_size = max(remaining * 2, 200)
            batch_size = min(batch_size, 2000)

            df_batch = self._sample_raw_batch(class_label, batch_size)
            if len(df_batch) == 0:
                continue

            df_batch = self._enforce_constructive_constraints(df_batch)

            for _, row in df_batch.iterrows():
                self.stats.record_attempt(class_label)
                total_attempts += 1

                if total_attempts > max_attempts:
                    raise RuntimeError(
                        f"CTGAN generation impossible for class "
                        f"{class_label.value} : {total_attempts} attempts, "
                        f"only {len(accepted)}/{n} accepted. "
                        f"The model is probably under-trained. "
                        f"Increase `epochs` (recommendation : ≥ 300) "
                        f"or `max_total_attempts_factor`."
                    )

                candidate = row.to_dict()
                candidate["class_label"] = class_label.value

                result = validate(candidate, class_label)
                if result.is_valid:
                    candidate["patient_id"] = Patient.new_id()
                    accepted.append(Patient.from_mapping(candidate))
                    self.stats.record_acceptance(class_label)
                    if len(accepted) >= n:
                        return accepted
                else:
                    self.stats.record_rejection(class_label, result.failed_rule)

        return accepted


def _build_cli_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description="CTGAN Generation (Method 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--training", required=True,
                        help="Training dataset CSV (e.g., dataset_copula.csv).")
    parser.add_argument("--m", "--m-per-class", type=int, default=1000,
                        dest="m_per_class",
                        help="Patients per class to generate.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="CTGAN batch size.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Reproducible seed.")
    parser.add_argument("--output", "-o", default="dataset_ctgan.csv",
                        help="Path to the output CSV file.")
    parser.add_argument("--max-attempts-factor", type=int, default=500,
                        help="Rejection tolerance factor (n × max attempts factor).")
    return parser


def main(argv=None) -> int:
    import logging
    import time
    import pandas as pd
    from ..io_utils import read_dataset_csv
    from ..logging_setup import setup_logging

    setup_logging()
    log = logging.getLogger("pipeline")

    args = _build_cli_parser().parse_args(argv)

    log.info("=" * 60)
    log.info("MMethod 2 — CTGAN")
    log.info(f"  Training       : {args.training}")
    log.info(f"  M par classe   : {args.m_per_class}  →  total {6 * args.m_per_class}")
    log.info(f"  Epochs         : {args.epochs}")
    log.info(f"  Batch size     : {args.batch_size}")
    log.info(f"  Seed           : {args.seed}")
    log.info("=" * 60)

    log.info(f"Loading training dataset…")
    training = read_dataset_csv(args.training)
    log.info(f"  → {len(training)} patients chargés")

    generator = CTGANGenerator(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_total_attempts_factor=args.max_attempts_factor,
        seed=args.seed,
    )

    log.info(f"Entraînement CTGAN ({args.epochs} epochs)…")
    log.info("  (A progress bar will be displayed for real-time tracking.)")
    t0 = time.time()
    generator.fit(training)
    log.info(f"  Training completed in {time.time() - t0:.0f}s")

    log.info("Generation by class…")
    t0 = time.time()
    df = generator.generate_balanced_dataset(m_per_class=args.m_per_class)
    dt = time.time() - t0

    stats = generator.stats.to_report()
    log.info("-" * 60)
    log.info(f"Completed in {dt:.0f}s")
    log.info(f"  Generated patients : {len(df)}")
    log.info(f"  Global rejection rate : {stats['global_rejection_rate']:.1%}")
    log.info(f"  Average attempts/accepted : {stats['average_attempts_per_accepted']:.2f}")

    df.to_csv(args.output, index=False)
    log.info(f"  Saved : {args.output}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
