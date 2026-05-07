from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import shared.child_only_physics_inference as child_only_mod
from shared.child_only_cli_utils import (
    infer_run_id_from_child_path,
    result_to_pretty_json,
)
from shared.child_only_physics_inference import (
    infer_changed_parameter_child_only,
    load_child_df,
)
from shared.noise_snr import (
    apply_signal_noise_only,
    load_noise_rules,
    noise_multiplier_from_snr_db,
)

def run_model_cli(model_id: str) -> None:
    @click.command()
    @click.option(
        "--child",
        required=True,
        type=click.Path(exists=True, path_type=Path),
        help="Child parquet file.",
    )
    @click.option(
        "--change-threshold-rel",
        type=float,
        default=None,
        help="Override global relative change threshold used by child-only inference (default is module value).",
    )
    @click.option(
        "--min-abs-delta",
        type=float,
        default=None,
        help="Override global minimum absolute delta gate for change detection (default is module value).",
    )
    @click.option(
        "--raw-signals-change-threshold-rel",
        type=float,
        default=None,
        help="Override raw-signal relative threshold used for change-point detection (e.g. InclinedPlane raw signals).",
    )
    @click.option(
        "--noise-mult",
        type=float,
        default=None,
        help="Optional Gaussian noise multiplier relative to local RMS envelope.",
    )
    @click.option(
        "--noise-snr-db",
        type=float,
        default=None,
        help="Alternative to --noise-mult. Converted as multiplier = 10^(-snr_db/20).",
    )
    @click.option(
        "--noise-seed",
        type=int,
        default=0,
        show_default=True,
        help="RNG seed used when --noise-mult is provided.",
    )
    @click.option(
        "--v-min",
        type=float,
        default=0.01,
        show_default=True,
        help="Minimum |v| threshold used by some models (e.g. DampedMassBetweenWalls) to ignore weak impacts/segments.",
    )
    def _cli(
        child: Path,
        change_threshold_rel: Optional[float],
        min_abs_delta: Optional[float],
        raw_signals_change_threshold_rel: Optional[float],
        noise_mult: Optional[float],
        noise_snr_db: Optional[float],
        noise_seed: int,
        v_min: float,
    ) -> None:
        child_df = load_child_df(child)
        if change_threshold_rel is not None:
            child_only_mod.CHANGE_THRESHOLD_REL = float(change_threshold_rel)
        if min_abs_delta is not None:
            child_only_mod.MINIMUM_ABS_DELTA = float(min_abs_delta)
        if raw_signals_change_threshold_rel is not None:
            child_only_mod.RAW_SIGNALS_CHANGE_THRESHOLD_REL = float(
                raw_signals_change_threshold_rel
            )
        if noise_mult is not None and noise_snr_db is not None:
            raise click.ClickException(
                "Use either --noise-mult or --noise-snr-db, not both."
            )
        resolved_noise_mult: Optional[float] = None
        if noise_snr_db is not None:
            resolved_noise_mult = noise_multiplier_from_snr_db(float(noise_snr_db))
        elif noise_mult is not None:
            resolved_noise_mult = float(noise_mult)
        if resolved_noise_mult is not None and resolved_noise_mult > 0:
            child_df = apply_signal_noise_only(
                child_df,
                noise_multiplier=float(resolved_noise_mult),
                seed=int(noise_seed),
                rules=load_noise_rules(),
            )
        run_id = infer_run_id_from_child_path(child)
        result = infer_changed_parameter_child_only(
            model_id=model_id,
            child_df=child_df,
            run_id=run_id,
            v_min=float(v_min),
        )
        print(result_to_pretty_json(result))

    _cli()
