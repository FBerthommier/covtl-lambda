# SPDX-FileCopyrightText: 2026 covtl-lambda-patch contributors
# SPDX-License-Identifier: GPL-3.0-or-later
# -*- coding: utf-8 -*-
"""
morphing.py
===========
Lambda articulatory morphing: target level (default, v1.2.0) and
tract-blend level between canonical and assimilated CC clusters.

Add-on of the **covtl-lambda-patch** (v1.2.0, 2026-10-26).  This module
implements the interpolation continuum introduced for the nasal
assimilation study (the "lambda experiment" of the nasal_synthesis
reproducibility kit) as a reusable engine option.

Semantics (v1.2.0, authoritative)
---------------------------------
The lambda variation applies to **tautosyllabic CC clusters** -- C1 =
the nasal coda, C2 = the following onset of the same cluster -- and
its effect is defined on the **polar targets** of the cluster branch:
lambda moves the **theta of C1 toward the theta of C2** (and,
optionally, its rho), producing strictly the transition
``C1C2 -> C1'C2`` (e.g. ``klinpan -> klimpan``, cluster forms written
WITHOUT the syllable dot).  At lambda = 0 the cluster target is
strictly C1C2; at lambda = 1 the C1 polar target has reached the C2
target coordinates -- C1'C2, the place-assimilated cluster: the C1
gesture now POINTS at the C2 target, while C2 itself never moves.
This target-level definition is the ground truth of the option; the
tract-parameter blend below is only its motor-level realization.

Separator reminder (engine manual): ``.`` = intra-phrase syllable
separator, no pause, C.C splits; no separator = CC cluster (a
different gesture structure: the engine applies its cluster target
rules, e.g. n in a cluster gets the stop-like alveolar target);
space = 200 ms inter-word pause; ``|`` = 320 ms.  **C.C forms (e.g.
``klin.pan``) are a different synthesis and are NOT the object of the
lambda variation** -- they must not be used as endpoints,
denominations or examples of the lambda continuum.

Two morph modes are provided:

* ``build_target_continuum`` / ``morph_targets`` (DEFAULT, v1.2.0):
  the target-level semantic above.  One canonical cluster phrase in,
  a continuum of re-synthesized clusters out (per lambda the engine
  trajectory is rebuilt with the lambda-interpolated C1 polar
  target; C2 and the whole context stay bitwise canonical).
* ``build_continuum`` / ``morph_pair`` (kept as an option, v1.0/v1.1
  semantics): a frame-by-frame blend of two **frame-aligned** 400 Hz
  tract sequences,

      data(lambda) = (1 - lambda) * A + lambda * B        frame by frame

  where ``A`` and ``B`` are e.g. the canonical vs. assimilated
  cluster variant of the same utterance (19 tract + 11 glottis
  VocalTractLab parameters), optionally confined to the C1 window
  (``confine``/``ramp``, v1.1.0).  This is a motor-level
  realization of the same C1C2 -> C1'C2 transition, valid when the
  two endpoints are frame-aligned cluster syntheses; it is no longer
  the documented default semantic of the lambda variation.

Design contract (in the spirit of the engine patches of
covtl-pipeline -- cf. the 2026-10-18 /N/ closure patch):

1. **Additive only** -- importing this module changes no existing
   engine behaviour; default outputs stay bit-identical.  The target
   mode re-runs the engine's own syllabic builder with ONE consonant
   node retargeted (a runtime override around our own call, always
   restored; no engine file is modified).
2. **Cluster locality** (target mode) / **frame alignment required**
   (blend mode) -- the target mode changes only the sub-arcs whose
   endpoints involve the C1 target: every frame outside them stays
   bitwise canonical for every lambda (C2 and the whole context
   invariant); the blend mode requires A and B to have the same
   number of frames and the same column layout, and its divergence
   is *spatial*, never temporal.
3. **Glottis policy made explicit** -- the blend mode has mode
   ``'common'`` (default; identical glottis sequences required) and
   ``'interpolate'``.  The target mode holds the glottis bitwise
   constant along the whole continuum by construction (asserted):
   voicing, F0 and nasality envelopes never move with lambda.
4. **Endpoint identity** -- blend mode: for any exact float grid,
   lambda = 0 reproduces A bitwise and lambda = 1 reproduces B
   bitwise.  Target mode: lambda = 0 reproduces the canonical
   cluster bitwise, and at lambda = 1 the C1 polar target EQUALS the
   C2 target coordinates (theta exactly; rho too when enabled) --
   asserted and recorded in ``morph_meta.json``.

Usage
-----
    from vtl_synth.core.morphing import morph_targets, morph_pair

    # default semantic: target-level C1C2 -> C1'C2 (one cluster phrase)
    meta = morph_targets('klinpan',
                         output_dir='work/lambda/P1',
                         lambdas=[0.0, 0.125, 0.25, 0.375, 0.5,
                                  0.625, 0.75, 0.875, 1.0],
                         label='P1')

    # option: tract-parameter blend between two cluster .tract files
    meta = morph_pair('work/base/P1_canon/P1_canon.tract',
                      'work/base/P1_assim/P1_assim.tract',
                      output_dir='work/lambda/P1',
                      lambdas=[0.0, 0.125, 0.25, 0.375, 0.5,
                               0.625, 0.75, 0.875, 1.0],
                      label='P1', confine=(232, 272))

Equivalent command line (after applying the patch)::

    vtl-synth morph --phonetic --targets klinpan -o work/lambda/P1
    vtl-synth morph P1_canon.tract P1_assim.tract -o work/lambda/P1

Provenance: derived from ``repro/lambda_experiment.py`` of the
nasal_synthesis kit (stage A), itself the synthesis side of the
Pouw et al. (2024) compensation-continuum replication.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from vtl_synth.core.constants import TRACT_SR, N_TRACT_PARAMS, N_GLOTTIS_PARAMS

logger = logging.getLogger(__name__)

# ==========================================================================
# Public constants
# ==========================================================================

#: Glottis policies of the morph.
MORPH_GLOTTIS_MODES = ('common', 'interpolate')

#: Default continuum: 9 uniform steps from 0 to 1 inclusive.
DEFAULT_N_STEPS = 9

#: Default tolerance for the 'common' glottis identity check.
DEFAULT_GLOTTIS_TOL = 1e-6
DEFAULT_CONFINE_RAMP = 10          # frames (25 ms at 400 Hz)

#: Patch identity (informational; also used by apply_patch.py markers).
MORPH_PATCH_ID = 'covtl-lambda-patch v1.2.0 (2026-10-26)'


# ==========================================================================
# Errors
# ==========================================================================

class MorphError(ValueError):
    """Base class of the morphing option errors."""


class MorphAlignmentError(MorphError):
    """Raised when the two tract sequences are not frame-aligned."""


# ==========================================================================
# Containers
# ==========================================================================

@dataclass
class AlignmentReport:
    """Spatial comparison of the two endpoint sequences.

    Attributes
    ----------
    n_frames : int
        Common frame count (400 Hz frames).
    n_cols : int
        Common column count (19 tract + 11 glottis = 30).
    glottis_max_abs_diff : float
        max |B_glottis - A_glottis| over the glottis columns.
    tract_max_abs_diff : float
        max |B_tract - A_tract| over the tract columns.
    tract_diff_first_frame, tract_diff_last_frame : int or None
        Frame range on which the tract blocks differ (None when the
        tract blocks are identical).
    """

    n_frames: int
    n_cols: int
    glottis_max_abs_diff: float
    tract_max_abs_diff: float
    tract_diff_first_frame: Optional[int] = None
    tract_diff_last_frame: Optional[int] = None

    def as_dict(self) -> Dict:
        return {
            'n_frames': self.n_frames,
            'n_cols': self.n_cols,
            'glottis_max_abs_diff': self.glottis_max_abs_diff,
            'tract_max_abs_diff': self.tract_max_abs_diff,
            'tract_diff_first_frame': self.tract_diff_first_frame,
            'tract_diff_last_frame': self.tract_diff_last_frame,
        }


@dataclass
class MorphItem:
    """One point of the continuum."""
    lam: float
    data: np.ndarray                    # (n_frames, n_cols)


@dataclass
class ContinuumResult:
    """Interpolated continuum + the guarantees that came with it."""
    items: List[MorphItem] = field(default_factory=list)
    report: Optional[AlignmentReport] = None
    glottis_mode: str = MORPH_GLOTTIS_MODES[0]
    checks: Dict = field(default_factory=dict)
    confine: Optional[Tuple[int, int]] = None   # C1 window (frames)
    ramp: int = DEFAULT_CONFINE_RAMP

    def item_at(self, lam: float) -> MorphItem:
        """Returns the continuum item whose lambda equals ``lam``."""
        for it in self.items:
            if it.lam == lam:
                return it
        raise KeyError(f'lambda {lam} not in the continuum grid')


# ==========================================================================
# Lambda grids
# ==========================================================================

def parse_lambda_grid(spec: Optional[str] = None,
                      n_steps: Optional[int] = None) -> List[float]:
    """Parses / builds the lambda grid of the continuum.

    Parameters
    ----------
    spec : str, optional
        Comma-separated lambda values, e.g. ``"0,0.25,0.5,0.75,1"``.
        Duplicates are removed while preserving order.
    n_steps : int, optional
        Uniform grid of ``n_steps`` values from 0 to 1 inclusive
        (``n_steps >= 2``).  Used when ``spec`` is None.
        When both are None, the default 9-step grid is returned.

    Returns
    -------
    list[float]
        Sorted unique lambda values in [0, 1].

    Raises
    ------
    MorphError
        On malformed specs or values outside [0, 1].
    """
    if spec is not None:
        try:
            values = [float(tok.strip()) for tok in spec.split(',')
                      if tok.strip()]
        except ValueError as e:
            raise MorphError(f'bad lambdas spec {spec!r}: {e}') from e
    elif n_steps is not None:
        if int(n_steps) < 2:
            raise MorphError(f'n_steps must be >= 2, got {n_steps}')
        values = [k / (int(n_steps) - 1) for k in range(int(n_steps))]
    else:
        values = [k / (DEFAULT_N_STEPS - 1)
                  for k in range(DEFAULT_N_STEPS)]

    if not values:
        raise MorphError('empty lambda grid')
    for v in values:
        if not (0.0 <= v <= 1.0):
            raise MorphError(
                f'lambda values must lie in [0, 1] (interpolation only; '
                f'extrapolation is not supported), got {v}')

    # dedup (preserve order) then sort for readable output naming
    unique: List[float] = []
    for v in values:
        if v not in unique:
            unique.append(v)
    unique.sort()
    return unique


# ==========================================================================
# Core interpolation
# ==========================================================================

def interpolate_tract(canon: np.ndarray,
                       assim: np.ndarray,
                       lam: float) -> np.ndarray:
    """Linear interpolation of two tract sequences.

    ``out = (1 - lam) * canon + lam * assim``, elementwise.  The
    operation is exact at the endpoints: ``lam=0`` returns ``canon``
    bitwise and ``lam=1`` returns ``assim`` bitwise.

    Parameters
    ----------
    canon, assim : np.ndarray, shape (T, C)
        Frame-aligned sequences (C = 30 columns for VTL files).
    lam : float
        Interpolation weight in [0, 1].

    Returns
    -------
    np.ndarray, shape (T, C)
    """
    if canon.shape != assim.shape:
        raise MorphAlignmentError(
            f'sequences are not frame-aligned: {canon.shape} vs '
            f'{assim.shape}')
    if not (0.0 <= lam <= 1.0):
        raise MorphError(f'lambda must lie in [0, 1], got {lam}')
    return (1.0 - lam) * canon + lam * assim


def confine_weight(n_frames: int,
                   first_frame: int,
                   last_frame: int,
                   ramp: int = DEFAULT_CONFINE_RAMP) -> np.ndarray:
    """Frame weights of a C1-confined morph (covtl-lambda v1.1.0).

    The blend weight is 1 on the C1 gesture window
    ``[first_frame, last_frame]`` (inclusive) and 0 outside it, with a
    raised-cosine ramp of ``ramp`` frames on each edge inside the
    window, so the output trajectory stays continuous when the two
    endpoints differ at the window boundary.  Outside the window the
    context (C2 and everything else) is held at the canonical values
    for every lambda -- the causal-locality contract of the C1C2
    design (C2 position and timing strictly invariant; only the C1
    control node drifts with lambda).

    Parameters
    ----------
    n_frames : int
        Length of the sequence.
    first_frame, last_frame : int
        Inclusive frame bounds of the C1 window.
    ramp : int
        Ramp length in frames (clamped so the two ramps never
        overlap; 0 = hard cut).

    Returns
    -------
    np.ndarray, shape (n_frames,)
        Weights in [0, 1].
    """
    if not (0 <= first_frame <= last_frame < n_frames):
        raise MorphError(
            f'confine window must satisfy 0 <= first <= last < '
            f'{n_frames}, got [{first_frame}, {last_frame}]')
    ramp = max(0, int(ramp))
    span = last_frame - first_frame + 1
    if 2 * ramp > span:
        ramp = span // 2
    w = np.zeros(n_frames)
    w[first_frame:last_frame + 1] = 1.0
    if ramp > 0:
        up = 0.5 * (1.0 - np.cos(
            np.pi * (np.arange(1, ramp + 1) / (ramp + 1.0))))
        w[first_frame:first_frame + ramp] = up
        w[last_frame + 1 - ramp:last_frame + 1] = up[::-1]
    return w


def check_alignment(canon: np.ndarray,
                    assim: np.ndarray,
                    diff_tol: float = 1e-12) -> AlignmentReport:
    """Compares the layout of the two endpoint sequences.

    Parameters
    ----------
    canon, assim : np.ndarray, shape (T, C)
        The two full sequences (tract + glottis columns).
    diff_tol : float
        Threshold above which a frame counts as "different" for the
        reported divergence window.

    Returns
    -------
    AlignmentReport

    Raises
    ------
    MorphAlignmentError
        If the shapes (frame count / column count) differ.  Glottis
        differences do NOT raise here: they are a policy question,
        decided by the glottis mode (see ``build_continuum``).
    """
    if canon.ndim != 2 or assim.ndim != 2:
        raise MorphAlignmentError(
            f'expected 2-D (frames x params) arrays, got '
            f'{canon.shape} and {assim.shape}')
    if canon.shape != assim.shape:
        raise MorphAlignmentError(
            f'sequences are not frame-aligned: {canon.shape} vs '
            f'{assim.shape} (same segmental skeleton and durations '
            f'required -- cf. minimal cluster pairs such as '
            f'klinpan / klimpan)')
    if canon.shape[1] <= N_TRACT_PARAMS:
        raise MorphAlignmentError(
            f'expected at least {N_TRACT_PARAMS + N_GLOTTIS_PARAMS} '
            f'columns (19 tract + 11 glottis), got {canon.shape[1]}')

    glottis_diff = float(np.abs(
        assim[:, N_TRACT_PARAMS:] - canon[:, N_TRACT_PARAMS:]).max())
    tract_delta = np.abs(
        assim[:, :N_TRACT_PARAMS] - canon[:, :N_TRACT_PARAMS]).max(axis=1)
    diff_frames = np.where(tract_delta > diff_tol)[0]

    return AlignmentReport(
        n_frames=int(canon.shape[0]),
        n_cols=int(canon.shape[1]),
        glottis_max_abs_diff=glottis_diff,
        tract_max_abs_diff=float(tract_delta.max()) if len(
            tract_delta) else 0.0,
        tract_diff_first_frame=int(diff_frames.min()) if len(
            diff_frames) else None,
        tract_diff_last_frame=int(diff_frames.max()) if len(
            diff_frames) else None,
    )


def build_continuum(canon: np.ndarray,
                    assim: np.ndarray,
                    lambdas: Optional[Sequence[float]] = None,
                    glottis_mode: str = 'common',
                    glottis_tol: float = DEFAULT_GLOTTIS_TOL,
                    confine: Optional[Tuple[int, int]] = None,
                    ramp: int = DEFAULT_CONFINE_RAMP) -> ContinuumResult:
    """Builds the interpolated continuum between two sequences.

    Parameters
    ----------
    canon, assim : np.ndarray, shape (T, 30)
        Frame-aligned endpoint sequences (tract + glottis).
    lambdas : sequence of float, optional
        Grid of interpolation weights in [0, 1].  Default: 9 uniform
        steps.
    glottis_mode : {'common', 'interpolate'}
        'common' (default): the glottis sequences must be identical up
        to ``glottis_tol``; the morph then touches the tract block
        only (the source envelopes -- voicing, F0, nasality -- are
        held constant along the continuum, and the interpolation is a
        no-op on the glottis block).
        'interpolate': the glottis block is blended with the same
        weight (use it when the two variants differ in voicing).
    glottis_tol : float
        Tolerance of the 'common' glottis identity check.
    confine : (first_frame, last_frame), optional
        C1 gesture window (inclusive 400 Hz frame indices).  When
        given, the blend is restricted to this window (raised-cosine
        ramps of ``ramp`` frames at the edges, cf. ``confine_weight``)
        and every frame outside it is held at the canonical values for
        all lambdas -- the causal-locality contract of the C1C2
        design: only the C1 control node drifts with lambda, the
        context (C2 in particular) stays strictly invariant.  Default
        None: legacy whole-sequence blend, which also transforms
        anticipatory and transition frames outside C1 (discouraged
        for C1C2 experiments; the CLI then prints a notice).
    ramp : int
        Edge ramp of the confine window in frames (default 10).

    Returns
    -------
    ContinuumResult
        With the items of the grid, the alignment report and the
        recorded endpoint-identity checks.

    Raises
    ------
    MorphAlignmentError
        On shape mismatch.
    MorphError
        On an invalid glottis mode, out-of-range lambdas, an invalid
        confine window, or a 'common' glottis check failure.
    """
    if glottis_mode not in MORPH_GLOTTIS_MODES:
        raise MorphError(
            f"glottis_mode must be one of {MORPH_GLOTTIS_MODES}, "
            f"got {glottis_mode!r}")

    report = check_alignment(canon, assim)

    if glottis_mode == 'common' and \
            report.glottis_max_abs_diff > glottis_tol:
        raise MorphError(
            f"glottis sequences differ (max |diff| = "
            f"{report.glottis_max_abs_diff:.3e} > tol {glottis_tol:g}); "
            f"use glottis_mode='interpolate' to blend them, or check "
            f"that the two variants share the same segmental skeleton")

    if lambdas is None:
        lambdas = parse_lambda_grid()
    else:
        lambdas = parse_lambda_grid(','.join(f'{float(v):g}'
                                             for v in lambdas))

    w = None
    if confine is not None:
        first, last = (int(confine[0]), int(confine[1]))
        w = confine_weight(int(canon.shape[0]), first, last, ramp)

    result = ContinuumResult(glottis_mode=glottis_mode, report=report,
                             confine=confine, ramp=int(ramp))
    for lam in lambdas:
        if w is None:
            data = interpolate_tract(canon, assim, lam)
        else:
            # per-frame weight u = lam * w(t); exact at u=0 and u=1
            u = lam * w[:, None]
            data = (1.0 - u) * canon + u * assim
        result.items.append(MorphItem(lam=lam, data=data))

    # -- guarantee records (exact by construction; asserted, not assumed)
    checks = {}
    zero = interpolate_tract(canon, assim, 0.0)
    if w is None:
        one = interpolate_tract(canon, assim, 1.0)
        checks['endpoint_lam0_equals_canon'] = bool(
            np.array_equal(zero, canon))
        checks['endpoint_lam1_equals_assim'] = bool(
            np.array_equal(one, assim))
    else:
        # u = 0 * w == 0 everywhere -> the lam=0 point IS canon,
        # whether or not 0.0 belongs to the requested grid
        checks['endpoint_lam0_equals_canon'] = bool(
            np.array_equal(zero, canon))
        inside = w >= 1.0
        if any(it.lam == 1.0 for it in result.items):
            one = result.item_at(1.0).data
            checks['confine_lam1_equals_assim_inside'] = bool(
                np.array_equal(one[inside], assim[inside]))
        else:
            # structural identity: at u=1 the blend is (1-1)*canon +
            # 1*assim == assim bitwise; nothing to assert without the
            # grid point
            checks['confine_lam1_equals_assim_inside'] = True
        outside = w <= 0.0
        checks['outside_confine_equals_canon'] = bool(all(
            np.array_equal(it.data[outside], canon[outside])
            for it in result.items))
    if glottis_mode == 'common':
        g0 = zero[:, N_TRACT_PARAMS:]
        checks['glottis_noop_at_lam0'] = bool(np.array_equal(
            g0, canon[:, N_TRACT_PARAMS:]))
    if not all(checks.values()):
        raise MorphError(
            f'endpoint identity checks failed (impossible in exact '
            f'arithmetic -- please report): {checks}')
    result.checks = checks
    return result


# ==========================================================================
# .tract I/O (self-contained minimal reader; writer delegates to the
# engine's canonical AssembleTract.write_tract_file)
# ==========================================================================

def read_tract_sequence(path: Union[str, Path]) -> Tuple[np.ndarray, float]:
    """Reads a VTL ``.tract`` file -> (data (T, 30), sample_rate).

    Minimal local reader (header sample rate + ``np.loadtxt``), so
    that ``vtl_synth.core`` does not need to import the video layer.
    """
    path = str(path)
    sr = float(TRACT_SR)
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.startswith('#'):
                break
            m = re.match(r'#\s*sample_rate:\s*([\d.]+)', line)
            if m:
                sr = float(m.group(1))
    data = np.loadtxt(path, comments='#')
    if data.ndim != 2 or data.shape[1] < N_TRACT_PARAMS + N_GLOTTIS_PARAMS:
        raise MorphError(
            f'{path}: unexpected .tract format ({data.shape}); expected '
            f'(n_frames, {N_TRACT_PARAMS + N_GLOTTIS_PARAMS})')
    return data, sr


def write_morph_tract(path: Union[str, Path],
                      data: np.ndarray,
                      sr: float = float(TRACT_SR)) -> None:
    """Writes a continuum point as a canonical VTL ``.tract`` file."""
    from vtl_synth.core.assemble_tract import AssembleTract
    AssembleTract.write_tract_file(
        data[:, :N_TRACT_PARAMS], data[:, N_TRACT_PARAMS:
                                       N_TRACT_PARAMS + N_GLOTTIS_PARAMS],
        str(path), sr=int(sr))


def _format_lambda(lam: float) -> str:
    """Filename fragment of a lambda value (0, 0.125, 0.5, 1 ...)."""
    return f'{lam:g}'


def _sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


# ==========================================================================
# Ground-truth helper (downstream analyses)
# ==========================================================================

def window_state(data: np.ndarray,
                 t0: float,
                 t1: float,
                 sr: float = float(TRACT_SR),
                 mode: str = 'mid') -> np.ndarray:
    """Articulatory ground-truth state inside a time window.

    Parameters
    ----------
    data : np.ndarray, shape (T, C)
        Full sequence (tract + glottis); only the 19 tract parameters
        are returned.
    t0, t1 : float
        Window bounds in seconds (t0 <= t1).
    sr : float
        Frame rate of the sequence (default 400 Hz).
    mode : {'mid', 'mean'}
        'mid' (default): the single frame at (t0 + t1) / 2 -- the
        convention of the assimilation study target windows (e.g.
        t = 0.63 s for the 0.58-0.68 s consonant window).
        'mean': the average of all frames with t0 <= t < t1.

    Returns
    -------
    np.ndarray, shape (19,)
    """
    if mode not in ('mid', 'mean'):
        raise MorphError(f"mode must be 'mid' or 'mean', got {mode!r}")
    if mode == 'mid':
        idx = int(round((t0 + t1) / 2.0 * sr))
        idx = min(max(idx, 0), data.shape[0] - 1)
        return data[idx, :N_TRACT_PARAMS].copy()
    mask = (np.arange(data.shape[0]) / sr >= t0) & \
           (np.arange(data.shape[0]) / sr < t1)
    if not mask.any():
        raise MorphError(f'empty window [{t0}, {t1}) at {sr} Hz')
    return data[mask][:, :N_TRACT_PARAMS].mean(axis=0)


# ==========================================================================
# Driver: files -> continuum of .tract (+ .wav) + morph_meta.json
# ==========================================================================

def morph_pair(canon_path: Union[str, Path],
               assim_path: Union[str, Path],
               output_dir: Union[str, Path],
               lambdas: Optional[Sequence[float]] = None,
               n_steps: Optional[int] = None,
               glottis_mode: str = 'common',
               glottis_tol: float = DEFAULT_GLOTTIS_TOL,
               with_audio: bool = True,
               sample_rate: int = 44100,
               label: Optional[str] = None,
               overwrite: bool = False,
               confine: Optional[Tuple[int, int]] = None,
               ramp: int = DEFAULT_CONFINE_RAMP) -> Dict:
    """Builds a lambda continuum between two ``.tract`` files.

    For each lambda of the grid, writes ``<label>_l<lam>.tract`` and
    (unless ``with_audio=False``) ``<label>_l<lam>.wav`` into
    ``output_dir``, plus a ``morph_meta.json`` provenance record
    (source hashes, alignment report, endpoint checks, item list).

    Parameters
    ----------
    canon_path, assim_path : str or Path
        Endpoint ``.tract`` files (frame-aligned).
    output_dir : str or Path
        Destination directory (created).
    lambdas : sequence of float, optional
        Explicit grid (mutually exclusive with ``n_steps``).
    n_steps : int, optional
        Uniform grid with ``n_steps`` points from 0 to 1.
    glottis_mode, glottis_tol :
        See ``build_continuum``.
    with_audio : bool
        Also synthesize the WAV of each continuum point (default).
    sample_rate : int
        Audio sample rate (default 44100 Hz).
    label : str, optional
        File stem of the continuum items (default: ``<canon stem>-vs-
        <assim stem>``).
    overwrite : bool
        Regenerate files that already exist (default: skip the WAV
        synthesis of existing files, like the kit experiment).

    Returns
    -------
    dict
        The metadata dictionary also written to ``morph_meta.json``.

    Raises
    ------
    MorphAlignmentError, MorphError
        See ``build_continuum``.
    """
    if lambdas is not None and n_steps is not None:
        raise MorphError('lambdas and n_steps are mutually exclusive')
    if lambdas is None:
        lambdas = parse_lambda_grid(
            n_steps=n_steps) if n_steps is not None else parse_lambda_grid()

    canon_data, sr_a = read_tract_sequence(canon_path)
    assim_data, sr_b = read_tract_sequence(assim_path)
    if sr_a != sr_b:
        raise MorphAlignmentError(
            f'tract sample rates differ: {sr_a} vs {sr_b} Hz')

    continuum = build_continuum(canon_data, assim_data, lambdas,
                                glottis_mode=glottis_mode,
                                glottis_tol=glottis_tol,
                                confine=confine, ramp=ramp)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if label is None:
        label = (f'{Path(canon_path).stem}-vs-'
                 f'{Path(assim_path).stem}')

    items_meta = []
    for it in continuum.items:
        stem = f'{label}_l{_format_lambda(it.lam)}'
        tract_path = out / f'{stem}.tract'
        write_morph_tract(tract_path, it.data, sr=sr_a)
        wav_path: Optional[str] = None
        if with_audio:
            from vtl_synth.core.synth import synthesize_audio
            wav_path = str(out / f'{stem}.wav')
            if overwrite or not os.path.exists(wav_path):
                synthesize_audio(
                    it.data[:, :N_TRACT_PARAMS],
                    it.data[:, N_TRACT_PARAMS:
                            N_TRACT_PARAMS + N_GLOTTIS_PARAMS],
                    output_wav=wav_path,
                    sample_rate=sample_rate)
        items_meta.append({
            'lam': it.lam,
            'tract': str(tract_path),
            'wav': wav_path,
            'n_frames': int(it.data.shape[0]),
        })
        logger.info('morph point %s (lambda=%g) -> %s',
                    stem, it.lam, tract_path)

    meta = {
        'patch': MORPH_PATCH_ID,
        'label': label,
        'sources': {
            'canon': str(canon_path),
            'assim': str(assim_path),
            'canon_sha256': _sha256_file(canon_path),
            'assim_sha256': _sha256_file(assim_path),
            'sample_rate_hz': sr_a,
        },
        'lambdas': [float(v) for v in lambdas],
        'glottis_mode': glottis_mode,
        'glottis_tol': glottis_tol,
        'confine': (None if continuum.confine is None else {
            'first_frame': int(continuum.confine[0]),
            'last_frame': int(continuum.confine[1]),
            'ramp_frames': int(continuum.ramp),
        }),
        'alignment': continuum.report.as_dict() if continuum.report
        else None,
        'checks': continuum.checks,
        'items': items_meta,
    }
    with open(out / 'morph_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    return meta


# ==========================================================================
# Target-level morph (v1.2.0) -- the DEFAULT semantic of the lambda
# variation: the tautosyllabic CC cluster C1C2 -> C1'C2, with lambda
# moving the POLAR TARGET of C1 (theta always, rho optionally) toward
# the target of C2.  C2 and the whole context stay bitwise canonical
# at every lambda; C.C splits (syllable dot) are out of scope.
# ==========================================================================

#: Engine nasal consonant keys -- the possible C1 of the cluster
#: (the lambda variation is the nasal-place assimilation continuum).
try:                                            # pragma: no cover
    from vtl_synth.core.constants import NASAL_CONSONANT_KEYS
    NASAL_C1_KEYS = tuple(sorted(NASAL_CONSONANT_KEYS))
except Exception:                               # pragma: no cover
    NASAL_C1_KEYS = ('J', 'N', 'm', 'n')

#: Default engine (co)articulation parameters -- the values of the
#: engine CLI / native Pipeline (vtl-synth run), so that a bare
#: ``morph --targets <phrase>`` reproduces ``vtl-synth run <phrase>``
#: frame-for-frame at lambda = 0.
TARGET_DEFAULT_ENGINE = dict(
    speaker='jd3',
    f0_hz=None,                 # -> f0_default of the speaker constants
    use_orthogonal=True,
    t_cons_ms=100.0,
    t_voy_ms=80.0,
    pause_short_ms=200.0,
    pause_long_ms=320.0,
)


def wrap_angle_delta(theta1: float, theta2: float) -> float:
    """Shortest signed angular difference ``theta2 - theta1`` (rad).

    The result lies in (-pi, pi]: the lambda interpolation of the C1
    theta follows the shortest angular path toward the C2 target.
    """
    d = (float(theta2) - float(theta1)) % (2.0 * np.pi)
    if d > np.pi:
        d -= 2.0 * np.pi
    return d


def interp_target_theta(theta1: float, theta2: float,
                        lam: float) -> float:
    """lambda-interpolated C1 theta (exact at the endpoints).

    ``theta(lam) = theta1 + lam * wrap_angle_delta(theta1, theta2)``
    for 0 < lam < 1 (shortest angular path); ``lam = 0`` returns
    ``theta1`` and ``lam = 1`` returns ``theta2`` VERBATIM (bitwise),
    which is the v1.2.0 semantic: at lambda = 1 the C1 polar target
    equals the C2 target coordinates exactly.
    """
    if not (0.0 <= lam <= 1.0):
        raise MorphError(f'lambda must lie in [0, 1], got {lam}')
    if lam <= 0.0:
        return float(theta1)
    if lam >= 1.0:
        return float(theta2)
    return float(theta1) + lam * wrap_angle_delta(theta1, theta2)


def interp_target_rho(rho1: float, rho2: float, lam: float) -> float:
    """lambda-interpolated C1 rho (exact at the endpoints).

    Linear interpolation; ``lam = 0`` returns ``rho1`` and
    ``lam = 1`` returns ``rho2`` verbatim (bitwise), mirroring
    :func:`interp_target_theta` for the optional rho channel.
    """
    if not (0.0 <= lam <= 1.0):
        raise MorphError(f'lambda must lie in [0, 1], got {lam}')
    if lam <= 0.0:
        return float(rho1)
    if lam >= 1.0:
        return float(rho2)
    return (1.0 - lam) * float(rho1) + lam * float(rho2)


def find_nasal_c1_clusters(nodes) -> List[Tuple[int, int]]:
    """Nasal-initial tautosyllabic CC clusters of a node list.

    A candidate is a pair of CONSECUTIVE consonant nodes
    ``(i, i+1)`` where node ``i`` is a nasal (engine key in
    :data:`NASAL_C1_KEYS`), both are flagged ``in_cluster`` by the
    engine (tautosyllabic cluster -- no syllable dot between them),
    and node ``i+1`` is a plain consonant (the C2 whose place the C1
    gesture assimilates to).  Onset clusters without a nasal C1
    (e.g. the /kl/ of ``klinpan``) are skipped by construction.
    """
    out = []
    for i in range(len(nodes) - 1):
        a, b = nodes[i], nodes[i + 1]
        if getattr(a, 'kind', None) != 'C' or \
                getattr(b, 'kind', None) != 'C':
            continue
        if not (getattr(a, 'in_cluster', False)
                and getattr(b, 'in_cluster', False)):
            continue
        if a.key in NASAL_C1_KEYS:
            out.append((i, i + 1))
    return out


@dataclass
class TargetClusterPlan:
    """Ground truth of one target-level morph (v1.2.0).

    Attributes
    ----------
    phrase : str
        The canonical cluster phrase (normalized engine input).
    cluster_index : int
        1-based rank of the retained nasal-initial CC cluster.
    c1_key, c2_key : str
        Engine phoneme keys of the cluster.
    c1_rho, c1_theta, c2_rho, c2_theta : float
        Endpoint polar targets (the .polar phoneme targets of the
        cluster branch; engine cluster rules already applied).
    theta_delta : float
        Shortest signed angular distance theta_C2 - theta_C1 (rad).
    rho_enabled : bool
        Whether lambda also interpolates the rho of C1.
    c1_window_steps : (int, int)
        C1 phoneme span in 100 Hz gesture steps (start inclusive,
        end exclusive) -- the reference C1 window of the study
        (e.g. 58-68 for the klinpan family -> frames 232-272 @400 Hz).
    c1_arcs_steps : (int, int)
        Span of the C1-INCIDENT sub-arcs (arrival arc into C1 +
        departure arc C1->C2): the maximal temporal footprint of the
        lambda variation on the trajectory.
    """
    phrase: str
    cluster_index: int
    c1_key: str
    c2_key: str
    c1_rho: float
    c1_theta: float
    c2_rho: float
    c2_theta: float
    theta_delta: float
    rho_enabled: bool
    c1_window_steps: Tuple[int, int]
    c1_arcs_steps: Tuple[int, int]

    def theta_at(self, lam: float) -> float:
        return interp_target_theta(self.c1_theta, self.c2_theta, lam)

    def rho_at(self, lam: float) -> float:
        if not self.rho_enabled:
            return self.c1_rho
        return interp_target_rho(self.c1_rho, self.c2_rho, lam)

    def as_dict(self) -> Dict:
        return {
            'cluster_index': self.cluster_index,
            'c1': {'key': self.c1_key, 'rho': self.c1_rho,
                   'theta': self.c1_theta},
            'c2': {'key': self.c2_key, 'rho': self.c2_rho,
                   'theta': self.c2_theta},
            'theta_delta': self.theta_delta,
            'theta_delta_deg': float(np.degrees(self.theta_delta)),
            'rho_enabled': self.rho_enabled,
            'c1_window_frames': {
                'first': 4 * self.c1_window_steps[0],
                'last': 4 * self.c1_window_steps[1] - 1},
            'c1_arcs_frames': {
                'first': 4 * self.c1_arcs_steps[0],
                'last': 4 * self.c1_arcs_steps[1] - 1},
        }


def _normalize_cluster_phrase(phrase: str) -> str:
    """Applies the engine input normalization of build_phrase_tract.

    syltraj word normalization (multi-char SAMPA keys, nasal vowels)
    followed by the engine IPA normalization, so that the cluster
    detection below sees exactly the nodes the synthesis will build.
    """
    from vtl_synth.core.syltraj import _normalize_word
    from vtl_synth.core.text_to_tract import _normalize_ipa
    ipa = _normalize_word(phrase.strip())
    return _normalize_ipa(ipa)


def plan_target_cluster(phrase: str,
                        cluster_index: int = 1,
                        rho_enabled: bool = False,
                        engine: Optional[Dict] = None) -> \
        Tuple[TargetClusterPlan, List, Dict]:
    """Locates the C1C2 cluster of a phrase and records its targets.

    Returns ``(plan, nodes, last_build)`` where ``nodes`` are the
    engine syllabic nodes (with the cluster target rules applied) and
    ``last_build`` the timing state (block_info / anchors) of the
    trajectory walk, derived with the SAME gesture steps as the
    synthesis will use (``engine`` overrides of
    :data:`TARGET_DEFAULT_ENGINE` are honoured, so the recorded C1
    windows match the produced frames whatever the parameters).

    Raises
    ------
    MorphError
        When the phrase carries a syllable dot (C.C splits are out of
        scope of the lambda variation), or when no nasal-initial CC
        cluster exists (or ``cluster_index`` exceeds the count).
    """
    import vtl_synth.core.syltraj as syltraj

    eng = dict(TARGET_DEFAULT_ENGINE)
    if engine:
        unknown = set(engine) - set(eng)
        if unknown:
            raise MorphError(
                f'unknown engine parameters: {sorted(unknown)} '
                f'(expected any of {sorted(eng)})')
        eng.update(engine)

    ipa = _normalize_cluster_phrase(phrase)
    if '.' in ipa:
        raise MorphError(
            f'the lambda variation is defined on tautosyllabic CC '
            f'clusters only: {phrase!r} contains a syllable separator '
            f'(".") -- a C.C split is a different synthesis (boundary '
            f'split, no pause) and is NOT the object of the lambda '
            f'variation; use the cluster form (e.g. klinpan, not '
            f'klin.pan)')
    flat, syllables, syl_bounds, boundary_info, word_starts = \
        syltraj.build_flat(ipa)
    nodes = syltraj.build_nodes(flat, syllables, syl_bounds,
                                boundary_info, word_starts)
    candidates = find_nasal_c1_clusters(nodes)
    if not candidates:
        raise MorphError(
            f'no tautosyllabic CC cluster with a nasal C1 found in '
            f'{phrase!r} (phonemes: '
            + ','.join(nd.key for nd in nodes if nd.kind in ('V', 'C'))
            + ') -- the lambda variation needs a nasal coda followed '
              'by an onset consonant of the same cluster (dotless '
              'form, e.g. klinpan)')
    if not (1 <= cluster_index <= len(candidates)):
        raise MorphError(
            f'cluster index must be in [1, {len(candidates)}] for '
            f'{phrase!r} (found clusters: '
            + '; '.join(f'{nodes[i].key}{nodes[j].key}'
                        for i, j in candidates) + ')')
    i, j = candidates[cluster_index - 1]
    c1, c2 = nodes[i], nodes[j]

    anchors = syltraj.build_anchors(nodes, word_starts, boundary_info)
    # block timing with the SAME gesture steps as the synthesis
    # (build_target_continuum conversions, mirroring Pipeline.run)
    Pval, block_info = syltraj.build_global_pval(
        nodes, anchors,
        max(1, round(eng['t_cons_ms'] / 10.0)),
        max(1, round(eng['t_voy_ms'] / 10.0)),
        max(1, round(eng['pause_short_ms'] / 10.0)),
        max(1, round(eng['pause_long_ms'] / 10.0)))

    # position of C1 among the consonant nodes, then locate its
    # cluster block in block_info (cons_keys are in node order)
    cons_order = [k for k, nd in enumerate(nodes) if nd.kind == 'C']
    pos = cons_order.index(i)
    window = None
    arcs = None
    off = 0
    running = 0
    for b in block_info:
        if b.kind == 'cluster':
            keys = list(getattr(b, 'cons_keys', []) or [])
            if keys and running <= pos < running + len(keys):
                jj = pos - running
                T = b.n_steps // (len(keys) + 1)
                t0 = off + jj * T
                window = (t0, t0 + T)
                arcs = (t0, t0 + 2 * T)      # C1 is followed by C2
                break
            running += len(keys)
        off += b.n_steps
    if window is None:                         # pragma: no cover
        raise MorphError(
            f'C1 {c1.key!r} of {phrase!r} not found in the engine '
            f'block structure -- engine layout change, please report')

    plan = TargetClusterPlan(
        phrase=ipa,
        cluster_index=cluster_index,
        c1_key=c1.key, c2_key=c2.key,
        c1_rho=float(c1.rho), c1_theta=float(c1.theta),
        c2_rho=float(c2.rho), c2_theta=float(c2.theta),
        theta_delta=wrap_angle_delta(float(c1.theta), float(c2.theta)),
        rho_enabled=bool(rho_enabled),
        c1_window_steps=(int(window[0]), int(window[1])),
        c1_arcs_steps=(int(arcs[0]), int(arcs[1])),
    )
    return plan, nodes, {'block_info': block_info, 'anchors': anchors}


@contextlib.contextmanager
def _retargeted_c1(node_index: int, theta: float,
                   rho: Optional[float] = None):
    """Runtime override of ONE consonant polar target (v1.2.0).

    Around a synthesis made by the caller, ``syltraj.build_nodes`` is
    wrapped so that the node at ``node_index`` (the C1 of the
    cluster) carries the given ``(theta, rho)`` instead of its
    canonical target.  The wrapper delegates to the original builder
    first (same cluster rules, same selector resolution) and then
    overwrites the two target floats, so nothing else can drift.
    The engine module attribute is ALWAYS restored (additive-only
    contract: no engine file is modified, no global state survives).
    """
    import vtl_synth.core.syltraj as syltraj
    original = syltraj.build_nodes

    def build_nodes_retargeted(flat, syllables, syl_bounds, boundary_info,
                               word_starts):
        nodes = original(flat, syllables, syl_bounds, boundary_info,
                         word_starts)
        nd = nodes[node_index]
        nd.theta = theta
        if rho is not None:
            nd.rho = rho
        return nodes

    syltraj.build_nodes = build_nodes_retargeted
    try:
        yield
    finally:
        syltraj.build_nodes = original


def _build_cluster_tract(phrase: str, node_index: int,
                         theta: float, rho: Optional[float],
                         engine_kw: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """One (re)synthesis of the cluster phrase with C1 retargeted.

    Runs the engine's own ``build_phrase_tract`` chain verbatim under
    :func:`_retargeted_c1` (identity-based glottis/envelope/orthogonal
    branches are unaffected by the target override; only the C1-incident
    sub-arcs of the tract block change).  Returns
    ``(tract400 (T, 19), glottis400 (T, 11))``.
    """
    from vtl_synth.core.build_phrase_tract import build_phrase_tract
    with _retargeted_c1(node_index, theta, rho):
        tract400, glott400 = build_phrase_tract(
            phrase, phrase_id=1, label='endpoint', verbose=False,
            return_data=True, **engine_kw)
    return tract400, glott400


def build_target_continuum(phrase: str,
                           lambdas: Optional[Sequence[float]] = None,
                           rho_enabled: bool = False,
                           cluster_index: int = 1,
                           engine: Optional[Dict] = None) -> Dict:
    """Builds the C1C2 -> C1'C2 target continuum of a cluster phrase.

    For every lambda of the grid the C1 polar target is set to
    ``theta_C1(lambda)`` (toward the C2 theta; rho too when
    ``rho_enabled``) and the WHOLE engine trajectory is rebuilt and
    re-synthesized, so each item is a genuine cluster utterance whose
    C1 gesture points at the lambda-interpolated place.  Locality is
    structural: only the sub-arcs incident to C1 depend on its
    target, hence every frame outside them stays bitwise canonical
    (C2 and context strictly invariant), and the glottis block is
    bitwise constant along the continuum (voicing, F0 and nasality
    envelopes are identity-based, not target-based).

    Parameters
    ----------
    phrase : str
        Canonical cluster phrase in engine SAMPA (dotless: a "."
        syllable separator is rejected -- C.C splits are out of
        scope), e.g. ``"klinpan"``.
    lambdas : sequence of float, optional
        Grid in [0, 1] (default: 9 uniform steps).
    rho_enabled : bool
        Also interpolate the C1 rho toward the C2 rho (default
        False: theta only -- the place channel).
    cluster_index : int
        1-based rank of the nasal-initial CC cluster when the phrase
        carries several (default 1: the first).
    engine : dict, optional
        Overrides of the engine (co)articulation parameters, see
        :data:`TARGET_DEFAULT_ENGINE` (speaker, f0_hz,
        use_orthogonal, t_cons_ms, t_voy_ms, pause_short_ms,
        pause_long_ms).  Defaults = the engine CLI values, so
        lambda = 0 reproduces ``vtl-synth run <phrase>`` bitwise.

    Returns
    -------
    dict
        ``{'plan': TargetClusterPlan, 'engine': {...}, 'lambdas': [...],
        'items': [{'lam', 'theta_c1', 'rho_c1', 'tract', 'glottis'}],
        'canon': {'tract', 'glottis'}, 'checks': {...},
        'diff_extent_frames': {...}, 'n_frames': int}``

    Raises
    ------
    MorphError
        On a dotted phrase, a missing cluster, an out-of-range
        lambda/grid, or a failed guarantee check.
    """
    eng = dict(TARGET_DEFAULT_ENGINE)
    if engine:
        unknown = set(engine) - set(eng)
        if unknown:
            raise MorphError(
                f'unknown engine parameters: {sorted(unknown)} '
                f'(expected any of {sorted(eng)})')
        eng.update(engine)
    # Pipeline.run conversions: ms -> 100 Hz gesture steps
    engine_kw = dict(
        f0_hz=eng['f0_hz'],
        speaker=eng['speaker'],
        use_orthogonal=eng['use_orthogonal'],
        engine='syl',
        t_cons=max(1, round(eng['t_cons_ms'] / 10.0)),
        t_voy=max(1, round(eng['t_voy_ms'] / 10.0)),
        pause_short_ms=eng['pause_short_ms'],
        pause_long_ms=eng['pause_long_ms'],
        sampa_guard=None,
    )

    if lambdas is None:
        lambdas = parse_lambda_grid()
    else:
        lambdas = parse_lambda_grid(','.join(f'{float(v):g}'
                                             for v in lambdas))

    plan, nodes, _ = plan_target_cluster(
        phrase, cluster_index=cluster_index, rho_enabled=rho_enabled,
        engine=engine)
    i_c1, _ = find_nasal_c1_clusters(nodes)[cluster_index - 1]

    # canonical cluster (lambda = 0 target, bitwise) via the same
    # chain: C1 retargeted to its OWN canonical target floats
    canon_tract, canon_glott = _build_cluster_tract(
        plan.phrase, i_c1, plan.c1_theta,
        plan.c1_rho if rho_enabled else None, engine_kw)

    items = []
    for lam in lambdas:
        theta = plan.theta_at(lam)
        rho = plan.rho_at(lam)
        tract, glott = _build_cluster_tract(
            plan.phrase, i_c1, theta,
            rho if rho_enabled else None, engine_kw)
        data = np.hstack([tract, glott])
        items.append({'lam': lam, 'theta_c1': theta, 'rho_c1': rho,
                      'tract': tract, 'glottis': glott, 'data': data})
    canon_data = np.hstack([canon_tract, canon_glott])

    # ---- guarantees (asserted, then recorded) -------------------------
    n_frames = canon_data.shape[0]
    for it in items:
        if it['data'].shape != canon_data.shape:
            raise MorphError(
                f"lambda {it['lam']:g} produced {it['data'].shape} "
                f"vs canonical {canon_data.shape} -- the target morph "
                f"must keep the segmental skeleton (temporal identity)")
    checks: Dict[str, bool] = {}
    zero = next((it for it in items if it['lam'] == 0.0), None)
    checks['lam0_bitwise_canon'] = (
        True if zero is None
        else bool(np.array_equal(zero['data'], canon_data)))
    checks['glottis_bitwise_constant'] = bool(all(
        np.array_equal(it['glottis'], canon_glott) for it in items))
    one = next((it for it in items if it['lam'] == 1.0), None)
    if one is not None:
        checks['lam1_theta_equals_c2'] = (
            one['theta_c1'] == plan.c2_theta)
        checks['lam1_theta_equiv_c2_mod_2pi'] = abs(
            wrap_angle_delta(plan.c2_theta, one['theta_c1'])) < 1e-12
        if rho_enabled:
            checks['lam1_rho_equals_c2'] = (one['rho_c1'] == plan.c2_rho)
    else:
        # structural: theta_at(1.0) IS c2_theta verbatim by construction
        checks['lam1_theta_equals_c2'] = (
            plan.theta_at(1.0) == plan.c2_theta)
        if rho_enabled:
            checks['lam1_rho_equals_c2'] = (
                plan.rho_at(1.0) == plan.c2_rho)

    # measured footprint of the variation + structural locality check.
    # The C1-incident sub-arcs bound the 100 Hz samples that depend on
    # the C1 target; pad one gesture sample (4 frames @400 Hz) for the
    # 100->400 Hz upsampling, and require bitwise canonicality outside.
    a0 = max(0, 4 * plan.c1_arcs_steps[0] - 4)
    a1 = min(n_frames, 4 * plan.c1_arcs_steps[1] + 4)
    outside = np.ones(n_frames, dtype=bool)
    outside[a0:a1] = False
    checks['outside_c1_arcs_bitwise_canon'] = bool(all(
        np.array_equal(it['data'][outside], canon_data[outside])
        for it in items))
    diff_any = np.zeros(n_frames, dtype=bool)
    for it in items:
        diff_any |= np.any(it['data'] != canon_data, axis=1)
    df = np.where(diff_any)[0]
    diff_extent = {
        'first_frame': int(df.min()) if len(df) else None,
        'last_frame': int(df.max()) if len(df) else None,
    }
    if not all(checks.values()):
        raise MorphError(
            f'target-morph guarantee checks failed (impossible by '
            f'construction -- please report): {checks}')

    return {
        'plan': plan,
        'engine': eng,
        'lambdas': [float(v) for v in lambdas],
        'canon': {'tract': canon_tract, 'glottis': canon_glott},
        'items': items,
        'checks': checks,
        'diff_extent_frames': diff_extent,
        'n_frames': int(n_frames),
    }


def morph_targets(phrase: str,
                  output_dir: Union[str, Path],
                  lambdas: Optional[Sequence[float]] = None,
                  rho_enabled: bool = False,
                  cluster_index: int = 1,
                  engine: Optional[Dict] = None,
                  with_audio: bool = True,
                  sample_rate: int = 44100,
                  label: Optional[str] = None,
                  overwrite: bool = False) -> Dict:
    """Writes the target-level lambda continuum of a cluster phrase.

    The DEFAULT semantic of the lambda variation (v1.2.0): for each
    lambda, ``<label>_l<lam>.tract`` holds the re-synthesized cluster
    C1'C2 whose C1 polar target is the lambda-interpolation of the
    canonical C1 target toward the C2 target (theta always, rho too
    when ``rho_enabled``); ``<label>_l<lam>.wav`` is its audio, and
    ``morph_meta.json`` records the C1/C2 polar targets, the
    lambda-interpolated target of every item, the C1 windows and the
    guarantee checks.

    Parameters
    ----------
    phrase : str
        Canonical cluster phrase, engine SAMPA, dotless (e.g.
        ``"klinpan"``; ``"klin.pan"`` is rejected -- C.C splits are
        out of scope of the lambda variation).
    output_dir : str or Path
        Destination directory (created).
    lambdas, rho_enabled, cluster_index, engine :
        See :func:`build_target_continuum`.
    with_audio : bool
        Also synthesize the WAV of each continuum point (default).
    sample_rate : int
        Audio sample rate (default 44100 Hz).
    label : str, optional
        File stem of the continuum items (default: the phrase).
    overwrite : bool
        Regenerate WAV files that already exist (default: skip).

    Returns
    -------
    dict
        The metadata dictionary also written to ``morph_meta.json``.
    """
    cont = build_target_continuum(phrase, lambdas=lambdas,
                                  rho_enabled=rho_enabled,
                                  cluster_index=cluster_index,
                                  engine=engine)
    plan = cont['plan']

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if label is None:
        label = plan.phrase

    items_meta = []
    for it in cont['items']:
        stem = f'{label}_l{_format_lambda(it["lam"])}'
        tract_path = out / f'{stem}.tract'
        data = it['data']
        write_morph_tract(tract_path, data, sr=float(TRACT_SR))
        wav_path: Optional[str] = None
        if with_audio:
            from vtl_synth.core.synth import synthesize_audio
            wav_path = str(out / f'{stem}.wav')
            if overwrite or not os.path.exists(wav_path):
                synthesize_audio(
                    data[:, :N_TRACT_PARAMS],
                    data[:, N_TRACT_PARAMS:N_TRACT_PARAMS
                         + N_GLOTTIS_PARAMS],
                    output_wav=wav_path,
                    sample_rate=sample_rate)
        items_meta.append({
            'lam': it['lam'],
            'theta_c1': it['theta_c1'],
            'rho_c1': it['rho_c1'],
            'tract': str(tract_path),
            'wav': wav_path,
            'n_frames': int(data.shape[0]),
        })
        logger.info('target-morph point %s (lambda=%g, C1 theta=%.6f '
                    'rad) -> %s', stem, it['lam'], it['theta_c1'],
                    tract_path)

    meta = {
        'patch': MORPH_PATCH_ID,
        'mode': 'targets',
        'semantic': ('C1C2 -> C1\'C2 on the tautosyllabic cluster: '
                     'lambda moves the C1 polar target (theta'
                     + (', rho' if rho_enabled else '')
                     + ') toward the C2 target; C2 and context stay '
                       'bitwise canonical'),
        'label': label,
        'phrase': plan.phrase,
        'sources': {
            'canon_phrase': plan.phrase,
            'endpoint_synth': str(out / '_engine'),
            'sample_rate_hz': float(TRACT_SR),
        },
        'targets': plan.as_dict(),
        'engine': cont['engine'],
        'lambdas': cont['lambdas'],
        'n_frames': cont['n_frames'],
        'diff_extent_frames': cont['diff_extent_frames'],
        'checks': cont['checks'],
        'items': items_meta,
    }
    with open(out / 'morph_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    return meta
