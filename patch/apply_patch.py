#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 covtl-lambda-patch contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Apply the covtl-lambda-patch (v1.2.0) to a covtl-pipeline checkout.

In the spirit of the engine's own patches (cf. the 2026-10-18 /N/
closure patch, ``patch_N.py``):

  * **additive**: installs one new engine module
    (``vtl_synth/core/morphing.py``), one new CLI module
    (``vtl_synth/cli/morph_cmd.py``) and one new test file
    (``tests/test_morphing.py``); no existing file other than
    ``vtl_synth/cli/main.py`` is touched;
  * **assert-guarded**: the single insertion anchor in ``main.py``
    must be found exactly once, otherwise nothing is written;
  * **EOL-safe**: the insertion inherits the target file's line
    endings (the shipped ``main.py`` uses CRLF);
  * **reversible**: ``main.py`` is backed up to
    ``main.py.lambda-morph.bak``; ``revert_patch.py`` restores it
    and removes the three added files;
  * **idempotent**: re-running detects the already-applied state.

Usage::

    python patch/apply_patch.py [--pipeline-root DIR] [--force]

Exit codes: 0 applied, 2 already applied (use --force to re-copy the
module files), 1 error.
"""
import argparse
import os
import py_compile
import shutil
import subprocess
import sys

PATCH_ID = 'covtl-lambda-patch v1.2.0 (2026-10-26)'
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, 'files')

# (source in patch/files/, destination relative to the pipeline root)
NEW_FILES = [
    ('core_morphing.py', os.path.join('vtl_synth', 'core', 'morphing.py')),
    ('cli_morph_cmd.py', os.path.join('vtl_synth', 'cli', 'morph_cmd.py')),
    ('test_morphing.py', os.path.join('tests', 'test_morphing.py')),
]

CLI_MAIN = os.path.join('vtl_synth', 'cli', 'main.py')
BACKUP_SUFFIX = '.lambda-morph.bak'

# Unique anchor of build_parser() in cli/main.py (v1.1.0 / v1.2.0 line
# layouts): the global-options block that closes the subcommand list.
ANCHOR = ('    # ---- global options (before the subcommand on the '
          'command line) ----')

INSERTION = [
    '    # ---- morph (lambda-morph option, covtl-lambda-patch) '
    '---------------',
    '    # 2026-10-22: articulatory interpolation continuum option;',
    '    # 2026-10-26: v1.2.0 semantic -- the lambda variation acts on',
    '    # the tautosyllabic CC cluster C1C2 -> C1\'C2: --targets moves',
    '    # the C1 polar target (theta, optionally rho) toward the C2',
    '    # target; the two-endpoint tract blend remains available.',
    '    # Additive option -- no existing subcommand is modified.',
    '    # Implementation: vtl_synth/core/morphing.py',
    '    # + vtl_synth/cli/morph_cmd.py; tests: tests/test_morphing.py.',
    '    # Apply/revert: patch/apply_patch.py, patch/revert_patch.py.',
    '    from vtl_synth.cli.morph_cmd import add_morph_subparser',
    '    add_morph_subparser(sub, _add_engine_options)',
    '',
]

# version-agnostic marker of an applied insertion (matches the v1.0.0
# and v1.1.0 block headers alike, so an upgrade apply never inserts
# the block twice)
MORPH_MARKER = 'lambda-morph option, covtl-lambda-patch'


# --------------------------------------------------------------------------
# locate the pipeline root
# --------------------------------------------------------------------------

def find_pipeline_root(explicit=None):
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get('COVTL_PIPELINE_ROOT'):
        cands.append(os.environ['COVTL_PIPELINE_ROOT'])
    cands.append(os.path.join(os.getcwd(), 'covtl-pipeline'))
    # walk up from the patch directory (repo checked out *inside* the
    # pipeline? or beside it)
    d = HERE
    for _ in range(4):
        cands.append(d)
        cands.append(os.path.join(d, 'covtl-pipeline'))
        d = os.path.dirname(d)
    for cand in cands:
        if cand and os.path.isfile(os.path.join(cand, CLI_MAIN)):
            return os.path.abspath(cand)
    return None


def main():
    ap = argparse.ArgumentParser(
        description=f'Apply the {PATCH_ID} to a covtl-pipeline checkout.')
    ap.add_argument('--pipeline-root', default=None, metavar='DIR',
                    help='covtl-pipeline checkout (default: $COVTL_'
                         'PIPELINE_ROOT, ./covtl-pipeline, or auto-'
                         'search from this script)')
    ap.add_argument('--force', action='store_true',
                    help='re-copy the module files even if already '
                         'present (the main.py insertion stays guarded)')
    args = ap.parse_args()

    root = find_pipeline_root(args.pipeline_root)
    if root is None:
        print('ERROR: no covtl-pipeline checkout found (looked for '
              f'{CLI_MAIN}); pass --pipeline-root DIR')
        return 1
    print(f'[apply] pipeline root : {root}')

    main_path = os.path.join(root, CLI_MAIN)
    morphing_dst = os.path.join(root, NEW_FILES[0][1])

    already = os.path.isfile(morphing_dst)
    if already and not args.force:
        print('[apply] vtl_synth/core/morphing.py already present -- '
              'patch seems applied; use --force to re-copy the modules')
        return 2

    # ---- 1. new module files --------------------------------------------
    for src, rel in NEW_FILES:
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(FILES, src), dst)
        print(f'[apply] + {rel}')
        py_compile.compile(dst, doraise=True)

    # ---- 2. CLI insertion (assert-guarded, EOL-safe) ---------------------
    with open(main_path, 'r', encoding='utf-8', newline='') as f:
        text = f.read()

    if MORPH_MARKER in text:
        print('[apply] cli/main.py already carries the morph insertion')
    else:
        n = text.count(ANCHOR)
        if n != 1:
            print(f'ERROR: expected exactly 1 anchor in {CLI_MAIN}, '
                  f'found {n} -- aborting (nothing else was written to '
                  f'main.py; remove the copied files or keep them, and '
                  f'report the layout)')
            return 1
        eol = '\r\n' if '\r\n' in text else '\n'
        # eol.join over INSERTION ends with eol (the list ends with '')
        block = eol.join(INSERTION)
        if text.count(ANCHOR + eol) == 1:
            text = text.replace(ANCHOR + eol, block + ANCHOR + eol, 1)
        else:
            # anchor present but not followed by the dominant EOL
            text = text.replace(ANCHOR, block + ANCHOR, 1)
        backup = main_path + BACKUP_SUFFIX
        shutil.copyfile(main_path, backup)
        with open(main_path, 'w', encoding='utf-8', newline='') as f:
            f.write(text)
        eol_name = 'CRLF' if eol == '\r\n' else 'LF'
        print(f'[apply] ~ {CLI_MAIN} (4-line insertion before the '
              f'global-options block; EOL={eol_name})')
        print(f'[apply] backup      : {CLI_MAIN}{BACKUP_SUFFIX}')

    # ---- 3. verification --------------------------------------------------
    py_compile.compile(main_path, doraise=True)
    try:
        probe = subprocess.run(
            [sys.executable, '-c',
             'from vtl_synth.cli.main import build_parser as b; '
             'ap = b(); subs = [a for a in ap._actions '
             'if isinstance(a, __import__("argparse")._SubParsersAction)]; '
             'assert any("morph" in s.choices for s in subs), '
             '"morph subcommand missing"; print("morph subcommand OK")'],
            cwd=root, capture_output=True, text=True, timeout=120)
        if probe.returncode == 0:
            print(f'[apply] verify      : {probe.stdout.strip()}')
        else:
            print('[apply] verify      : import probe skipped/failed '
                  '(engine dependencies?) -- py_compile only\n'
                  f'         {probe.stderr.strip()[:300]}')
    except Exception as e:                                    # pragma: no cover
        print(f'[apply] verify      : import probe not run ({e}) -- '
              'py_compile only')

    print(f'[apply] done -- {PATCH_ID}')
    print('[apply] next: '
          f'cd {root} && python -m pytest tests/test_morphing.py -q')
    print('[apply] try : vtl-synth morph --phonetic --targets klinpan '
          '-o morph_out --label P1')
    return 0


if __name__ == '__main__':
    sys.exit(main())
