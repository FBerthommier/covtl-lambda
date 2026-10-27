#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 covtl-lambda-patch contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Revert the covtl-lambda-patch (v1.0.0) from a covtl-pipeline checkout.

Two strategies, tried in order:

  1. **backup restore** -- if ``vtl_synth/cli/main.py.lambda-morph.bak``
     exists (written by ``apply_patch.py``), copy it back;
  2. **marker removal** -- otherwise, remove the insertion block by its
     comment markers (``covtl-lambda-patch v1.0.0`` ... 
     ``add_morph_subparser(sub, _add_engine_options)``), EOL-agnostic.

Then the three added files are removed:
``vtl_synth/core/morphing.py``, ``vtl_synth/cli/morph_cmd.py``,
``tests/test_morphing.py``.  Nothing else is touched -- engine patches
applied by other means (e.g. the /N/ closure patch in ``constants.py``)
are preserved.

Usage::

    python patch/revert_patch.py [--pipeline-root DIR] [--keep-modules]

Exit codes: 0 reverted (or nothing to do), 1 error.
"""
import argparse
import os
import py_compile
import re
import shutil
import sys

PATCH_TAG = 'lambda-morph option, covtl-lambda-patch'   # version-agnostic
HERE = os.path.dirname(os.path.abspath(__file__))

CLI_MAIN = os.path.join('vtl_synth', 'cli', 'main.py')
BACKUP_SUFFIX = '.lambda-morph.bak'
ADDED_FILES = [
    os.path.join('vtl_synth', 'core', 'morphing.py'),
    os.path.join('vtl_synth', 'cli', 'morph_cmd.py'),
    os.path.join('tests', 'test_morphing.py'),
]


def find_pipeline_root(explicit=None):
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get('COVTL_PIPELINE_ROOT'):
        cands.append(os.environ['COVTL_PIPELINE_ROOT'])
    cands.append(os.path.join(os.getcwd(), 'covtl-pipeline'))
    d = HERE
    for _ in range(4):
        cands.append(d)
        cands.append(os.path.join(d, 'covtl-pipeline'))
        d = os.path.dirname(d)
    for cand in cands:
        if cand and os.path.isfile(os.path.join(cand, CLI_MAIN)):
            return os.path.abspath(cand)
    return None


# EOL-agnostic block pattern: from the "# ---- morph" marker line up to
# the add_morph_subparser call line, both inclusive (version-agnostic:
# matches v1.0.0 and later block headers).
BLOCK_RE = re.compile(
    r'[ \t]*# ---- morph \(lambda-morph option, covtl-lambda-patch'
    r'[^\r\n]*'
    r'(?:\r?\n[^\r\n]*)*?'
    r'\r?\n[ \t]*add_morph_subparser\(sub, _add_engine_options\)'
    r'\r?\n?')


def revert_main_py(main_path):
    """Returns True if main.py was modified."""
    backup = main_path + BACKUP_SUFFIX
    with open(main_path, 'r', encoding='utf-8', newline='') as f:
        text = f.read()
    if PATCH_TAG not in text:
        print(f'[revert] {CLI_MAIN}: no insertion found (already clean)')
        return False

    if os.path.isfile(backup):
        shutil.copyfile(backup, main_path)
        os.remove(backup)
        print(f'[revert] ~ {CLI_MAIN}: restored from '
              f'{os.path.basename(backup)}')
        return True

    new_text, n = BLOCK_RE.subn('', text, count=1)
    if n != 1:
        print(f'ERROR: could not locate the insertion block in '
              f'{CLI_MAIN} and no backup exists -- restore it manually '
              f'from version control')
        return False
    with open(main_path, 'w', encoding='utf-8', newline='') as f:
        f.write(new_text)
    print(f'[revert] ~ {CLI_MAIN}: insertion block removed by markers')
    return True


def main():
    ap = argparse.ArgumentParser(
        description='Revert the covtl-lambda-patch from a covtl-pipeline '
                    'checkout.')
    ap.add_argument('--pipeline-root', default=None, metavar='DIR')
    ap.add_argument('--keep-modules', action='store_true',
                    help='only undo the cli/main.py insertion; keep '
                         'morphing.py / morph_cmd.py / test_morphing.py '
                         '(e.g. to keep using the Python API)')
    args = ap.parse_args()

    root = find_pipeline_root(args.pipeline_root)
    if root is None:
        print('ERROR: no covtl-pipeline checkout found; pass '
              '--pipeline-root DIR')
        return 1
    print(f'[revert] pipeline root : {root}')

    main_path = os.path.join(root, CLI_MAIN)
    modified = revert_main_py(main_path)
    py_compile.compile(main_path, doraise=True)
    print(f'[revert] py_compile    : {CLI_MAIN} OK')

    if not args.keep_modules:
        for rel in ADDED_FILES:
            path = os.path.join(root, rel)
            if os.path.isfile(path):
                os.remove(path)
                print(f'[revert] - {rel}')
            else:
                print(f'[revert] - {rel} (absent)')

    if not modified and not any(os.path.isfile(
            os.path.join(root, rel)) for rel in ADDED_FILES):
        print('[revert] nothing to do -- the patch was not applied')
    print('[revert] done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
