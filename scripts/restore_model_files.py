"""Restore large backbone files from Git LFS parts using only the standard library."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'models/final_diffusion/backbone/parts_manifest.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def restore():
    rows = json.loads(MANIFEST.read_text())['files']
    for row in rows:
        target = ROOT / row['path']
        if target.exists() and target.stat().st_size == row['size'] and digest(target) == row['sha256']:
            print(f'OK: {row["path"]}', flush=True)
            continue
        # Validate all parts before writing. An LFS pointer is never a valid weight part.
        for part in row['parts']:
            p = ROOT / part['path']
            if not p.exists() or p.stat().st_size != part['size'] or digest(p) != part['sha256']:
                raise RuntimeError(f'Missing/corrupt LFS part: {part["path"]}. Run git lfs pull first.')
        temporary = target.with_name(target.name + '.restoring')
        h = hashlib.sha256()
        with temporary.open('wb') as out:
            for part in row['parts']:
                with (ROOT / part['path']).open('rb') as stream:
                    for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        out.write(data)
                        h.update(data)
        if temporary.stat().st_size != row['size'] or h.hexdigest() != row['sha256']:
            raise RuntimeError(f'Restored file checksum mismatch: {row["path"]}')
        os.replace(temporary, target)
        print(f'Restored: {row["path"]}', flush=True)


def verify():
    manifest = json.loads((ROOT / 'models/final_diffusion/download_manifest.json').read_text())
    for row in manifest['files']:
        p = ROOT / 'models/final_diffusion' / row['path']
        if not p.exists() or p.stat().st_size != row['size'] or digest(p) != row['sha256']:
            raise RuntimeError(f'Original model file missing/corrupt: {p.relative_to(ROOT)}')
    print(f'All {len(manifest["files"])} original model files verified.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Also verify all 46 original model files')
    args = parser.parse_args()
    restore()
    if args.verify:
        verify()
