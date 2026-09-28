"""Check tracked source structure without importing or executing project code."""
from pathlib import Path
import ast
import json
import re
import subprocess
import sys
from urllib.parse import unquote

root = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
tracked = {Path(p) for p in paths if p}
errors = []
counts = {'python': 0, 'notebooks': 0, 'json': 0, 'links': 0}
for rel in sorted(tracked):
    p = root / rel
    if not p.is_file():
        errors.append(f'{rel}: tracked file missing')
        continue
    if p.stat().st_size >= 25 * 1024 * 1024:
        errors.append(f'{rel}: large asset should be distributed separately')
    if p.suffix in {'.keras', '.h5', '.hdf5', '.pickle', '.pkl', '.db', '.pt', '.bin', '.safetensors'} or p.name.startswith('.env'):
        errors.append(f'{rel}: local-only artifact is tracked')
    if p.suffix in {'.docx', '.mp4', '.mkv', '.mov'}:
        errors.append(f'{rel}: report or recording is tracked; see reports/README.md')
    try:
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(rel))
            counts['python'] += 1
        elif p.suffix == '.ipynb':
            data = json.loads(p.read_text())
            assert data.get('nbformat') == 4, 'Expected notebook format 4'
            assert isinstance(data.get('cells'), list), 'Missing cells'
            widgets = data.get('metadata', {}).get('widgets', {})
            for state in widgets.values():
                assert 'state' in state, 'Widget metadata without a state key will not render on GitHub'
            for cell in data['cells']:
                assert cell.get('cell_type') in {'markdown', 'code', 'raw'}, 'Invalid cell type'
                assert isinstance(cell.get('source'), (str, list)), 'Missing cell source'
                if cell['cell_type'] == 'code':
                    assert isinstance(cell.get('outputs'), list), 'Missing stored output list'
            counts['notebooks'] += 1
        elif p.suffix == '.json':
            json.loads(p.read_text())
            counts['json'] += 1
    except (SyntaxError, ValueError, AssertionError) as exc:
        errors.append(f'{rel}: {exc}')
    if p.name in {'README.md', 'CONTRIBUTING.md'} or rel.parts[0] == 'docs':
        if p.suffix != '.md':
            continue
        for ref in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            if ref.startswith(('https:', 'http:', 'mailto:', '#')):
                continue
            target = (p.parent / unquote(ref.split('#')[0].strip('<>'))).resolve()
            try:
                target_rel = target.relative_to(root)
            except ValueError:
                errors.append(f'{rel}: link leaves repository: {ref}')
                continue
            present = target_rel in tracked or any(target_rel in item.parents for item in tracked)
            if not present:
                errors.append(f'{rel}: link absent from tracked files: {ref}')
            counts['links'] += 1
if errors:
    print('\n'.join(errors))
    sys.exit(1)
print(json.dumps({'tracked_files': len(tracked), **counts, 'result': 'passed'}, indent=2))
print('Static checks only: no model download, inference, Gradio launch or notebook execution.')
