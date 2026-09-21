"""Command-line compiler with automatic static/general lowering selection."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

from .compiler import CompileError, compile_source


def main(argv=None):
    p = argparse.ArgumentParser(description='Compile Remora directly to a standalone PyTorch module')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('-f', '--file', type=Path)
    source.add_argument('-e', '--expression')
    p.add_argument('--remora', default=os.environ.get('REMORA', 'remora'), help='Path to the existing Remora frontend binary')
    p.add_argument('-o', '--output', type=Path, help='Output module; stdout if omitted')
    p.add_argument('--timeout', type=float, default=120, help='Frontend timeout in seconds')
    p.add_argument('--lowering', choices=('auto', 'native', 'tensor', 'static', 'general'), default='auto', help='Native library ops, structural tensor ops, static tensors, general checked-AST lowering, or automatic selection')
    a = p.parse_args(argv)
    try:
        code = compile_source(source=a.file, expression=a.expression, remora=a.remora, timeout=a.timeout, lowering=a.lowering)
        if a.output:
            # Never leave a half-written artifact on errors.
            with tempfile.NamedTemporaryFile('w', dir=a.output.parent, delete=False) as f:
                temporary = Path(f.name)
                f.write(code)
            try:
                temporary.replace(a.output)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            sys.stdout.write(code)
    except (CompileError, OSError) as err:
        p.exit(1, f'remora-torch: {err}\n')


if __name__ == '__main__':
    main()
