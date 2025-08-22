import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / "src"
src_str = str(src)
if src_str not in sys.path:
    sys.path.insert(0, src_str)

if "PYTHONPATH" in os.environ:
    if src_str not in os.environ["PYTHONPATH"].split(os.pathsep):
        os.environ["PYTHONPATH"] = src_str + os.pathsep + os.environ["PYTHONPATH"]
else:
    os.environ["PYTHONPATH"] = src_str
