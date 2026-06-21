import os
import sys

# Make `text_io` (src/) and the test helpers importable under pytest.
_ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tests"))
