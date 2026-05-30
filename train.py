import sys
from pathlib import Path

# Add the project root to sys.path to ensure absolute imports resolve properly.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_model import main

if __name__ == "__main__":
    main()
