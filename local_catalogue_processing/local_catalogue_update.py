import logging
import os
from pathlib import Path


# Load .env file from root directory
def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value


load_env()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_local_files(directory, extensions):
    """
    List all files with the given extensions from a local directory
    """
    pass
