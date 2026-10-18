# config.py
# Settings shared by the Python scripts: the paths in local_paths.env (in the repo root, gitignored).
# Copy local_paths.env.example to local_paths.env and fill it in.
import os

repo_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(repo_dir, 'data')

local_paths_file = os.path.join(repo_dir, 'local_paths.env')
if not os.path.exists(local_paths_file):
    raise FileNotFoundError(
        f"{local_paths_file} not found. Copy local_paths.env.example to local_paths.env and fill in your paths."
    )

local_paths = {}
with open(local_paths_file) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            name, value = line.split('=', 1)
            local_paths[name.strip()] = value.strip()


def get_local_path(name):
    if not local_paths.get(name):
        raise KeyError(f"{name} is not set in {local_paths_file}")
    return local_paths[name]


# The storage folder: every script reads and writes its data objects here
OUTPUT_DIR = get_local_path('STORAGE_DIR')
