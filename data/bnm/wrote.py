import pathlib

# Path to your folder
path = pathlib.Path("/Users/choojunheng/.gemini/antigravity/scratch/agent/data")

# List only items that are files
files = [f.name for f in path.iterdir() if f.is_file()]

print(files)