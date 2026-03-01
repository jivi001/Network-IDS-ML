import os
from pathlib import Path

# Create directories as shell command failed due to missing powershell
dirs = [
    r"E:\NIDS ML
ids\governance",
    r"E:\NIDS ML
ids\adversarial",
    r"E:\NIDS ML\scripts	esting"
]
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
