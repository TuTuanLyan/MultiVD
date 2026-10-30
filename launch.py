import os
import runpy
import sys

if os.environ.get("PYTHONHASHSEED") != "42":
    sys.exit("launch.py: PYTHONHASHSEED must be 42 and set before Python starts (run.py sets it)")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

code_dir, trainer = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.abspath(code_dir))
sys.argv = [trainer] + sys.argv[3:]
runpy.run_path(os.path.join(code_dir, trainer), run_name="__main__")
