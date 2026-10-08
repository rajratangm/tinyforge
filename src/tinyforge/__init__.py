"""tinyforge: train, evaluate and serve small LLMs on modest GPUs."""

import os

__version__ = "0.1.1"

# tinyforge is PyTorch-only. transformers imports TensorFlow/Flax whenever they are installed, and a broken or
# mismatched one (Colab ships TensorFlow; llama.cpp's requirements downgrade protobuf under it) makes every
# command crash on import. Opt out unless the user has set these themselves; child processes inherit them.
for _var in ("USE_TF", "USE_FLAX", "USE_JAX"):
    os.environ.setdefault(_var, "0")
