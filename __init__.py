"""OpenFace for PyTorch using the PyTorch weights `net.pth`."""

import pathlib
import net

openface_model_path = str(pathlib.Path(__file__).resolve().parent / "net.pth")
