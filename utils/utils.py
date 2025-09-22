import plotly.express as px
import pandas as pd
import transformer_lens.utils as utils
import json, random
import torch
import numpy as np
from pathlib import Path
import os
from datetime import datetime

def imshow(tensor, renderer=None, midpoint=0, **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=midpoint, color_continuous_scale="RdBu", **kwargs).show(renderer)

def line(tensor, renderer=None, **kwargs):
    px.line(y=utils.to_numpy(tensor), **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

def load_json(filename):
    with open(filename, 'r') as fp:
        return json.load(fp)
    
def first_occurrence(sentence: str, coarse_tag: str, ptb_map: dict, nlp):
    doc = nlp(sentence)
    for tok in doc:
        ptb = tok.tag_              # Penn Treebank POS
        coarse = ptb_map.get(ptb, "other")
        if coarse == coarse_tag:
            return tok.text
    return None

def show_path_patching_heatmap(arr_or_path, title=None, renderer=None,
                               midpoint=0.0, symmetric=True, zmax=None,
                               origin="upper", annotate=False, fmt=".2f", path=None):
    """
    arr_or_path: np.ndarray of shape (n_layers, n_heads) or path to .npy
    origin: "upper" puts layer 0 at top; use "lower" to put it at bottom
    """
    A = np.load(arr_or_path) if isinstance(arr_or_path, (str, Path)) else np.asarray(arr_or_path)
    assert A.ndim == 2, "Expected (layers, heads) array"

    n_layers, n_heads = A.shape
    if symmetric:
        vmax = float(np.nanmax(np.abs(A))) if zmax is None else float(zmax)
        zmin, zmax = -vmax, vmax
    else:
        zmin = None

    fig = px.imshow(
        A,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=midpoint,
        zmin=zmin, zmax=zmax,
        origin=origin,
        labels=dict(color="% path-patch score")
    )
    fig.update_xaxes(title="Head", tickmode="array", tickvals=list(range(n_heads)))
    fig.update_yaxes(title="Layer", tickmode="array", tickvals=list(range(n_layers)))
    fig.update_layout(title=title,
                      height=600,                     # keep same vertical size
                    width=n_heads * 50)

    if annotate:
        text = np.vectorize(lambda x: f"{x:{fmt}}")(A)
        fig.update_traces(text=text, texttemplate="%{text}", textfont_size=10)

    #fig.show(renderer)
    fig.write_html(path, include_plotlyjs="inline", full_html=True)


def save_output(
    output,
    receiver_nodes=None,
    base="deneme.json",
    folder=".",
    add_nodes=True,
    add_timestamp=False,
    ext=".npy",
):
    """
    Save `output` to a NumPy .npy file.

    Args:
        output: np.ndarray | torch.Tensor | sequence
        receiver_nodes: iterable[iterable], e.g. [[1,2,None],[3,4]]
        base: starting filename (its extension will be replaced by `ext`)
        folder: directory to write into (created if missing)
        add_nodes: include a suffix derived from `receiver_nodes`
        add_timestamp: append YYYYMMDD_HHMMSS to the filename
        ext: file extension to write (default ".npy")

    Returns:
        str: full path to the saved file.
    """
    # Build clean base name (avoid .strip() pitfalls)
    root = os.path.splitext(os.path.basename(base))[0]
    parts = [root]

    # Build node suffix like "1-2_3-4" (ignoring None)
    if add_nodes and receiver_nodes:
        recv_str = "_".join(
            "-".join(str(si) for si in s if si is not None) for s in receiver_nodes
        )
        if recv_str:
            parts.append(recv_str)

    if add_timestamp:
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))

    filename = "_".join(parts) + ext
    path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Normalize to numpy
    try:
        import torch  # optional; only used if available
        if isinstance(output, torch.Tensor):
            arr = output.detach().cpu().numpy()
        else:
            arr = output if isinstance(output, np.ndarray) else np.array(output)
    except Exception:
        arr = output if isinstance(output, np.ndarray) else np.array(output)

    print("Saving to", path)
    np.save(path, arr)
    return path

