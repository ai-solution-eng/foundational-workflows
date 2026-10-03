from glob import glob
from os.path import basename as bn
from os.path import join as pj
from pathlib import Path
from typing import ClassVar

import numpy as np

from multimodal_rag.utils.general_tools import cosine_sim

TOLERANCE = 5e-3

# Path to the current script file
try:
    script_path = Path(__file__).parent.resolve()
except Exception:
    script_path = Path("./")

np.set_printoptions(linewidth=160)

# Best performing for RAG
comps = glob(pj(script_path, "embs", "*.npy"))
comparisons = {bn(k): np.load(k, allow_pickle=True).item() for k in comps}
default_dict = comparisons.pop("vllm_local.npy")


class SaveLogAndPrint:
    all_data: ClassVar[list[str]] = []

    def __call__(self, *str_args) -> None:
        input_str = " ".join(str_args)
        self.all_data.append(input_str)
        print(*str_args)

    def write_file(self, output_path: str = "comparison.txt"):
        with open(output_path, "w") as fid:
            fid.write("\n".join(self.all_data))


printer = SaveLogAndPrint()


def stringify_array(np_array, prefix: str = "    "):
    return prefix + np.array_str(np_array).replace("\n", "\n    ")


for key, comp_dict in comparisons.items():
    printer(f"Comparing VLLM Local Python (default) to {key[:-4]}:")
    for key in ["text_embeddings", "image_embeddings", "joint_embeddings"]:
        # Comp1 and 2
        printer("    ", key)
        def_emb = default_dict[key]
        comp_emb = comp_dict[key]

        cos_sim = cosine_sim(def_emb, comp_emb)
        delta = np.abs(1 - cos_sim.diagonal())

        # assert (abs_delta < TOLERANCE).all(), abs_delta
        printer("    ", f"Max delta is {delta.max()}")
        printer("    ", stringify_array(delta), "\n")

    key = "similarities"
    printer("    ", key)

    delta = default_dict[key] - comp_dict[key]
    abs_delta = np.abs(delta)

    # assert (abs_delta < TOLERANCE).all(), abs_delta
    printer(
        "    ",
        f"Max delta is {abs_delta.max()}\n",
    )

for comp_dict in [default_dict] + list(comparisons.values()):
    printer(
        comp_dict["name"],
        "Similarities:\n",
        stringify_array(comp_dict["similarities"]),
        "\n",
    )

printer.write_file()
