from glob import glob
from os.path import basename as bn
from os.path import join as pj
from pathlib import Path
from typing import ClassVar

import numpy as np

TOLERANCE = 0.05

# Path to the current script file
try:
    script_path = Path(__file__).parent.resolve()
except Exception:
    script_path = Path("./")

np.set_printoptions(linewidth=160)

# Best performing for RAG
comps = sorted(glob(pj(script_path, "embs", "*.npy")))
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
    printer(f"\n\n\nComparing {key[:-4]} to local VLLM python implementation:")
    for key in ["text_image_scores", "text_joint_scores", "image_joint_scores"]:
        # Comp1 and 2
        printer("\n  ", key)
        def_score = default_dict[key]
        comp_score = comp_dict[key]

        def_sort = def_score.argsort(axis=-1)
        comp_sort = comp_score.argsort(axis=-1)
        sort_check = def_sort == comp_sort

        delta = np.abs(def_score - comp_score)
        stacked_sort = np.stack([d[ds] for d, ds in zip(delta, def_sort)])

        comp_str = "\n".join(
            f"{x}\t{y}"
            for x, y in zip(
                stringify_array(comp_score).split("\n"),
                stringify_array(def_score).split("\n"),
            )
        )

        printer("\n    ", f"Comparison scores vs. Base\n\n{comp_str}")
        printer("\n    ", f"Sorted Dictionary Comparison\n\n{stringify_array(sort_check)}")
        printer(
            "\n    ",
            f"delta matrix, sorted by top score\n\n{stringify_array(stacked_sort)}",
        )

printer.write_file()
