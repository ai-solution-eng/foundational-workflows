from functools import partial
from os.path import join as pj
from pathlib import Path
from pprint import pprint

import numpy as np
from sentence_transformers import CrossEncoder
import torch

from multimodal_rag.utils.pcai_models import qwen3_vl_reranker_8B

np.set_printoptions(linewidth=120)

from shared_queries_and_documents import (  # noqa: E402
    text_only,
    image_only,
    joint_text_image,
    base_prompt as prompt,
)

# Path to the current script file
try:
    script_path = Path(__file__).parent.resolve()
except Exception:
    script_path = Path("./")

emb_path = pj(script_path, "embs", "hugging_face")
print(emb_path)


# Load the model
def main():
    model = CrossEncoder(
        "Qwen/Qwen3-VL-Reranker-8B",
        model_kwargs=dict(torch_dtype="bfloat16"),
        processor_kwargs=dict(mm_processor_kwargs=qwen3_vl_reranker_8B.mm_processor_kwargs),
        local_files_only=True,
    )

    # Encode queries and documents
    predict = partial(model.predict, prompt=prompt, activation_fn=torch.nn.Sigmoid())

    text_image_scores = [[predict((x, y)) for y in image_only] for x in text_only]
    text_joint_scores = [[predict((x, y)) for y in joint_text_image] for x in text_only]
    image_joint_scores = [[predict((x, y)) for y in joint_text_image] for x in image_only]

    data_dict = dict(
        name="sentence_transformers",
        text_only=text_only,
        image_only=image_only,
        joint=joint_text_image,
        text_image_scores=np.array(text_image_scores),
        text_joint_scores=np.array(text_joint_scores),
        image_joint_scores=np.array(image_joint_scores),
    )

    print("Text to image comparison")
    pprint(data_dict["text_image_scores"])

    print("Text to joint comparison")
    pprint(data_dict["text_joint_scores"])

    print("image to joint comparison")
    pprint(data_dict["image_joint_scores"])

    np.save(emb_path, data_dict)  # type: ignore


if __name__ == "__main__":
    main()
