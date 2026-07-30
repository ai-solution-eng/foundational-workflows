import asyncio
from functools import partial
from os.path import join as pj
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from multimodal_rag.utils.model_adapters import InputConversion
from multimodal_rag.utils.pcai_models import qwen3_vl_reranker_8B

np.set_printoptions(linewidth=120)

from shared_queries_and_documents import (  # noqa: E402
    base_prompt as prompt,
)
from shared_queries_and_documents import (  # noqa: E402
    image_only,
    joint_text_image,
    text_only,
)

# Path to the current script file
try:
    script_path = Path(__file__).parent.resolve()
except Exception:
    script_path = Path("./")

emb_path = pj(script_path, "embs", "hugging_face_with_base64")
print(emb_path)


async def replace_image_links_with_base64(images, joint):
    ic = InputConversion(qwen3_vl_reranker_8B)
    image_b64 = await ic(images, add_conversational_elements=False)

    for im, im64 in zip(images, image_b64):
        im["image"] = im64["content"][0]["image_url"]["url"]

    for jt, im64 in zip(joint, image_b64):
        jt["image"] = im64["content"][0]["image_url"]["url"]

    return images, joint


# Load the model
async def main():
    model = CrossEncoder(
        "Qwen/Qwen3-VL-Reranker-8B",
        model_kwargs={"torch_dtype": "bfloat16"},
        processor_kwargs={"mm_processor_kwargs": qwen3_vl_reranker_8B.mm_processor_kwargs},
        local_files_only=True,
    )

    image_b64, joint_b64 = await replace_image_links_with_base64(image_only, joint_text_image)

    # Encode queries and documents
    predict = partial(model.predict, prompt=prompt, activation_fn=torch.nn.Sigmoid())
    text_image_scores = [[predict((x, y)) for y in image_b64] for x in text_only]
    text_joint_scores = [[predict((x, y)) for y in joint_b64] for x in text_only]
    image_joint_scores = [[predict((x, y)) for y in joint_b64] for x in image_b64]

    data_dict = {
        "name": "sentence_transformers",
        "text_only": text_only,
        "image_only": image_b64,
        "joint": joint_b64,
        "text_image_scores": np.array(text_image_scores),
        "text_joint_scores": np.array(text_joint_scores),
        "image_joint_scores": np.array(image_joint_scores),
    }

    print("Text to image comparison")
    pprint(data_dict["text_image_scores"])

    print("Text to joint comparison")
    pprint(data_dict["text_joint_scores"])

    print("image to joint comparison")
    pprint(data_dict["image_joint_scores"])

    np.save(emb_path, data_dict)  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
