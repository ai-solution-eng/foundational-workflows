from os.path import join as pj
from pprint import pprint

import numpy as np

from shared_queries_and_documents import (
    all_inputs,
    text_only,
    image_only,
    joint_text_image,
    multiimage_data,
    script_path,
)
from multimodal_rag.utils.model_adapters import MultiModalReranker
from multimodal_rag.utils.pcai_models import qwen3_vl_reranker_8B

emb_path = pj(script_path, "embs", "pcai_class")
print(emb_path)

np.set_printoptions(linewidth=120)
qwen3_vl_reranker_8B.remote()


async def main():
    model = qwen3_vl_reranker_8B.model

    assert isinstance(model, MultiModalReranker)
    # Just for demonstration.
    openai_inputs = await model._prepare_inputs([], all_inputs)

    text_image_scores = model.score(text_only, image_only)
    text_joint_scores = model.score(text_only, joint_text_image)
    image_joint_scores = model.score(image_only, joint_text_image)

    multiimage_score_example = model.score(all_inputs, multiimage_data)

    all_scores = dict(
        text=openai_inputs[::3],
        image=openai_inputs[1::3],
        joint=openai_inputs[2::3],
        text_image_scores=np.array(text_image_scores),
        text_joint_scores=np.array(text_joint_scores),
        image_joint_scores=np.array(image_joint_scores),
        multiimage_score_example=np.array(multiimage_score_example),
    )

    print("\nSimilarity Scores (Text-Image):")
    pprint(all_scores["text_image_scores"])

    print("\nSimilarity Scores (Text-Joint):")
    pprint(all_scores["text_joint_scores"])

    print("\nSimilarity Scores (Image-Joint):")
    pprint(all_scores["image_joint_scores"])

    print("\nSimilarity Scores (All-(2 images in one sample)):")
    pprint(all_scores["multiimage_score_example"])

    np.save(emb_path, all_scores)  # type: ignore


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
