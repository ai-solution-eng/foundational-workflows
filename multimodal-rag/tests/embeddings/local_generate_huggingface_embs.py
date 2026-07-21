from os.path import join as pj
from pprint import pprint

from sentence_transformers import SentenceTransformer
import numpy as np

from multimodal_rag.utils.pcai_models import qwen3_vl_8B
from shared_queries_and_documents import (
    text_only,
    image_only,
    joint_text_image,
    script_path,
)

np.set_printoptions(linewidth=120)

# Path to the current script file. If running in a notebook, use except statement.

emb_path = pj(script_path, "embs", "hugging_face")
print(emb_path)


# Load the model
def main() -> None:
    model = SentenceTransformer(
        "Qwen/Qwen3-VL-Embedding-8B",
        model_kwargs=dict(torch_dtype="bfloat16"),
        processor_kwargs=dict(mm_processor_kwargs=qwen3_vl_8B.mm_processor_kwargs),
    )

    # Encode queries and documents
    text_embeddings = model.encode(text_only)
    image_embeddings = model.encode(image_only)
    joint_embeddings = model.encode(joint_text_image)

    print(text_embeddings.shape, image_embeddings.shape)

    # Compute similarities
    similarities: np.ndarray = model.similarity(text_embeddings, image_embeddings).numpy()
    pprint(similarities)

    data_dict = dict(
        name="sentence_transformers",
        text_only=text_only,
        text_embeddings=text_embeddings,
        image_only=image_only,
        image_embeddings=image_embeddings,
        joint=joint_text_image,
        joint_embeddings=joint_embeddings,
        similarities=similarities,
    )

    np.save(emb_path, data_dict)  # type: ignore


if __name__ == "__main__":
    main()
