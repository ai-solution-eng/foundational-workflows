from os.path import join as pj
from pathlib import Path
from pprint import pprint

import numpy as np
from shared_queries_and_documents import all_inputs

from multimodal_rag.utils.general_tools import cosine_sim
from multimodal_rag.utils.pcai_models import qwen3_vl_8B

np.set_printoptions(linewidth=120)

qwen3_vl_8B.remote()

# Path to the current script file
try:
    script_path = Path(__file__).parent.resolve()
except Exception:
    script_path = Path("./")

emb_path = pj(script_path, "embs", "vllm_pcai_template")
print(emb_path)


# Test
def main():
    assert qwen3_vl_8B.preprocessor is not None

    vllm_inputs = [
        qwen3_vl_8B.preprocessor(inp.get("text", ""), inp.get("image"), convert_base64=True) for inp in all_inputs
    ]
    # prompts_only = [v['prompt'] for v in vllm_inputs]
    prompts_only = [
        (
            v["prompt"].replace("<|image_pad|>", v["multi_modal_data"]["image"][:8192])
            if v["multi_modal_data"] is not None
            else v["prompt"]
        )
        for v in vllm_inputs
    ]

    model = qwen3_vl_8B.model

    response = model.embed_documents(prompts_only)
    embedding_array = np.array(response)

    text_embeddings = embedding_array[::3]
    image_embeddings = embedding_array[1::3]
    joint_embeddings = embedding_array[2::3]

    similarities = cosine_sim(text_embeddings, image_embeddings)
    print("Similarities\n")
    pprint(similarities)

    data_dict = {
        "name": "pcai_vllm_templated",
        "text_only": vllm_inputs[::3],
        "text_embeddings": text_embeddings,
        "image_only": vllm_inputs[1::3],
        "image_embeddings": image_embeddings,
        "joint": vllm_inputs[2::3],
        "joint_embeddings": joint_embeddings,
        "similarities": similarities,
        "text_joint_similarities": cosine_sim(text_embeddings, joint_embeddings),
        "image_joint_similarities": cosine_sim(image_embeddings, joint_embeddings),
    }

    np.save(emb_path, data_dict)  # type: ignore


if __name__ == "__main__":
    main()
