import asyncio
from os.path import join as pj

import numpy as np
from shared_queries_and_documents import all_inputs, multiimage_data, script_path

from multimodal_rag.utils.general_tools import cosine_sim
from multimodal_rag.utils.model_adapters import MultiModalEmbeddings
from multimodal_rag.utils.pcai_models import qwen3_vl_8B

qwen3_vl_8B.remote()

emb_path = pj(script_path, "embs", "vllm_openai_client_class")
print(emb_path)

np.set_printoptions(linewidth=120)


async def main():
    model = qwen3_vl_8B.model
    assert isinstance(model, MultiModalEmbeddings)

    # Just for demonstration
    openai_inputs = await model.input_conversion.acall(all_inputs)

    embeddings = model.embed_documents(all_inputs)
    # Verify it works
    multiimage_embeddings = model.embed_documents(multiimage_data)

    print(f"Number of embeddings: {len(embeddings)}")

    # Calculate similarities
    embedding_array = np.array(embeddings)

    text_embeddings = embedding_array[::3]
    image_embeddings = embedding_array[1::3]
    joint_embeddings = embedding_array[2::3]

    similarities = cosine_sim(text_embeddings, image_embeddings)

    data_dict = {
        "name": "pcai_vllm_openai_client",
        "text_only": openai_inputs[::3],
        "text_embeddings": text_embeddings,
        "image_only": openai_inputs[1::3],
        "image_embeddings": image_embeddings,
        "joint": openai_inputs[2::3],
        "joint_embeddings": joint_embeddings,
        "multiimage_data": multiimage_data,
        "multiimage_embeddings": multiimage_embeddings,
        "similarities": similarities,
        "text_joint_similarities": cosine_sim(text_embeddings, joint_embeddings),
        "image_joint_similarities": cosine_sim(image_embeddings, joint_embeddings),
    }

    np.save(emb_path, data_dict)  # type: ignore

    return data_dict


if __name__ == "__main__":
    result = asyncio.run(main())
    print("Done!")
    print(f"Similarities:\n{result['similarities']}")
