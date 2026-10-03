import os
from os.path import join as pj
from pprint import pprint
from typing import Any

import numpy as np
from shared_queries_and_documents import all_inputs, script_path
from vllm import LLM, EngineArgs
from vllm.multimodal.utils import fetch_image

from multimodal_rag.utils.general_tools import cosine_sim
from multimodal_rag.utils.pcai_models import qwen3_vl_8B

emb_path = pj(script_path, "embs", "vllm_local")
print(emb_path)

np.set_printoptions(linewidth=120)


def format_input_to_conversation(
    input_dict: dict[str, Any], instruction: str = "Represent the user's input."
) -> list[dict]:
    content = []

    text = input_dict.get("text")
    image = input_dict.get("image")

    if image:
        image_content = None
        if isinstance(image, str):
            if image.startswith(("http", "https", "oss")):
                image_content = image
            else:
                abs_image_path = os.path.abspath(image)
                image_content = "file://" + abs_image_path
        else:
            image_content = image

        if image_content:
            content.append(
                {
                    "type": "image",
                    "image": image_content,
                }
            )

    if text:
        content.append({"type": "text", "text": text})

    if not content:
        content.append({"type": "text", "text": ""})

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]

    return conversation


def prepare_vllm_inputs(
    input_dict: dict[str, Any],
    llm,
    instruction: str = "Represent the user's input.",
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mm_processor_kwargs is None:
        mm_processor_kwargs = {}
    image = input_dict.get("image")

    conversation = format_input_to_conversation(input_dict, instruction)

    prompt_text = llm.llm_engine.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    multi_modal_data = None
    if image:
        if isinstance(image, str):
            if image.startswith(("http", "https", "oss")):
                try:
                    image_obj = fetch_image(image)
                    multi_modal_data = {"image": image_obj}
                except Exception as e:
                    print(f"Warning: Failed to fetch image {image}: {e}")
            else:
                abs_image_path = os.path.abspath(image)
                if os.path.exists(abs_image_path):
                    from PIL import Image

                    image_obj = Image.open(abs_image_path)
                    multi_modal_data = {"image": image_obj}
                else:
                    print(f"Warning: Image file not found: {abs_image_path}")
        else:
            multi_modal_data = {"image": image}

    result = {"prompt": prompt_text, "multi_modal_data": multi_modal_data}
    if result["multi_modal_data"] is not None:
        result["mm_processor_kwargs"] = mm_processor_kwargs

    return result


def main():
    eng_args = EngineArgs(
        model="Qwen/Qwen3-VL-Embedding-8B",
        runner="pooling",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192,
        mm_processor_kwargs=qwen3_vl_8B.mm_processor_kwargs,
    )

    llm = LLM(**vars(eng_args))

    all_inputs = [
        (
            x
            if isinstance(x, dict)
            else (
                {"text": x}
                if isinstance(x, str) and not x.startswith(("file://", "http://", "https://"))
                else {"image": x}
            )
        )
        for x in all_inputs  # noqa: F823
    ]
    vllm_inputs = [prepare_vllm_inputs(inp, llm) for inp in all_inputs]

    outputs = llm.embed(vllm_inputs)

    embeddings_list = []
    for i, output in enumerate(outputs):
        emb = output.outputs.embedding
        embeddings_list.append(emb)
        print(f"Input {i} embedding shape: {len(emb)}")

    embeddings = np.array(embeddings_list)
    print(f"\nEmbeddings shape: {embeddings.shape}")

    text_embeddings = embeddings[::3]
    image_embeddings = embeddings[1::3]
    joint_embeddings = embeddings[2::3]

    similarities = cosine_sim(text_embeddings, image_embeddings)

    print("\nSimilarity Scores:")
    pprint(similarities)

    data_dict = {
        "name": "local_vllm_python",
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
