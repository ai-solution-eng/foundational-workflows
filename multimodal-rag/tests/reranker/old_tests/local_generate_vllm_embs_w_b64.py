from os.path import join as pj
from pathlib import Path
from pprint import pprint
from typing import Dict, Any

from vllm import LLM, EngineArgs
import numpy as np

from multimodal_rag.utils.model_adapters import InputConversion
from multimodal_rag.utils.pcai_models import qwen3_vl_reranker_8B

# Path to the current script file
try:
    script_path = Path(__file__).parent.resolve()
# For running locally
except Exception:
    script_path = Path("./")

emb_path = pj(script_path, "embs", "vllm_local_with_base64")
print(emb_path)

np.set_printoptions(linewidth=120)


async def replace_image_links_with_base64(images, joint):
    ic = InputConversion(qwen3_vl_reranker_8B)
    image_b64 = await ic(images, add_conversational_elements=False)

    for im, im64 in zip(images, image_b64):
        im["image"] = im64["content"][0]["image_url"]["url"]

    for jt, im64 in zip(joint, image_b64):
        jt["image"] = im64["content"][0]["image_url"]["url"]

    return images, joint


def format_document_to_score_param(doc_dict: Dict[str, Any]):
    content = []

    text = doc_dict.get("text")
    image = doc_dict.get("image")

    if image:
        image_url = image
        content.append({"type": "image_url", "image_url": {"url": image_url}})

    if text:
        content.append({"type": "text", "text": text})

    if not content:
        content.append({"type": "text", "text": ""})

    return {"content": content}


async def main() -> None:
    eng_args = EngineArgs(
        model="Qwen/Qwen3-VL-Reranker-8B",
        runner="pooling",
        dtype="bfloat16",
        trust_remote_code=True,
        hf_overrides={
            "architectures": ["Qwen3VLForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
        mm_processor_kwargs=qwen3_vl_reranker_8B.mm_processor_kwargs,
        max_model_len=8192,
    )

    chat_template = Path("temp_template.jinja").read_text()

    llm = LLM(**vars(eng_args))

    from shared_queries_and_documents import text_only, image_only, joint_text_image

    image_b64, joint_b64 = await replace_image_links_with_base64(image_only, joint_text_image)

    all_scores: dict[str, Any] = dict(
        text=text_only,
        image=image_b64,
        joint=joint_b64,
    )

    text_image_comparisons = []
    text_joint_comparisons = []
    image_joint_comparisons = []

    for query_dict in text_only:
        query_text = query_dict.get("text", "")
        print(f"\nQuery: {query_text}")

        scores = []
        for doc_dict in image_b64:
            doc_param = format_document_to_score_param(doc_dict)
            outputs = llm.score(query_text, doc_param, chat_template=chat_template)
            score = outputs[0].outputs.score
            scores.append(score)

        print(scores)
        text_image_comparisons.append(scores)

    for query_dict in text_only:
        query_text = query_dict.get("text", "")
        print(f"\nQuery: {query_text}")

        scores = []
        for doc_dict in joint_b64:
            doc_param = format_document_to_score_param(doc_dict)
            outputs = llm.score(query_text, doc_param, chat_template=chat_template)
            score = outputs[0].outputs.score
            scores.append(score)

        print(scores)
        text_joint_comparisons.append(scores)

    # TODO: Figure out if this is the correct approach for image data.
    for idx, image_dict in enumerate(image_b64):
        query_text = format_document_to_score_param(image_dict)
        print("Image", idx)

        scores = []
        for doc_dict in joint_b64:
            doc_param = format_document_to_score_param(doc_dict)
            outputs = llm.score(query_text, doc_param, chat_template=chat_template)
            score = outputs[0].outputs.score
            scores.append(score)

        print(scores)
        image_joint_comparisons.append(scores)

    all_scores["text_image_scores"] = np.array(text_image_comparisons)
    all_scores["text_joint_scores"] = np.array(text_joint_comparisons)
    all_scores["image_joint_scores"] = np.array(image_joint_comparisons)

    print("\nSimilarity Scores (Text-Image):")
    pprint(all_scores["text_image_scores"])

    print("\nSimilarity Scores (Text-Joint):")
    pprint(np.array(all_scores["text_joint_scores"]))

    print("\nSimilarity Scores (Image-Joint):")
    pprint(np.array(all_scores["image_joint_scores"]))

    np.save(emb_path, all_scores)  # type: ignore


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
