import asyncio
import base64
import mimetypes
from os.path import exists, join as pj
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import openai
from openai.types.create_embedding_response import CreateEmbeddingResponse

from shared_queries_and_documents import all_inputs
from multimodal_rag.utils.pcai_models import qwen3_vl_8B
from multimodal_rag.utils.general_tools import cosine_sim

qwen3_vl_8B.remote()

# Path to the current script file
try:
    script_path = Path(__file__).parent.resolve()
except Exception:
    script_path = Path("./")

emb_path = pj(script_path, "embs", "vllm_openai_client")
print(emb_path)

np.set_printoptions(linewidth=120)


def _get_image_mime_type(file_path: str) -> str:
    """
    Detect image MIME type. Works in Python 3.13+ (imghdr replacement).
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith("image/"):
        return mime_type

    with open(file_path, "rb") as f:
        header = f.read(12)

    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    elif header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return "image/gif"
    elif header.startswith(b"BM"):
        return "image/bmp"
    elif header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"

    return "image/jpeg"


async def _fetch_image_async(url: str, http_client: httpx.AsyncClient) -> tuple[str, str]:
    """
    Fetch image and return (base64_data, mime_type).
    """
    if url.startswith(("http://", "https://")):
        response = await http_client.get(url, follow_redirects=True)
        response.raise_for_status()
        image_data = base64.b64encode(response.content).decode("utf-8")
        content_type = response.headers.get("content-type", "image/jpeg")
        mime_type = content_type.split(";")[0].strip()
    else:
        if url.startswith("file://"):
            url = url[7:]
        with open(url, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        mime_type = _get_image_mime_type(url)

    return image_data, mime_type


def _add_extras(requests: list[dict[str, Any]]):
    return [
        [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Represent the user's input."},
                ],
            },
            x,
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                ],
            },
        ]
        for x in requests
    ]


async def _convert_to_openai_client_friendly_parallel(
    inputs: list[dict[str, Any]],
    http_client: httpx.AsyncClient,
    convert_to_bytes: bool = True,
):
    """
    Convert inputs to OpenAI-compatible message format (fully async, parallel image fetching).
    """
    new_request = []

    # First pass: collect all inputs
    for dictionary in inputs:
        content = []

        if "text" in dictionary:
            content.append({"type": "text", "text": dictionary["text"]})

        image_url = dictionary.get("image")

        new_request.append({"role": "user", "content": content, "_image_url": image_url})  # Temporary storage

    # Parallel fetch all images
    if convert_to_bytes:
        fetch_tasks = []
        for item in new_request:
            if item["_image_url"] and isinstance(item["_image_url"], str):
                task = _fetch_image_async(item["_image_url"], http_client)
                fetch_tasks.append(task)
            else:
                fetch_tasks.append(None)  # type: ignore[arg-type]

        results = await asyncio.gather(*[t for t in fetch_tasks if t is not None])

        # Assign results back
        result_idx = 0
        for item in new_request:
            if item["_image_url"] and isinstance(item["_image_url"], str):
                image_data, mime_type = results[result_idx]
                # Image should come first?
                item["content"].insert(  # type: ignore[union-attr]
                    0,
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                    },
                )
                result_idx += 1
            # Remove temporary storage
            del item["_image_url"]
    else:
        for item in new_request:
            image_url = item.pop("_image_url")
            if image_url and isinstance(image_url, str):
                if exists(image_url) and not image_url.startswith("http"):
                    image_url = f"file://{image_url}"
                item["content"].insert(  # type: ignore[union-attr]
                    0, {"type": "image_url", "image_url": {"url": image_url}}
                )

    return _add_extras(new_request)


async def _get_embedding(
    client: openai.AsyncOpenAI,
    messages: list,
    model: str = "Qwen/Qwen3-VL-Embedding-8B",
) -> list[float]:
    """Get embedding for a single input."""
    response = await client.post(
        "/embeddings",
        cast_to=CreateEmbeddingResponse,
        body=dict(
            model=model,
            messages=messages,
            encoding_format="float",
            continue_final_message=True,
            add_special_tokens=True,
            mm_processor_kwargs=dict(max_pixels=504 * 504),
        ),
    )
    return response.data[0].embedding


async def get_all_embeddings(
    client: openai.AsyncOpenAI, inputs: list, model: str = "Qwen/Qwen3-VL-Embedding-8B"
) -> list[list[float]]:
    """Get embeddings for all inputs in parallel."""
    tasks = [_get_embedding(client, input_item, model) for input_item in inputs]
    return await asyncio.gather(*tasks)


async def main():
    # Get async clients
    async_http_client = httpx.AsyncClient()
    async_client = qwen3_vl_8B.async_client

    # Convert inputs (async, parallel image fetching)
    openai_inputs = await _convert_to_openai_client_friendly_parallel(
        all_inputs,
        async_http_client,
    )

    print(f"Number of inputs: {len(openai_inputs)}")

    # Get all embeddings in parallel
    embeddings = await get_all_embeddings(async_client, openai_inputs, model=qwen3_vl_8B.model_name)

    print(f"Number of embeddings: {len(embeddings)}")

    # Calculate similarities
    embedding_array = np.array(embeddings)

    text_embeddings = embedding_array[::3]
    image_embeddings = embedding_array[1::3]
    joint_embeddings = embedding_array[2::3]

    similarities = cosine_sim(text_embeddings, image_embeddings)

    data_dict = dict(
        name="pcai_vllm_openai_client",
        text_only=openai_inputs[::3],
        text_embeddings=text_embeddings,
        image_only=openai_inputs[1::3],
        image_embeddings=image_embeddings,
        joint=openai_inputs[2::3],
        joint_embeddings=joint_embeddings,
        similarities=similarities,
        text_joint_similarities=cosine_sim(text_embeddings, joint_embeddings),
        image_joint_similarities=cosine_sim(image_embeddings, joint_embeddings),
    )

    np.save(emb_path, data_dict)  # type: ignore

    # Cleanup
    await async_http_client.aclose()

    return data_dict


if __name__ == "__main__":
    result = asyncio.run(main())
    print("Done!")
    print(f"Similarities:\n{result['similarities']}")
