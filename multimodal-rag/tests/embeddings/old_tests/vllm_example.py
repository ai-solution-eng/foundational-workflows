# https://github.com/vllm-project/vllm/blob/main/examples/pooling/embed/vision_embedding_online.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example Python client for multimodal embedding API using vLLM API server.

Refer to each `run_*` function for the command to run the server for that model.
"""

from typing import Literal

from openai import OpenAI
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessageParam
from openai.types.create_embedding_response import CreateEmbeddingResponse
from vllm.utils.print_utils import print_embeddings

from multimodal_rag.utils.pcai_models import qwen3_vl_8B

qwen3_vl_8B.remote()
client = qwen3_vl_8B.client
model_name = qwen3_vl_8B.model_name

image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/cat_snow.jpg"
text = "A cat standing in the snow."


def create_chat_embeddings(
    client: OpenAI,
    *,
    messages: list[ChatCompletionMessageParam],
    model: str,
    encoding_format: Literal["base64", "float"] | NotGiven = NOT_GIVEN,
    continue_final_message: bool = False,
    add_special_tokens: bool = False,
) -> CreateEmbeddingResponse:
    """
    Convenience function for accessing vLLM's Chat Embeddings API,
    which is an extension of OpenAI's existing Embeddings API.
    """
    return client.post(
        "/embeddings",
        cast_to=CreateEmbeddingResponse,
        body={
            "messages": messages,
            "model": model,
            "encoding_format": encoding_format,
            "continue_final_message": continue_final_message,
            "add_special_tokens": add_special_tokens,
        },
    )


def run_qwen3_vl(client: OpenAI = client, model: str = model_name):
    """
    Start the server using:

    vllm serve Qwen/Qwen3-VL-Embedding-2B \
        --runner pooling \
        --max-model-len 8192
    """

    default_instruction = "Represent the user's input."

    print("Text embedding output:")
    response = create_chat_embeddings(
        client,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": default_instruction},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                ],
            },
        ],
        model=model,
        encoding_format="float",
        continue_final_message=True,
        add_special_tokens=True,
    )
    print_embeddings(response.data[0].embedding)

    print("Image embedding output:")
    response = create_chat_embeddings(
        client,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": default_instruction},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": ""},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                ],
            },
        ],
        model=model,
        encoding_format="float",
        continue_final_message=True,
        add_special_tokens=True,
    )
    print_embeddings(response.data[0].embedding)

    print("Image+Text embedding output:")
    response = create_chat_embeddings(
        client,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": default_instruction},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "text",
                        "text": f"{text}",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},
                ],
            },
        ],
        model=model,
        encoding_format="float",
        continue_final_message=True,
        add_special_tokens=True,
    )
    print_embeddings(response.data[0].embedding)
