import httpx
import json
from os.path import join as pj
from pathlib import Path
import numpy as np
from HPE.MultimodalRAG.tests.old_tests.shared_queries_and_documents_old import all_inputs

from multimodal_rag.utils.pcai_models import qwen3_vl_8B

qwen3_vl_8B.remote()

# Path to the current script file
try:
    script_path = Path(__file__).parent.resolve()
except Exception:
    script_path = Path("./")

emb_path = pj(script_path, "embs", "vllm_pcai_pooling")
print(emb_path)

# Test
assert qwen3_vl_8B.preprocessor is not None

vllm_inputs = [
    qwen3_vl_8B.preprocessor(inp.get("text", ""), inp.get("image"), convert_base64=True) for inp in all_inputs
]


def get_emb(data):
    data["model"] = qwen3_vl_8B.model_name
    if "prompt" in data:
        data["input"] = data.pop("prompt")
    assert "input" in data

    response = httpx.post(
        qwen3_vl_8B.base_url[:-3] + "/pooling",
        json=data,
        headers={"Authorization": f"Bearer {qwen3_vl_8B.api_key}"},
        verify=False,
    )

    res = json.loads(response.content.decode())
    return res["data"][0]["data"]


embeddings = [get_emb(x) for x in vllm_inputs]

embedding_array = np.array(embeddings)

query_embeddings = embedding_array[:4]
doc_embeddings = embedding_array[4:]

similarities = (query_embeddings @ doc_embeddings.T) / (
    np.reshape(np.linalg.norm(query_embeddings, axis=-1), (4, 1))
    * np.reshape(np.linalg.norm(doc_embeddings, axis=-1), (1, 3))
)

print("Similarities\n", similarities)

data_dict = dict(
    name="pcai_pooling",
    queries=vllm_inputs[:4],
    query_embeddings=query_embeddings,
    documents=vllm_inputs[4:],
    doc_embeddings=doc_embeddings,
    similarities=similarities,
)

np.save(emb_path, data_dict)  # type: ignore
