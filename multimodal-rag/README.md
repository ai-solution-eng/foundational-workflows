# Multimodal RAG Implementation

## Goal

The stated goal of this repo is to develop an end-to-end solution for multimodal RAG. The supported use cases must include:
* Support for a joint text, audio, video embedding space. This includes permutations (e.g. 2 images) of these objects.
* Additional support for audio, perhaps by mapping it to text through ASR by cohere_transcribe_3_2b (previously whisper).
    * Note the deficiency for things like music, action scenes, and other acoustic events. We don't currently have a general purpose audio encoder deployed, but support for this can be added by request. Additionally, some newer models include the audio modality.
* Allow the user to also input e.g. text with an image and search the embedding space with joint data types. A perfect example could be video with a captioned audio.
* Configurable routing to a reranker to improve performance at the potential cost of latency.
* Configurable routing to a VLLM (e.g. Gemma 4 31B) for large LLMs that only support text (e.g. Deepseek V4 Flash/Pro).
* API support to return not only the LLM tuned answer based on retrieved data, but also the data itself. Reference capturing at minimum.

## Usage

The full E2E module is still under development, but [hosted here](https://github.com/ai-solution-eng/internal-projects/tree/main/multimodal-rag-project/src/multimodal_rag/rag_system.py) for reference.

### Input

I found the simplest way to represent the data to be that of the huggingface API. Strings are auto-detected: plain text stays text, URLs/data URIs/local paths to media files are recognised. For dicts, the ``image``, ``video`` and ``audio`` keys accept either a single URL or a list for multiple items:

```
inputs = [
    "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    {
        "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    },
    {
        "text": "Some text related to the images.",
        "image": ["/path/to/image1.jpeg", "/path/to/image2.jpeg"],
    },
    {
        "text": "A caption of the video.",
        "video": "/path/to/video.mp4",
    },
]
```

### Embedding module

This example is essentially taken from tests. The following will be reused for the reranker portion.

```
all_inputs = [{'text': 'The aurora borealis over a snowy mountain'},
 {'image': 'https://fastly.picsum.photos/id/901/3517/1726.jpg?hmac=u0_XUn-JRaNrL-9fSm-m87xL3JtQbHFxQ068EpJSgb4'},
 {'text': 'The aurora borealis over a snowy mountain',
  'image': 'https://fastly.picsum.photos/id/901/3517/1726.jpg?hmac=u0_XUn-JRaNrL-9fSm-m87xL3JtQbHFxQ068EpJSgb4'},
 {'text': 'A skyscraper high above the other buildings in a city on a cloudy day.'},
 {'image': 'https://fastly.picsum.photos/id/898/2655/1331.jpg?hmac=grTVBjfqQmnPY63ZCi1h82RC1Q1rDfGSmpSJSjfzIjU'},
 {'text': 'A skyscraper high above the other buildings in a city on a cloudy day.',
  'image': 'https://fastly.picsum.photos/id/898/2655/1331.jpg?hmac=grTVBjfqQmnPY63ZCi1h82RC1Q1rDfGSmpSJSjfzIjU'},
 {'text': 'The top of a skyscraper with an antenna in the clouds.'},
 {'image': 'https://fastly.picsum.photos/id/500/2960/1555.jpg?hmac=lWAHvok_5yk5PpJwOxgU-bLEr4gPAHoXrJlkmZdkl_I'},
 {'text': 'The top of a skyscraper with an antenna in the clouds.',
  'image': 'https://fastly.picsum.photos/id/500/2960/1555.jpg?hmac=lWAHvok_5yk5PpJwOxgU-bLEr4gPAHoXrJlkmZdkl_I'},
 {'text': 'Black and white image of the middle of the statue of liberty.'},
 {'image': 'https://fastly.picsum.photos/id/742/3784/1140.jpg?hmac=AzDecEd-uYZFG4vVKpP9XY17gY7TjRdKs5iQn5LxIn8'},
 {'text': 'Black and white image of the middle of the statue of liberty.',
  'image': 'https://fastly.picsum.photos/id/742/3784/1140.jpg?hmac=AzDecEd-uYZFG4vVKpP9XY17gY7TjRdKs5iQn5LxIn8'},
 {'text': 'Black and white image of a lake reflecting the trees by its side.'},
 {'image': 'https://fastly.picsum.photos/id/412/3630/1502.jpg?hmac=Cg4GcGfWz7q3cI-Cf9Sxfrx2j75BzYGsHZgPDdH-ns8'},
 {'text': 'Black and white image of a lake reflecting the trees by its side.',
  'image': 'https://fastly.picsum.photos/id/412/3630/1502.jpg?hmac=Cg4GcGfWz7q3cI-Cf9Sxfrx2j75BzYGsHZgPDdH-ns8'},
 {'text': 'A man crouching staring down at the tops of clouds from a mountain.'},
 {'image': 'https://fastly.picsum.photos/id/685/2853/1335.jpg?hmac=X4eZPprxEVmxX--D-0yNI235iDLFdn9ifMhQKNNX4vU'},
 {'text': 'A man crouching staring down at the tops of clouds from a mountain.',
  'image': 'https://fastly.picsum.photos/id/685/2853/1335.jpg?hmac=X4eZPprxEVmxX--D-0yNI235iDLFdn9ifMhQKNNX4vU'}]
```

Execution:
```
from multimodal_rag.utils.pcai_models import qwen3_vl_8B
import numpy as np
from multimodal_rag.utils.general_tools import cosine_sim

np.set_printoptions(linewidth=150, precision=3)

qwen3_vl_8B.remote()  # Required if running on your local machine.
model = qwen3_vl_8B.model

embeddings = model.embed_documents(all_inputs)

from multimodal_rag.utils.general_tools import cosine_sim
cosine_sim(np.array(embeddings), np.array(embeddings))
```

Output. Note the 1s are for exact matches, and the 3x3 block diagonal is for samples with shared image and/or caption.
```
array([[1.   , 0.67 , 0.828, 0.11 , 0.073, 0.109, 0.178, 0.102, 0.165, 0.12 , 0.091, 0.118, 0.115, 0.209, 0.223, 0.285, 0.205, 0.307],
       [0.67 , 1.   , 0.77 , 0.003, 0.059, 0.035, 0.077, 0.092, 0.074, 0.09 , 0.098, 0.078, 0.035, 0.127, 0.111, 0.186, 0.211, 0.203],
       [0.828, 0.77 , 1.   , 0.085, 0.061, 0.129, 0.132, 0.09 , 0.174, 0.119, 0.107, 0.149, 0.088, 0.164, 0.233, 0.243, 0.186, 0.304],
       [0.11 , 0.003, 0.085, 1.   , 0.54 , 0.737, 0.612, 0.415, 0.487, 0.169, 0.264, 0.225, 0.075, 0.057, 0.096, 0.241, 0.162, 0.246],
       [0.073, 0.059, 0.061, 0.54 , 1.   , 0.746, 0.445, 0.622, 0.541, 0.198, 0.305, 0.236, 0.035, 0.123, 0.066, 0.14 , 0.214, 0.184],
       [0.109, 0.035, 0.129, 0.737, 0.746, 1.   , 0.532, 0.561, 0.702, 0.21 , 0.311, 0.31 , 0.067, 0.085, 0.159, 0.207, 0.169, 0.282],
       [0.178, 0.077, 0.132, 0.612, 0.445, 0.532, 1.   , 0.598, 0.738, 0.254, 0.278, 0.254, 0.083, 0.058, 0.075, 0.334, 0.27 , 0.352],
       [0.102, 0.092, 0.09 , 0.415, 0.622, 0.561, 0.598, 1.   , 0.777, 0.264, 0.365, 0.278, 0.053, 0.087, 0.074, 0.215, 0.245, 0.248],
       [0.165, 0.074, 0.174, 0.487, 0.541, 0.702, 0.738, 0.777, 1.   , 0.295, 0.337, 0.362, 0.105, 0.098, 0.17 , 0.268, 0.208, 0.34 ],
       [0.12 , 0.09 , 0.119, 0.169, 0.198, 0.21 , 0.254, 0.264, 0.295, 1.   , 0.641, 0.752, 0.364, 0.2  , 0.245, 0.13 , 0.045, 0.097],
       [0.091, 0.098, 0.107, 0.264, 0.305, 0.311, 0.278, 0.365, 0.337, 0.641, 1.   , 0.786, 0.143, 0.129, 0.139, 0.162, 0.17 , 0.171],
       [0.118, 0.078, 0.149, 0.225, 0.236, 0.31 , 0.254, 0.278, 0.362, 0.752, 0.786, 1.   , 0.222, 0.14 , 0.302, 0.16 , 0.113, 0.195],
       [0.115, 0.035, 0.088, 0.075, 0.035, 0.067, 0.083, 0.053, 0.105, 0.364, 0.143, 0.222, 1.   , 0.641, 0.71 , 0.072, 0.028, 0.063],
       [0.209, 0.127, 0.164, 0.057, 0.123, 0.085, 0.058, 0.087, 0.098, 0.2  , 0.129, 0.14 , 0.641, 1.   , 0.731, 0.12 , 0.152, 0.144],
       [0.223, 0.111, 0.233, 0.096, 0.066, 0.159, 0.075, 0.074, 0.17 , 0.245, 0.139, 0.302, 0.71 , 0.731, 1.   , 0.111, 0.095, 0.194],
       [0.285, 0.186, 0.243, 0.241, 0.14 , 0.207, 0.334, 0.215, 0.268, 0.13 , 0.162, 0.16 , 0.072, 0.12 , 0.111, 1.   , 0.606, 0.794],
       [0.205, 0.211, 0.186, 0.162, 0.214, 0.169, 0.27 , 0.245, 0.208, 0.045, 0.17 , 0.113, 0.028, 0.152, 0.095, 0.606, 1.   , 0.768],
       [0.307, 0.203, 0.304, 0.246, 0.184, 0.282, 0.352, 0.248, 0.34 , 0.097, 0.171, 0.195, 0.063, 0.144, 0.194, 0.794, 0.768, 1.   ]])
```

### Reranker Module

Very similar syntax to the embedding module. Note there is also a `rerank` function which returns the ranked samples. It matches the scores API.

```
from multimodal_rag.utils.pcai_models import qwen3_vl_reranker_8B
import numpy as np
from multimodal_rag.utils.general_tools import cosine_sim

np.set_printoptions(linewidth=150, precision=3)

qwen3_vl_reranker_8B.remote()  # Required if running on your local machine.
model = qwen3_vl_reranker_8B.model

# To align the image and joint samples corresponding to the text.
matched_samples = [z for x,y in zip(all_inputs[1::3], all_inputs[2::3]) for z in (x,y)]

np.array(model.score(all_inputs[::3], all_inputs[1::3] + all_inputs[2::3]))
```

Output. The 1x2 diagonal elements show the matched elements:

```
array([[0.662, 0.96 , 0.007, 0.006, 0.008, 0.007, 0.015, 0.007, 0.041, 0.033, 0.018, 0.014],
       [0.008, 0.007, 0.432, 0.848, 0.236, 0.192, 0.056, 0.036, 0.017, 0.014, 0.03 , 0.027],
       [0.022, 0.018, 0.243, 0.262, 0.626, 0.937, 0.077, 0.05 , 0.017, 0.012, 0.046, 0.04 ],
       [0.008, 0.004, 0.023, 0.019, 0.032, 0.026, 0.733, 0.964, 0.04 , 0.058, 0.015, 0.01 ],
       [0.011, 0.007, 0.009, 0.008, 0.007, 0.005, 0.022, 0.026, 0.682, 0.947, 0.015, 0.013],
       [0.028, 0.02 , 0.02 , 0.022, 0.032, 0.041, 0.077, 0.041, 0.041, 0.03 , 0.738, 0.951]])
```

### Full RAG

There is a full pipe for RAG testing available [here](https://github.com/ai-solution-eng/internal-projects/blob/main/multimodal-rag-project/src/multimodal_rag/rag_system.py#L1043) with a test_script [here](https://github.com/ai-solution-eng/internal-projects/blob/main/multimodal-rag-project/tests/full_pipeline/run_pipeline.py). 

I ran this pipeline with some local LLM pdfs, images (of my kid), and a video of 2 people playing an old video game. I will develop a universal input parser to make this part transparent to the user. 

Overall, the results were excellent. It could retrieve images when needed, including from within pdfs, as well as videos. Some results are shown below.

The flow is as follows at query time:
1. LLM determines if it needs a RAG call.
2. Data is passed through an optional preprocessor component. This handles any missing modality, (in the current model case, specifically audio for transcribing videos and audio), and converts it to text.
3. The query is embedded, and retrival occurs.
4. An optional reranking occurs to improve performance and limit results.
5. A post-processor is applied, for e.g. a VLM to convert images back to text for a text only LLM (such as deepseek v4).

The flow is a bit more complicated (containing splits of video and images) when creating or appending the database. This is summarized in the image below:

<div align="center"><img src="./rag_system_flow-1.png" width="700" alt="Flow diagram"></div>

#### Example execution

```
def run_call(query: str, **kwargs):
    print(query)
    print('\n', '#'*40, '\n')
    print('Logging Info:')
    output = rag.generate(query, **kwargs)
    print('\n', '#'*40, '\n')
    print(output)
    print('\n', '#'*120, '\n')

run_call('Can you find data of or about a kid getting his hair cut?', route=True)
run_call('Can you find and describe images of a child climbing through a snow tunnel?', route=True, use_reranker=True)
run_call('Can you describe to me what the most important new approaches were in the Deepseek V4 Flash family of models', route=True, use_reranker=True, top_k=20, reranker_top_k=5)
run_call('Can you show me videos of an old video game?', route=True, use_reranker=True, top_k=10, reranker_top_k=5)
```

Output with benchmarks (only enabled with the verbose logging level included in this repo):
<pre><code style="white-space: pre-wrap;">
Can you find data of or about a kid getting his hair cut?

 ######################################## 

Logging Info:
2026-06-11 09:10:56,454 - VERBOSE - 1.74s route(llm)  → RAG needed
2026-06-11 09:10:57,019 - VERBOSE - 0.57s retrieve  — 10 docs (top_k=10, reranker=no)
2026-06-11 09:11:00,437 - VERBOSE -   3.41s vlm describe  — 1 media items
2026-06-11 09:11:00,988 - VERBOSE -   3.97s vlm describe  — 1 media items
2026-06-11 09:11:01,062 - VERBOSE -   4.04s vlm describe  — 1 media items
2026-06-11 09:11:01,180 - VERBOSE -   4.16s vlm describe  — 1 media items
2026-06-11 09:11:01,676 - VERBOSE -   4.65s vlm describe  — 1 media items
2026-06-11 09:11:02,415 - VERBOSE -   5.39s vlm describe  — 1 media items
2026-06-11 09:11:02,415 - VERBOSE - 5.40s postproc  — VLM/ASR conversion (6 docs with media)
2026-06-11 09:11:07,645 - VERBOSE - 5.23s llm       — generation (192 tokens?)  [total 12.93s]

 ######################################## 

Based on the provided context, there are multiple images showing a child getting a haircut. For example:

- **DSC01367.JPG**: A person in a blue shirt uses an electric hair clipper on a young child with blonde hair, who sits in a wooden chair and looks at the camera.
- **DSC01365.JPG**: An adult's arms hold a black clipper above a baby in a high chair, preparing to cut their hair.
- **DSC01368.JPG**: A close-up shows a toddler in a maroon shirt looking upward while an adult holds clippers toward them.
- **DSC01370.JPG**: A man (bearded, gray shirt) crouches to cut the hair of a toddler boy seated in a wooden chair, with a bookshelf and fireplace in the background.
- **DSC01366.JPG**: An adult in a blue shirt steadies a toddler's forehead while clipping the top of their hair.
- **DSC01364.JPG**: An adult in a blue hoodie trims a child's blonde hair with gray clippers; the child sits in a high chair.

These descriptions collectively document a toddler’s first haircut, showing the setting, clothing, tools (electric clippers), and interactions. The sources are the image filenames (e.g., DSC01367.JPG) but no specific page numbers are provided.

 ######################################################################################################################## 

Can you find and describe images of a child climbing through a snow tunnel?

 ######################################## 

Logging Info:
2026-06-11 09:11:08,810 - VERBOSE - 1.16s route(llm)  → RAG needed
2026-06-11 09:11:10,504 - VERBOSE - 1.69s retrieve  — 3 docs (top_k=10, reranker=yes)
2026-06-11 09:11:13,645 - VERBOSE -   3.14s vlm describe  — 1 media items
2026-06-11 09:11:13,706 - VERBOSE -   3.20s vlm describe  — 1 media items
2026-06-11 09:11:13,929 - VERBOSE -   3.43s vlm describe  — 1 media items
2026-06-11 09:11:13,929 - VERBOSE - 3.43s postproc  — VLM/ASR conversion (3 docs with media)
2026-06-11 09:11:18,318 - VERBOSE - 4.39s llm       — generation (224 tokens?)  [total 10.67s]

 ######################################## 

Based on the provided context, there are three images showing a child in a snow tunnel. Below are descriptions of each:

1. **[DSC01379.JPG]** – A high-angle, close-up shot of a young child crawling through a snow tunnel. The child wears a dark navy blue winter jacket with red accents and "CYBER" text on the sleeve, plus a grey knit beanie covered in snow. Their face is partially visible, looking downward as they navigate the tunnel. The snow forms a thick white frame around the child.

2. **[DSC01378.JPG]** – A medium shot taken from inside the snow tunnel, looking outward. A child is peering in from the opening, wearing a black winter coat and black knit hat with small pom-poms. They are smiling and looking directly at the camera. The tunnel walls are rough and uneven, and the opening circularly frames the child on the right.

3. **[DSC01380.JPG]** – A close-up, eye-level shot of a young child smiling and looking down while crawling through the snow tunnel. The child has a dark blue coat with red accents, a grey beanie covered in snow, and black/grey gloves. The tunnel is made of thick, packed snow, creating a bright white frame. Lighting is diffused and overcast.

All three depict a child in or at the entrance of a snow tunnel, with variations in angle, attire, and perspective.

 ######################################################################################################################## 

Can you describe to me what the most important new approaches were in the Deepseek V4 Flash family of models

 ######################################## 

Logging Info:
2026-06-11 09:11:19,885 - VERBOSE - 1.57s route(llm)  → RAG needed
2026-06-11 09:11:21,636 - VERBOSE - 1.75s retrieve  — 5 docs (top_k=20, reranker=yes)
2026-06-11 09:11:21,636 - VERBOSE - 0.00s postproc  — text-only (skipped VLM/ASR)
2026-06-11 09:11:35,634 - VERBOSE - 14.00s llm       — generation (386 tokens?)  [total 17.32s]

 ######################################## 

Based on the provided context, the DeepSeek-V4 Flash family introduced several key architectural and methodological innovations that distinguish it from its predecessor, DeepSeek-V3.2. These new approaches focus on extreme long-context efficiency and a revamped post-training pipeline:

1. **Hybrid Attention with CSA and HCA**  
   - The attention layers use a combination of **Compressed Sparse Attention (CSA)** and **Heavily Compressed Attention (HCA)**. CSA compresses keys/values with a rate `m=4` and selects a top‑k of 512 KV entries for sparse attention. HCA applies a much higher compression rate of 128.  
   - The first two transformer layers use pure sliding window attention (window size 128), while subsequent layers interleave CSA and HCA.  
   - This design dramatically reduces inference FLOPs and KV cache size — for a 1M‑token context, DeepSeek‑V4‑Flash achieves only **10% of the single‑token FLOPs and 7% of the KV cache size** compared to DeepSeek‑V3.2 (source: Page 5).

2. **Precision Optimizations**  
   - The routed expert parameters are stored and computed in **FP4 precision**, while other operations use FP8. Although current hardware treats FP4×FP8 at the same peak FLOPs as FP8×FP8, the architecture is designed to be 1/3 more efficient on future hardware (source: Page 5).

3. **Manifold-Constrained Hyper-Connections (mHC)**  
   - Residual connections are strengthened using **mHC** (expansion factor `n_hc=4`, Sinkhorn‑Knopp iterations `t_max=20`), which improves gradient flow and training stability (source: Page 6 and 24).

4. **MoE Architecture with Aggressive Sparsity**  
   - The model employs Mixture‑of‑Experts (MoE) with **256 routed experts** and **1 shared expert**, activating only **6 experts per token**. The first three MoE layers use hash routing for load balancing. Total parameters are 284B and activated parameters are only **13B** (source: Page 24).

5. **Post‑Training via On‑Policy Distillation (OPD)**  
   - Instead of the mixed Reinforcement Learning stage used in DeepSeek‑V3.2, the V4 Flash family adopts a two‑stage paradigm:  
     1. **Specialist training**: Domain‑specific experts are independently fine‑tuned (SFT) and then optimized with RL using Group Relative Policy Optimization (GRPO).  
     2. **Unified consolidation**: All specialist knowledge is merged into a single student model through **On‑Policy Distillation (OPD)**, where the student minimizes reverse KL divergence against the teacher experts (source: Pages 5 and 28).

These innovations allow DeepSeek‑V4‑Flash to natively support million‑token contexts efficiently while achieving competitive performance — especially in reasoning tasks — despite its much smaller activated parameter count (13B) compared to the larger V4‑Pro (49B) or V3.2 (37B).

 ######################################################################################################################## 

Can you show me videos of an old video game?

 ######################################## 

Logging Info:
2026-06-11 09:11:36,751 - VERBOSE - 1.12s route(llm)  → RAG needed
2026-06-11 09:11:42,538 - VERBOSE - 5.79s retrieve  — 5 docs (top_k=10, reranker=yes)
2026-06-11 09:11:49,057 - VERBOSE -   6.51s vlm describe  — 1 media items
2026-06-11 09:11:51,360 - VERBOSE -   8.81s vlm describe  — 1 media items
2026-06-11 09:11:51,757 - VERBOSE -   9.21s vlm describe  — 1 media items
2026-06-11 09:11:52,902 - VERBOSE -   10.36s vlm describe  — 1 media items
2026-06-11 09:11:52,956 - VERBOSE -   10.41s vlm describe  — 1 media items
2026-06-11 09:11:52,956 - VERBOSE - 10.42s postproc  — VLM/ASR conversion (5 docs with media)
2026-06-11 09:11:57,974 - VERBOSE - 5.02s llm       — generation (249 tokens?)  [total 22.34s]

 ######################################## 

Certainly! The retrieved context contains several clips from **Super Smash Bros.** for the **Nintendo 64**, which is indeed an older video game (released in 1999). The footage shows matches between **Yoshi** and **Captain Falcon** on the **Hyrule Castle** stage.

Here’s a summary of what you can see in these clips:

- **Stage selection** – The player starts on the Stage Select screen, choosing “Peach’s Castle” and then “Dream Land” before finally picking “Hyrule Castle”.
- **Multiple matches** – Each clip captures different portions of the gameplay, with varying damage percentages and knockouts. For example:
  - One clip shows Yoshi quickly building up Captain Falcon’s damage from 46% to 150% and then launching him off the stage.
  - Another shows a back‑and‑forth exchange where both characters trade hits, and Captain Falcon is eventually KO’d.
  - A later clip shows Yoshi being knocked out (with a “SCORE -1” indicator) and then respawning.
- **Additional visuals** – The videos are presented in a split‑screen layout: the game on the left, a system monitor/overlay (with white text and colored bars) on the top right, and a webcam feed of a man watching and reacting in the bottom right.

While I can’t play the videos directly, I can describe their content in detail or point you to the specific timestamps if you’d like to know more about a particular moment. Would you like me to focus on a specific part of the match, such as the opening, the first KO, or the later exchanges?

 ######################################################################################################################## 
</code></pre>

## Plan for Next Steps

I would like to wrap the components into a callable MCP server, with a html frontend hosted on PCAI, to enable the user to upload datasets for use. Since the models can take http(s) links, anything that is already available should only be stored as a link.

This would allow a model to be equipped with a multimodal rag dataset very easily in an agentic way.

## Debugging Effort:

This section highlights several of the challenges I encountered while setting up the base models for testing. For some of the highlights and past testing, see the test dir.

#### Summary

The following have examples in the [test dir](https://github.com/ai-solution-eng/internal-projects/tree/main/multimodal-rag-project/tests).

Embeddings:
* For the [embedding client](https://github.com/ai-solution-eng/internal-projects/blob/main/multimodal-rag-project/src/multimodal_rag/utils/langchain_embed_override.py#L420), I can get close matches between the VLLM implementation highlighted on [huggingface](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B#vllm-basic-usage-example). By close, I mean cosine distance is ~`1e-4`. This does differ slightly from the output of the [sentence-transformers demonstration](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B#sentence-transformers) of the same page, but is reasonable. See below for more details.
* The OpenAI Client with `client.embeddings.create` does not support calling with dictionaries, eliminating the possibility of image calls and also joint text-image calls. See [here](https://github.com/openai/openai-python/blob/main/src/openai/resources/embeddings.py#L178).
* The Langchain Embeddings class does not produce equivalent representations for Qwen3-VL-Embedding-8B as the local variants, even for simple text. There is a strange design decision to differ from the OpenAI client defaults [that I fixed here](https://github.com/ai-solution-eng/internal-projects/blob/main/multimodal-rag-project/src/multimodal_rag/utils/pcai_models.py#L85) to match offline text embedding results.

Reranker:
* [This tool](https://github.com/ai-solution-eng/internal-projects/blob/main/multimodal-rag-project/src/multimodal_rag/utils/langchain_embed_override.py#L509) was developed to serve as a reranker wrapper for remote models.
* The reference CrossEncoder implementation [highlighted here](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B#using-sentence-transformers) fails to load with the error `TypeError: LogitScore.__init__() missing 1 required positional argument: 'true_token_id'`. You need to place a file @ ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Reranker-8B/snapshots/b212dc8c91a8164aef1ea2de9c1a867611e75c04/1_CausalScoreHead/config.json with contents `{"true_token_id": 9693, "false_token_id": 2152}` to solve it. Then you get sensible results. Results agree with or without base64 encoding.
* [The VLLM local implementation](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B#using-vllm) is thus used for testing, which deploys successfully. Results look very reasonable for reranking.
* 

#### Comparisons of embeddings

I created captions for 6 images and created a dataset out of the captions (text), the images, and the paired image with caption data.

Full tests show equivalence on local tests of base64 to http links, giving confidence in the approach used to send base64 data.

##### Embedding Model

We achieve near perfect reproduction of the offline and online versions of vllm. The full results [are here](https://github.com/ai-solution-eng/internal-projects/tree/main/multimodal-rag-project/tests/embeddings/comparison.txt).

Highlights are the following (showing ~5e-4 deltas between the online and offline reference implementations):
```
Comparing VLLM Local Python (default) to vllm_openai_client_class:
     text_embeddings
     Max delta is 0.00014565398651933403
         [0.00013408 0.00013671 0.00012281 0.00014565 0.0001183  0.00011389] 

     image_embeddings
     Max delta is 0.0011019243880723284
         [0.00045613 0.00110192 0.0008799  0.00099878 0.0004261  0.00041882] 

     joint_embeddings
     Max delta is 0.0010816162411356744
         [0.00033086 0.00108162 0.00054624 0.00050759 0.00037544 0.00060869] 

     similarities
     Max delta is 0.0034334032552078564

pcai_vllm_openai_client Similarities:
     [[0.6686991  0.07171043 0.10108448 0.09070223 0.2119613  0.20400945]
     [0.00426799 0.5393116  0.41527759 0.26455841 0.0581812  0.16097312]
     [0.0780104  0.44381079 0.59721585 0.27857528 0.05747533 0.26933239]
     [0.09063973 0.19847573 0.26562265 0.64024973 0.2008373  0.04389279]
     [0.03512282 0.03478945 0.05370109 0.14362053 0.64233368 0.02691621]
     [0.18644896 0.13883967 0.21451706 0.16199826 0.12045625 0.6061989 ]] 
```

Hugging face shows non-negligible deviations for multimodal data, but the same rankings:

```
Comparing VLLM Local Python (default) to hugging_face:
     text_embeddings
     Max delta is 0.00026530458182660865
         [0.00022505 0.00024022 0.00019732 0.0002653  0.00021439 0.0001765 ] 

     image_embeddings
     Max delta is 0.14167977391919606
         [0.10464902 0.09484503 0.10432112 0.07152458 0.13629554 0.14167977] 

     joint_embeddings
     Max delta is 0.2115641036667637
         [0.17842363 0.17987777 0.11940851 0.15617061 0.2115641  0.20871015] 

     similarities
     Max delta is 0.08389077190705096

sentence_transformers Similarities:
     [[ 0.6649995   0.0837079   0.08595564  0.09187388  0.17617837  0.15345925]
     [-0.0013407   0.53528345  0.41411376  0.2389209   0.07588128  0.14183855]
     [ 0.06392697  0.44604892  0.5861363   0.2371358   0.06698076  0.25022933]
     [ 0.09435948  0.22998855  0.26015085  0.6084062   0.19419932  0.04270551]
     [ 0.04833154  0.03338791  0.04194595  0.13536198  0.64413106  0.02030892]
     [ 0.17369986  0.13561988  0.16302061  0.13100193  0.13746795  0.5198339 ]] 
```

##### Reranker

The reranker puts cross-comparisons on a `[0,1]` scale of similarity. My test data (18 samples, (text, image, joint text-image) of 6 reference internet samples). The full results [are here](https://github.com/ai-solution-eng/internal-projects/tree/main/multimodal-rag-project/tests/reranker/comparison.txt).

Comparing the PCAI VLLM implementation used, I get reasonable ranking parity between all samples used. Note that the values are sorted, so the fact that all indices on right side are True means that the top results match across modalities. For larger sample sets, the small ranking differences will not be in the top_k, 

```
   text_image_scores

    [[ True False False  True  True  True]
     [ True  True  True  True  True  True]
     [ True  True  True  True  True  True]
     [ True  True  True False False  True]
     [False False  True  True  True  True]
     [ True  True  True  True  True  True]]

   text_joint_scores

    [[ True  True  True  True  True  True]
     [ True  True  True  True  True  True]
     [ True  True False False  True  True]
     [ True  True  True  True  True  True]
     [ True False False  True  True  True]
     [False False  True False False  True]]

   image_joint_scores

    [[ True  True  True  True  True  True]
     [ True False False  True  True  True]
     [False False  True  True  True  True]
     [ True  True False False False  True]
     [ True  True False False  True  True]
     [False False False False False  True]]
```

Example text-images comparisons (most disimilar formats). Most deltas are very tiny (e-3 to e-2). In the largest case of .15, the delta favors the PCAI implementation (.62 to .47).

```
     Comparison scores vs. Base

    [[0.66415554 0.00757105 0.00800799 0.01477562 0.04203465 0.01772607]	   [[0.60803866 0.00609641 0.01068634 0.00885004 0.03788937 0.01508573]
     [0.00862418 0.43000829 0.23574413 0.05750899 0.01597923 0.03040552]	    [0.00540366 0.3943671  0.27404019 0.04759074 0.01103228 0.02306193]
     [0.02296895 0.24127614 0.62644047 0.07682119 0.01710314 0.04693875]	    [0.02430303 0.21788117 0.47401401 0.063894   0.02252277 0.05753885]
     [0.00790747 0.02349887 0.0321403  0.73150259 0.04074781 0.01475688]	    [0.00618581 0.02113354 0.03200972 0.72511709 0.0272368  0.01342753]
     [0.01096019 0.00863476 0.00703489 0.02172287 0.67634881 0.01539296]	    [0.0107048  0.00757734 0.00777538 0.01378601 0.61464477 0.01216942]
     [0.02842794 0.02109413 0.03252943 0.07569218 0.04111554 0.73441148]]	    [0.03095311 0.01695349 0.03262819 0.04642235 0.03784011 0.73697138]]

     delta matrix, sorted by top score

    [[1.47464033e-03 5.92557713e-03 2.67834961e-03 2.64034513e-03 4.14528698e-03 5.61168790e-02]
     [3.22051393e-03 4.94694710e-03 7.34359026e-03 9.91825759e-03 3.82960588e-02 3.56411934e-02]
     [5.41963428e-03 1.33408420e-03 1.06000975e-02 1.29271969e-02 2.33949721e-02 1.52426451e-01]
     [1.72166387e-03 1.32934470e-03 2.36533023e-03 1.35110077e-02 1.30586326e-04 6.38550520e-03]
     [1.05742272e-03 7.40490388e-04 2.55386345e-04 3.22353654e-03 7.93685578e-03 6.17040396e-02]
     [4.14064154e-03 2.52517313e-03 9.87611711e-05 3.27542797e-03 2.92698219e-02 2.55990028e-03]]
``