#!/usr/bin/env python3
"""Full RAG pipeline: ingest PDFs/photos/videos into Qdrant and query.

Usage
-----
    # Ingest files and run one query
    python tests/full_pipeline/run_pipeline.py \\
        --query "What is the capital of France?" \\
        docs/*.pdf --images photos/kitty.jpg --videos demo.mp4

    # Interactive loop
    python tests/full_pipeline/run_pipeline.py --interactive docs/*.pdf
"""

import argparse
import glob
from pathlib import Path

from multimodal_rag import MultiModalRAGSystem
from multimodal_rag.input_processing import (
    ImageProcessor,
    PDFProcessor,
    VideoProcessor,
)
from multimodal_rag.utils.logging_utils import setup_logger
from multimodal_rag.utils.pcai_models import (
    deepseek_v4_flash_280B,
    qwen3_vl_8B,
    qwen3_vl_reranker_8B,
    gemma4_31B,
    cohere_transcribe_3_2b,
)

setup_logger(level="VERBOSE")


# ---------------------------------------------------------------------------
# Ingest helpers — return processed docs without storing
# ---------------------------------------------------------------------------


def process_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    proc = PDFProcessor()
    chunks = proc.extract_chunks(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"  {Path(pdf_path).name}: {len(chunks)} chunks")
    return chunks


def process_image(image_path: str, max_pixels: int = 720 * 720, caption: str = "") -> list[dict]:
    proc = ImageProcessor(max_pixels=max_pixels)
    doc = proc.process(image_path, caption=caption)
    print(f"  {Path(image_path).name}: processed")
    return [doc]


def process_video(
    video_path: str,
    fps: float = 1.0,
    max_pixels: int = 720 * 720,
    total_pixels: int = 0,
    target_frames: int = 30,
    caption: str = "",
) -> list[dict]:
    proc = VideoProcessor(
        fps=fps,
        max_pixels=max_pixels,
        total_pixels=total_pixels,
        target_frames=target_frames,
    )
    docs = proc.process(video_path, caption=caption)
    print(f"  {Path(video_path).name}: {len(docs)} segments")
    return docs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal RAG Pipeline")
    parser.add_argument(
        "--qdrant-path",
        default="./qdrant_storage",
        help="Qdrant local storage directory",
    )
    parser.add_argument("--collection", default="documents", help="Qdrant collection name")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the Qdrant collection")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Target character count per PDF chunk",
    )
    parser.add_argument("pdfs", nargs="*", default=[], help="PDF files / glob patterns to ingest")
    parser.add_argument("--images", nargs="*", default=[], help="Image files to ingest")
    parser.add_argument("--videos", nargs="*", default=[], help="Video files to ingest")
    parser.add_argument("--query", default=None, help="Run a single query and exit")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive query loop after ingestion",
    )
    parser.add_argument("--route", action="store_true", help="Enable query routing")
    args = parser.parse_args()

    """
    args_dict = dict(
        qdrant_path='./qdrant_storage',
        collection='documents',
        reset=False,
        chunk_size=1000,
        pdfs=(
            '/home/andrew/Documents/LLM_Papers/*.pdf',
            '/home/andrew/Documents/Math/Papers/*/*.pdf'
        ),
        images=[
            '/home/andrew/Pictures/Camera/2026.02.01.Gabes.First.Haircut/DSC01363.JPG',
            '/home/andrew/Pictures/Camera/2026.02.01.Gabes.First.Haircut/DSC01364.JPG',
            '/home/andrew/Pictures/Camera/2026.02.01.Gabes.First.Haircut/DSC01365.JPG',
            '/home/andrew/Pictures/Camera/2026.02.01.Gabes.First.Haircut/DSC01366.JPG',
            '/home/andrew/Pictures/Camera/2026.02.01.Gabes.First.Haircut/DSC01367.JPG',
            '/home/andrew/Pictures/Camera/2026.02.01.Gabes.First.Haircut/DSC01368.JPG',
            '/home/andrew/Pictures/Camera/2026.02.01.Gabes.First.Haircut/DSC01369.JPG',
            '/home/andrew/Pictures/Camera/2026.02.01.Gabes.First.Haircut/DSC01370.JPG',
            '/home/andrew/Pictures/Camera/2026.02.01.Gabes.First.Haircut/DSC01371.JPG',
            '/home/andrew/Pictures/Camera/2026.02.01.Gabes.First.Haircut/DSC01372.JPG',
            '/home/andrew/Pictures/Camera/2026.02.22.Snow.Fun.Gabe/DSC01373.JPG',
            '/home/andrew/Pictures/Camera/2026.02.22.Snow.Fun.Gabe/DSC01374.JPG',
            '/home/andrew/Pictures/Camera/2026.02.22.Snow.Fun.Gabe/DSC01375.JPG',
            '/home/andrew/Pictures/Camera/2026.02.22.Snow.Fun.Gabe/DSC01378.JPG',
            '/home/andrew/Pictures/Camera/2026.02.22.Snow.Fun.Gabe/DSC01379.JPG',
            '/home/andrew/Pictures/Camera/2026.02.22.Snow.Fun.Gabe/DSC01380.JPG',
            '/home/andrew/Pictures/Camera/2026.02.22.Snow.Fun.Gabe/DSC01381.JPG',
            '/home/andrew/Pictures/Camera/2024.05.06.Annas.Photo.Shoot/DSC00915.JPG',
            '/home/andrew/Pictures/Camera/2024.05.06.Annas.Photo.Shoot/DSC00917.JPG',
            '/home/andrew/Pictures/Camera/2024.05.06.Annas.Photo.Shoot/DSC00914.JPG',
            '/home/andrew/Pictures/Camera/2024.05.06.Annas.Photo.Shoot/DSC00921.JPG',
            '/home/andrew/Pictures/Camera/2024.05.06.Annas.Photo.Shoot/DSC00910.JPG',
            '/home/andrew/Pictures/Camera/2024.05.06.Annas.Photo.Shoot/DSC00912.JPG',
            '/home/andrew/Pictures/Camera/2024.05.06.Annas.Photo.Shoot/DSC00920.JPG',
            '/home/andrew/Pictures/Camera/2024.05.06.Annas.Photo.Shoot/DSC00913.JPG',
            '/home/andrew/Pictures/Camera/2024.05.06.Annas.Photo.Shoot/DSC00916.JPG',
            '/home/andrew/Pictures/Camera/2024.05.06.Annas.Photo.Shoot/DSC00919.JPG',
            '/home/andrew/Pictures/Camera/2024.05.06.Annas.Photo.Shoot/DSC00918.JPG',
            '/home/andrew/Pictures/Camera/2024.05.06.Annas.Photo.Shoot/DSC00909.JPG',
            '/home/andrew/Pictures/Camera/2024.05.06.Annas.Photo.Shoot/DSC00911.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01169.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01179.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01183.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01176.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01180.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01164.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01160.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01173.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01166.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01175.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01184.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01177.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01155.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01187.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01157.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01158.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01161.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01171.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01181.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01182.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01174.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01156.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01185.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01178.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01162.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01192.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01189.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01168.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01165.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01186.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01163.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01172.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01188.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01190.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01191.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01170.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01159.JPG',
            '/home/andrew/Pictures/Camera/2025.09.08.Acadia.Jordans.Pond/DSC01167.JPG'],
        videos=(
            '/home/andrew/Code/HPE/MultimodalRAG/uploads_for_test_dataset/VID20250304160117_FirstHighFive.mp4',
            '/home/andrew/Code/HPE/MultimodalRAG/uploads_for_test_dataset/VID20260517153922.mp4'
        ),
        query=None,
        interactive=False,
        route=False
    )
    from argparse import Namespace
    args = Namespace(**args_dict)
    """

    # -- Reset collection if requested ------------------------------------------
    if args.reset:
        from qdrant_client import QdrantClient

        QdrantClient(path=args.qdrant_path).delete_collection(args.collection)
        print(f"Reset: deleted collection '{args.collection}'")

    # -- Build RAG system (vector_store dict = auto-create local Qdrant) --------
    print("Building RAG system ...")
    rag = MultiModalRAGSystem(
        llm=deepseek_v4_flash_280B,
        embedder=qwen3_vl_8B,
        reranker=qwen3_vl_reranker_8B,
        vlm=gemma4_31B,
        asr=cohere_transcribe_3_2b,
        caption_video=True,
        remote=True,
        vector_store={
            "qdrant_path": args.qdrant_path,
            "collection_name": args.collection,
        },
    )
    print(f"  LLM:      {rag.llm.model_name}")
    print(f"  Embedder: {rag.embedder.model_name}")
    print(f"  Reranker: {rag.reranker.model_name if rag.reranker else 'none'}")
    print(f"  VLM:      {rag.vlm.model_name if rag.vlm else 'none'}")
    print(f"  ASR:      {rag.asr.model_name if rag.asr else 'none'}")
    print(f"  Qdrant:   {args.qdrant_path}  ({args.collection})")

    # -- Collect all documents ------------------------------------------------
    mpk = rag.embedder.mm_processor_kwargs
    max_pixels = mpk.get("max_pixels", 720 * 720)
    fps = mpk.get("fps", 1.0)
    total_pixels = mpk.get("total_pixels", 0)
    chunk_size = getattr(rag.embedder, "chunk_size", 1000)
    chunk_overlap = getattr(rag.embedder, "chunk_overlap", 200)

    all_docs: list[str | dict] = []
    tasks, score = [], 0.0

    for pattern in args.pdfs:
        for pdf_path in sorted(glob.glob(pattern)):
            elements = process_pdf(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_docs.extend(elements)
            score += len(elements)
            if score >= 128:
                print(f"Executing tasks at pdf pattern {pattern}.")
                tasks.append(rag.add_to_vector_store(all_docs))
                all_docs = []
                score = 0
                print("Execution complete!.\n")

    for img_path in args.images or []:
        elements = process_image(img_path, max_pixels=max_pixels)
        all_docs.extend(elements)
        score += 5 * len(elements)
        if score >= 128:
            print(f"Executing tasks at image {img_path}.")
            tasks.append(rag.add_to_vector_store(all_docs))
            all_docs = []
            score = 0
            print("Execution complete!.\n")

    for vid_path in args.videos or []:
        elements = process_video(vid_path, fps=fps, max_pixels=max_pixels, total_pixels=total_pixels)
        all_docs.extend(elements)
        score += 12.8 * len(elements)
        if score >= 128:
            tasks.append(rag.add_to_vector_store(all_docs))
            all_docs = []
            score = 0
            print("Execution complete!.\n")

    if all_docs:
        print(f"\nStoring {len(all_docs)} documents in vector store ...")
        rag.add_to_vector_store(all_docs)
        print("Done.\n")

    """
    def run_call(query: str, **kwargs):
        print(query)
        print('\n', '#'*40, '\n')
        print('Logging Info:')
        output = rag.generate(query, **kwargs)
        print('\n', '#'*40, '\n')
        print(output)
        print('\n', '#'*120, '\n')

    run_call('Can you find data of or about a kid getting his hair cut?', route=True)
    run_call(
        'Can you find and describe images of a child climbing through a snow tunnel?',
        route=True,
        use_reranker=True,
    )
    run_call(
        'Can you describe to me what the most important new approaches'
        ' were in the Deepseek V4 Flash family of models',
        route=True,
        use_reranker=True,
        top_k=20,
        reranker_top_k=5,
    )
    run_call('Can you show me videos of an old video game?', route=True, use_reranker=True, top_k=10, reranker_top_k=5)
    """

    # -- Query ------------------------------------------------------------------
    if args.query:
        print(f"Query: {args.query}")
        print(f"Answer: {rag.generate(args.query, route=args.route)}\n")

    if args.interactive:
        print("Interactive mode. Type your queries (or 'quit' to exit).\n")
        while True:
            try:
                q = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("quit", "exit"):
                break
            print(rag.generate(q, route=args.route))
            print()


if __name__ == "__main__":
    main()
