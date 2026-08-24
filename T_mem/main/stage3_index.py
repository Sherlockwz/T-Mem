"""Stage 3: Memory-graph index building (BM25 + embedding).

Pipeline: memory graphs → item / scene / topic indexing → save.
"""

import sys
import json
import pickle
import asyncio
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TimeElapsedColumn, TimeRemainingColumn
)
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from T_mem.bootstrap import patch_providers  # noqa: E402
patch_providers()

from T_mem.config import ExperimentConfig
from T_mem.llm.embedding_provider import EmbeddingProvider

console = Console()


def ensure_nltk_data():
    """Ensure NLTK tokenizer + stopwords data are available."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def tokenize(text: str, stemmer: PorterStemmer, stop_words: set) -> List[str]:
    """NLTK tokenize + stem + stopword filter."""
    if not text:
        return []

    tokens = word_tokenize(text.lower())

    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token.isalpha() and len(token) >= 2 and token not in stop_words
    ]

    return processed_tokens


def build_item_searchable_text(item: Dict[str, Any]) -> str:
    """Build weighted BM25 searchable text for a memory item.

    Weighting: content ×3, query_patterns ×2, keywords/temporal/spatial ×1.
    """
    parts = []

    if item.get("content"):
        parts.extend([item["content"]] * 3)

    if item.get("query_patterns"):
        query_text = " ".join(item["query_patterns"])
        parts.extend([query_text] * 2)

    keywords = item.get("keywords") or item.get("tags") or []
    if keywords:
        if isinstance(keywords, list):
            parts.append(" ".join(keywords))
        else:
            parts.append(str(keywords))

    if item.get("temporal"):
        parts.append(str(item["temporal"]))
    if item.get("spatial"):
        parts.append(str(item["spatial"]))

    return " ".join(str(part) for part in parts if part)


def build_scene_searchable_text(scene: Dict[str, Any]) -> str:
    """Build weighted BM25 searchable text for a scene.

    Weighting: subject ×3, summary ×2, scene_description ×1, keywords ×1.
    """
    parts = []

    if scene.get("subject"):
        parts.extend([scene["subject"]] * 3)

    if scene.get("summary"):
        parts.extend([scene["summary"]] * 2)

    if scene.get("scene_description"):
        parts.append(scene["scene_description"])

    if scene.get("keywords"):
        if isinstance(scene["keywords"], list):
            parts.append(" ".join(scene["keywords"]))
        else:
            parts.append(str(scene["keywords"]))

    return " ".join(str(part) for part in parts if part)


def build_topic_searchable_text(topic: Dict[str, Any]) -> str:
    """Build BM25 searchable text for a topic.

    New schema: `topic["summary"]` is already the pre-weighted BM25 text
    (title*2 + keywords*1) produced by TopicExtractor; return it verbatim.

    Legacy fallback: if the topic dict still carries raw `title` / `keywords`
    (e.g. replaying an old JSON), rebuild the BM25 text on-the-fly
    using the same title*2 + keywords*1 weighting so BM25 semantics stay consistent.
    """
    summary = topic.get("summary", "") or ""
    legacy_title = topic.get("title")
    legacy_keywords = topic.get("keywords")

    # New schema.
    if not legacy_title and not legacy_keywords:
        return summary.strip()

    # Legacy path: reconstruct title*2 + keywords*1 on the fly.
    parts: list = []
    if legacy_title:
        parts.extend([legacy_title] * 2)
    if legacy_keywords:
        if isinstance(legacy_keywords, list):
            kw_text = " ".join(legacy_keywords)
        else:
            kw_text = str(legacy_keywords)
        if kw_text:
            parts.append(kw_text)
    return " ".join(str(p) for p in parts if p)


def build_bm25_index_for_single_conv(
    conv_id: int,
    data_dir: Path,
    bm25_save_dir: Path,
    stemmer: PorterStemmer,
    stop_words: set,
    skip_existing: bool = True
) -> tuple[bool, str]:
    """Build BM25 index for a single conversation's memory graph.

    Returns (success, status) where status ∈ {"completed", "skipped", "failed"}.
    """
    try:
        output_path = bm25_save_dir / f"memory_graph_bm25_index_conv_{conv_id}.pkl"
        if skip_existing and output_path.exists():
            return (True, "skipped")

        memory_graph_file = data_dir / f"memory_graph_conv_{conv_id}.json"
        if not memory_graph_file.exists():
            return (False, "failed")

        with open(memory_graph_file, "r", encoding="utf-8") as f:
            memory_graph = json.load(f)

        # ===== 1. Build index for items =====
        items = memory_graph.get("items", {})
        item_corpus = []
        item_docs = []

        for item_id, item_data in items.items():
            item_docs.append({
                "id": item_id,
                "type": "item",
                "data": item_data
            })
            searchable_text = build_item_searchable_text(item_data)
            tokenized_text = tokenize(searchable_text, stemmer, stop_words)
            item_corpus.append(tokenized_text)

        # ===== 2. Build index for scenes =====
        scenes = memory_graph.get("scenes", {})
        scene_corpus = []
        scene_docs = []

        for scene_id, scene_data in scenes.items():
            scene_docs.append({
                "id": scene_id,
                "type": "scene",
                "data": scene_data
            })
            searchable_text = build_scene_searchable_text(scene_data)
            tokenized_text = tokenize(searchable_text, stemmer, stop_words)
            scene_corpus.append(tokenized_text)

        # ===== 3. Build index for topics =====
        topics = memory_graph.get("topics", {})
        topic_corpus = []
        topic_docs = []

        for topic_id, topic_data in topics.items():
            topic_docs.append({
                "id": topic_id,
                "type": "topic",
                "data": topic_data
            })
            searchable_text = build_topic_searchable_text(topic_data)
            tokenized_text = tokenize(searchable_text, stemmer, stop_words)
            topic_corpus.append(tokenized_text)

        # ===== 4. Build unified BM25 index =====
        all_corpus = item_corpus + scene_corpus + topic_corpus
        all_docs = item_docs + scene_docs + topic_docs

        if not all_corpus:
            console.print(f"  [yellow][!] Conversation {conv_id}: no documents, skipping index creation[/yellow]")
            return False

        bm25 = BM25Okapi(all_corpus)

        # ===== 5. Save index =====
        index_data = {
            "bm25": bm25,
            "docs": all_docs,
            "item_count": len(item_docs),
            "scene_count": len(scene_docs),
            "topic_count": len(topic_docs)
        }

        with open(output_path, "wb") as f:
            pickle.dump(index_data, f)

        return (True, "completed")

    except Exception as e:
        console.print(f"  [red][X] Conversation {conv_id}: BM25 index building failed - {e}[/red]")
        return (False, "failed")


async def build_bm25_index_for_memory_graph(
    config: ExperimentConfig,
    data_dir: Path,
    bm25_save_dir: Path,
    progress: Progress = None,
    max_workers: int = 10,
    skip_existing: bool = True
):
    """Parallel BM25 index building for all conversations."""
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Starting Memory-Graph BM25 Index Building (Parallel Mode)[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    console.print(f"[bold]Skip existing:[/bold] {'Yes' if skip_existing else 'No'}\n")

    console.print("Ensuring NLTK data is available...")
    ensure_nltk_data()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    semaphore = asyncio.Semaphore(max_workers)

    async def process_with_semaphore(conv_id: int, task_id: int):
        async with semaphore:
            if progress:
                progress.start_task(task_id)
                progress.update(task_id, status="processing")

            loop = asyncio.get_event_loop()
            success, status = await loop.run_in_executor(
                None,
                build_bm25_index_for_single_conv,
                conv_id,
                data_dir,
                bm25_save_dir,
                stemmer,
                stop_words,
                skip_existing
            )

            if progress:
                if status == "skipped":
                    progress.update(task_id, status="[cyan]exists[/cyan]", completed=1)
                elif status == "completed":
                    progress.update(task_id, status="[green]done[/green]", completed=1)
                else:
                    progress.update(task_id, status="[yellow]skipped[/yellow]", completed=1)

            return (conv_id, success, status)

    if progress:
        tasks = []
        for i in range(config.num_conv):
            task_id = progress.add_task(
                f"[cyan]BM25 Index - Conversation {i}[/cyan]",
                total=1,
                status="waiting",
                start=False
            )
            tasks.append((i, task_id))

        coroutines = [process_with_semaphore(conv_id, task_id) for conv_id, task_id in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
    else:
        coroutines = []
        for i in range(config.num_conv):
            async def simple_process(conv_id):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    build_bm25_index_for_single_conv,
                    conv_id,
                    data_dir,
                    bm25_save_dir,
                    stemmer,
                    stop_words,
                    skip_existing
                )
            coroutines.append(simple_process(i))
        results = await asyncio.gather(*coroutines, return_exceptions=True)

    # Summarize results
    completed_count = 0
    skipped_count = 0
    failed_count = 0
    for r in results:
        if isinstance(r, tuple) and len(r) >= 3:
            status = r[2]
            if status == "skipped":
                skipped_count += 1
            elif status == "completed":
                completed_count += 1
            else:
                failed_count += 1
        elif isinstance(r, Exception):
            failed_count += 1

    console.print("\n[bold green][OK] BM25 index building complete[/bold green]")
    console.print(f"    New: {completed_count}, Skipped: {skipped_count}, Failed: {failed_count}")


def build_embedding_index_for_single_conv(
    conv_id: int,
    data_dir: Path,
    emb_save_dir: Path,
    embedding_provider: EmbeddingProvider,
    batch_size: int = 256,
    skip_existing: bool = True
) -> tuple[bool, str]:
    """Build vector embedding index for a single conversation's graph.

    Steps: (1) embed node texts with BGE-M3, (4) save node embeddings.
    Returns (success, status \u2208 {\"completed\", \"skipped\", \"failed\"}).
    """
    try:
        output_path = emb_save_dir / f"memory_graph_embedding_index_conv_{conv_id}.pkl"
        output_path_enhanced = emb_save_dir / f"memory_graph_embedding_index_enhanced_conv_{conv_id}.pkl"
        if skip_existing and output_path.exists() and output_path_enhanced.exists():
            return (True, "skipped")

        memory_graph_file = data_dir / f"memory_graph_conv_{conv_id}.json"
        if not memory_graph_file.exists():
            return (False, "failed")

        with open(memory_graph_file, "r", encoding="utf-8") as f:
            memory_graph = json.load(f)

        # ===== Step 1: Collect node texts and generate initial node embeddings =====
        print("  [Step 1] Generating initial node embeddings...")

        texts_to_embed = []
        node_text_map = []  # Record node info for each text

        # 1.1 Item nodes
        items = memory_graph.get("items", {})
        for item_id, item_data in items.items():
            parts = []

            if item_data.get("content"):
                parts.append(item_data["content"])

            if item_data.get("query_patterns"):
                parts.append(" ".join(item_data["query_patterns"]))

            keywords = item_data.get("keywords") or item_data.get("tags") or []
            if keywords:
                if isinstance(keywords, list):
                    parts.append(" ".join(keywords))
                else:
                    parts.append(str(keywords))

            if item_data.get("temporal"):
                parts.append(str(item_data["temporal"]))

            if item_data.get("spatial"):
                parts.append(str(item_data["spatial"]))

            item_text = " ".join(parts)
            if item_text.strip():
                texts_to_embed.append(item_text)
                node_text_map.append({
                    "node_type": "item",
                    "node_id": item_id,
                    "data": item_data
                })

        # 1.2 Scene nodes
        scenes = memory_graph.get("scenes", {})
        for scene_id, scene_data in scenes.items():
            parts = []
            if scene_data.get("subject"):
                parts.append(scene_data["subject"])
            if scene_data.get("summary"):
                parts.append(scene_data["summary"])
            if scene_data.get("scene_description"):
                parts.append(scene_data["scene_description"])

            if parts:
                scene_text = " ".join(parts)
                texts_to_embed.append(scene_text)
                node_text_map.append({
                    "node_type": "scene",
                    "node_id": scene_id,
                    "data": scene_data
                })

        # 1.3 Topic nodes
        topics = memory_graph.get("topics", {})
        for topic_id, topic_data in topics.items():
            # `topic_data["summary"]` is the pre-weighted retrieval text
            # (title*2 + keywords*1). We feed it verbatim to the embedder
            # under the new-schema path to avoid double-weighting title.
            #
            # Legacy path (for old JSONs that still carry
            # separate `title`/`keywords` fields): fall back to
            # title + keywords concatenation.
            parts = []
            legacy_title = topic_data.get("title")
            legacy_keywords = topic_data.get("keywords")
            topic_summary = topic_data.get("summary")
            if not legacy_title and not legacy_keywords:
                # New-schema path.
                if topic_summary:
                    parts.append(topic_summary)
            else:
                # Legacy-schema path.
                if legacy_title:
                    parts.append(legacy_title)
                if legacy_keywords:
                    if isinstance(legacy_keywords, list):
                        parts.append(" ".join(legacy_keywords))
                    else:
                        parts.append(str(legacy_keywords))

            if parts:
                topic_text = " ".join(parts)
                texts_to_embed.append(topic_text)
                node_text_map.append({
                    "node_type": "topic",
                    "node_id": topic_id,
                    "data": topic_data
                })

        if not texts_to_embed:
            return (False, "failed")

        # Batch generate node embeddings
        all_node_embeddings = []
        for j in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[j:j+batch_size]
            batch_embeddings = embedding_provider.embed(batch_texts)
            all_node_embeddings.extend(batch_embeddings)

        # Build node ID to embedding mapping
        node_embeddings = {}  # {(node_type, node_id): embedding}
        embedding_dim = None
        for node_info, embedding in zip(node_text_map, all_node_embeddings):
            key = (node_info["node_type"], node_info["node_id"])
            emb_array = np.array(embedding)
            node_embeddings[key] = emb_array
            if embedding_dim is None:
                embedding_dim = len(emb_array)

        # ===== Step 2 & 3 (removed): graph-edge embeddings + node update =====
        # Node embeddings are raw BGE-M3 output from Step 1; no propagation.
        updated_node_embeddings = dict(node_embeddings)

        # ===== Step 4: Organize and save all embeddings (nodes only) =====

        embedding_index_new = {
            "nodes": {},
            "metadata": {
                "num_nodes": len(updated_node_embeddings),
            }
        }

        # List format for existing retrieval code
        embedding_index = []

        for (node_type, node_id), embedding in updated_node_embeddings.items():
            if node_type not in embedding_index_new["nodes"]:
                embedding_index_new["nodes"][node_type] = {}

            if node_type == "item":
                original_data = items.get(node_id, {})
            elif node_type == "scene":
                original_data = scenes.get(node_id, {})
            elif node_type == "topic":
                original_data = topics.get(node_id, {})
            else:
                original_data = {}

            embedding_index_new["nodes"][node_type][node_id] = {
                "embedding": embedding.tolist(),
                "data": original_data,
            }

            embedding_index.append({
                "type": node_type,
                "id": node_id,
                "field": "combined",
                "embedding": embedding.tolist(),
                "data": original_data
            })

        output_path = emb_save_dir / f"memory_graph_embedding_index_conv_{conv_id}.pkl"
        emb_save_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(embedding_index, f)

        with open(output_path_enhanced, "wb") as f:
            pickle.dump(embedding_index_new, f)

        return (True, "completed")

    except Exception as e:
        console.print(f"  [red][X] Conversation {conv_id}: Embedding index building failed - {e}[/red]")
        return (False, "failed")


async def build_embedding_index_for_memory_graph(
    config: ExperimentConfig,
    data_dir: Path,
    emb_save_dir: Path,
    progress: Progress = None,
    max_workers: int = 10,
    skip_existing: bool = True
):
    """Parallel embedding index building for all conversations."""
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Starting Vector Embedding Index Building (Parallel Mode)[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    console.print(f"[bold]Skip existing:[/bold] {'Yes' if skip_existing else 'No'}\n")

    embedding_provider = EmbeddingProvider(
        base_url=config.embedding_config["base_url"],
        model_name=config.embedding_config["model_name"]
    )
    BATCH_SIZE = 256

    semaphore = asyncio.Semaphore(max_workers)

    async def process_with_semaphore(conv_id: int, task_id: int):
        async with semaphore:
            if progress:
                progress.start_task(task_id)
                progress.update(task_id, status="processing")

            loop = asyncio.get_event_loop()
            success, status = await loop.run_in_executor(
                None,
                build_embedding_index_for_single_conv,
                conv_id,
                data_dir,
                emb_save_dir,
                embedding_provider,
                BATCH_SIZE,
                skip_existing
            )

            if progress:
                if status == "skipped":
                    progress.update(task_id, status="[cyan]exists[/cyan]", completed=1)
                elif status == "completed":
                    progress.update(task_id, status="[green]done[/green]", completed=1)
                else:
                    progress.update(task_id, status="[red]failed[/red]", completed=1)

            return (conv_id, success, status)

    if progress:
        tasks = []
        for i in range(config.num_conv):
            task_id = progress.add_task(
                f"[cyan]Embedding Index - Conversation {i}[/cyan]",
                total=1,
                status="waiting",
                start=False
            )
            tasks.append((i, task_id))

        coroutines = [process_with_semaphore(conv_id, task_id) for conv_id, task_id in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
    else:
        coroutines = []
        for i in range(config.num_conv):
            async def simple_process(conv_id):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    build_embedding_index_for_single_conv,
                    conv_id,
                    data_dir,
                    emb_save_dir,
                    embedding_provider,
                    BATCH_SIZE,
                    skip_existing
                )
            coroutines.append(simple_process(i))
        results = await asyncio.gather(*coroutines, return_exceptions=True)

    # Summarize results
    completed_count = 0
    skipped_count = 0
    failed_count = 0
    for r in results:
        if isinstance(r, tuple) and len(r) >= 3:
            status = r[2]
            if status == "skipped":
                skipped_count += 1
            elif status == "completed":
                completed_count += 1
            else:
                failed_count += 1
        elif isinstance(r, Exception):
            failed_count += 1

    console.print("\n[bold green][OK] Embedding index building complete[/bold green]")
    console.print(f"    New: {completed_count}, Skipped: {skipped_count}, Failed: {failed_count}")


async def main():
    """Build BM25 + embedding indexes for all conversations."""
    config = ExperimentConfig()

    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Stage 3: Memory Graph Index Building[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")

    memory_graph_dir = config.memory_graph_dir()
    bm25_save_dir = config.bm25_index_dir()
    emb_save_dir = config.vectors_dir()

    bm25_save_dir.mkdir(parents=True, exist_ok=True)
    emb_save_dir.mkdir(parents=True, exist_ok=True)

    max_concurrent_tasks = 1

    # True: skip existing files; False: force regeneration of all indexes.
    skip_existing = True

    console.print(f"[bold]Experiment name:[/bold] {config.experiment_name}")
    console.print(f"[bold]Memory graph directory:[/bold] {memory_graph_dir}")
    console.print(f"[bold]BM25 index save directory:[/bold] {bm25_save_dir}")
    console.print(f"[bold]Vector index save directory:[/bold] {emb_save_dir}")
    console.print(f"[bold]Number of conversations:[/bold] {config.num_conv}")
    console.print(f"[bold]Concurrency:[/bold] {max_concurrent_tasks}")
    console.print(f"[bold]Checkpoint resume:[/bold] {'Enabled' if skip_existing else 'Disabled'}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.completed:>3}/{task.total:<3}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TextColumn("[bold]{task.fields[status]}"),
        console=console,
        transient=False
    ) as progress:
        await build_bm25_index_for_memory_graph(
            config,
            memory_graph_dir,
            bm25_save_dir,
            progress=progress,
            max_workers=max_concurrent_tasks,
            skip_existing=skip_existing
        )

        retrieval_type = getattr(config, 'retrieval_type', 'rrf').lower()
        need_emb = retrieval_type in ('vector', 'rrf')
        if need_emb:
            await build_embedding_index_for_memory_graph(
                config,
                memory_graph_dir,
                emb_save_dir,
                progress=progress,
                max_workers=max_concurrent_tasks,
                skip_existing=skip_existing
            )

    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]All Memory-Graph Index Building Complete![/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")


if __name__ == "__main__":
    asyncio.run(main())

