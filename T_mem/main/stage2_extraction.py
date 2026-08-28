"""Stage 2: Memory-graph extraction (topic-based memory-item extraction).

Pipeline: scenes → topic extraction → item extraction (per topic) → memory graph → save.
"""

import json
import os
import sys
import asyncio
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TimeElapsedColumn, TimeRemainingColumn
)
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from T_mem.bootstrap import patch_providers  # noqa: E402
patch_providers()

from T_mem.utils.datetime_utils import from_iso_format, to_iso_format
from T_mem.llm.llm_provider import LLMProvider
from T_mem.types import Scene, RawDataType, MemoryItem, Topic
from T_mem.extractors.memory_item_extractor import (
    MemoryItemExtractor,
    MemoryItemExtractResult,
)
from T_mem.extractors.topic_extractor import (
    TopicExtractor,
    TopicExtractRequest,
    TopicExtractResult,
)
from T_mem.extractors.memory_graph_extractor import MemoryGraphExtractor
from T_mem.structure import MemoryGraph

from T_mem.config import ExperimentConfig
import dataclasses

console = Console()

MAX_EXTRACTION_RETRIES = 2


def serialize_to_json(obj):
    """Recursively serialize an object to a JSON-compatible dictionary"""
    if obj is None:
        return None
    elif isinstance(obj, datetime):
        return to_iso_format(obj)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = serialize_to_json(value)
        return result
    elif hasattr(obj, 'model_dump'):
        try:
            return obj.model_dump(mode='json')
        except:
            dumped = obj.model_dump()
            return serialize_to_json(dumped)
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: serialize_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_to_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return str(obj)


def save_item_results(
    conv_id: str,
    item_results: List[MemoryItemExtractResult],
    save_dir: Path
) -> None:
    """Persist per-topic memory-item extraction results to `items_conv_{conv_id}.json`."""
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f"items_conv_{conv_id}.json"

    data = {
        "item_results": [serialize_to_json(r) for r in item_results],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def reconstruct_item(data: dict) -> MemoryItem:
    """Reconstruct a MemoryItem object from a dictionary."""
    timestamp = None
    if data.get('timestamp'):
        try:
            timestamp = from_iso_format(data['timestamp'])
        except:
            pass

    spatial = data.get('spatial')
    if isinstance(spatial, list):
        spatial = ', '.join(str(s) for s in spatial) if spatial else None

    temporal = data.get('temporal')
    if isinstance(temporal, list):
        temporal = ', '.join(str(t) for t in temporal) if temporal else None

    return MemoryItem(
        item_id=data['item_id'],
        content=data['content'],
        scene_ids=data.get('scene_ids', []),
        topic_id=data.get('topic_id', ''),
        temporal=temporal,
        spatial=spatial,
        keywords=data.get('keywords', []),
        query_patterns=data.get('query_patterns', []),
        timestamp=timestamp
    )


def load_item_results(
    conv_id: str,
    save_dir: Path
) -> Optional[List[MemoryItemExtractResult]]:
    """Load memory-item extraction results."""
    input_file = save_dir / f"items_conv_{conv_id}.json"

    if not input_file.exists():
        return None

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        item_results: List[MemoryItemExtractResult] = []
        for entry in data.get('item_results', []):
            items = [reconstruct_item(e) for e in entry.get('items', [])]
            result = MemoryItemExtractResult(
                topic_id=entry['topic_id'],
                items=items,
                reasoning=entry.get('reasoning', '')
            )
            item_results.append(result)

        return item_results

    except Exception as e:
        console.print(f"[yellow][!] Failed to load item results: {e}[/yellow]")
        import traceback
        traceback.print_exc()
        return None


def save_token_stats(
    conv_id: str,
    topic_token_stats: Optional[Dict],
    item_token_stats: Optional[Dict],
    save_dir: Path
) -> None:
    """Save token usage statistics.

    Note: the `total` field is aggregated after Stage 6 ends, not here.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f"token_stats_conv_{conv_id}.json"

    data = {
        "conv_id": conv_id,
        "topic_extraction": topic_token_stats,
        "item_extraction": item_token_stats,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_token_stats(conv_id: str, save_dir: Path) -> Optional[Dict]:
    """Load token statistics"""
    input_file = save_dir / f"token_stats_conv_{conv_id}.json"

    if not input_file.exists():
        return None

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# Topics are fully contained within memory_graph_conv_*.json; no standalone cache.


def load_scenes_from_json(file_path: str) -> List[Scene]:
    """Load a list of Scenes from a JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        scene_dicts = json.load(f)

    scenes = []
    for scene_dict in scene_dicts:
        if "timestamp" in scene_dict and scene_dict["timestamp"]:
            ts = scene_dict["timestamp"]
            if isinstance(ts, str):
                scene_dict["timestamp"] = from_iso_format(ts)
            elif isinstance(ts, (int, float)):
                scene_dict["timestamp"] = datetime.fromtimestamp(ts)

        if "type" in scene_dict and scene_dict["type"]:
            try:
                scene_dict["type"] = RawDataType(scene_dict["type"])
            except ValueError:
                scene_dict["type"] = RawDataType.CONVERSATION

        scene = Scene(**scene_dict)
        scenes.append(scene)

    return scenes


async def extract_topics_for_scenes(
    scenes: List[Scene],
    llm_provider: LLMProvider,
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None
) -> Optional[TopicExtractResult]:
    """Extract topics for a list of Scenes."""
    if not scenes:
        return None

    console.print("  [*] Using LLM-based topic matching (TopicExtractor)")
    topic_extractor = TopicExtractor(
        llm_provider=llm_provider,
        topic_match_batch_size=10,
    )

    topic_map: Dict[str, Topic] = {}

    if len(scenes) >= 1:
        try:
            first_scene = scenes[0]
            console.print("  [*] Creating initial topic for the first scene...")

            topic = await topic_extractor._extract_new_topic([first_scene])

            if topic:
                topic_map[topic.topic_id] = topic
                console.print(f"  [+] Created initial topic: {topic.title}")
            else:
                raise RuntimeError("Topic creation for the first scene returned None")

        except Exception as e:
            console.print(f"[red][!] Topic creation for the first scene failed: {e}[/red]")
            raise RuntimeError(f"Topic creation for the first scene failed: {e}") from e

    if len(scenes) == 1:
        if topic_map:
            return TopicExtractResult(
                topics=list(topic_map.values()),
                action="merged",
            )
        return None

    for idx in range(1, len(scenes)):
        if progress and task_id is not None:
            progress.update(task_id, completed=idx - 1)

        retry_count = 0
        success = False

        while retry_count < MAX_EXTRACTION_RETRIES and not success:
            try:
                new_scene = scenes[idx]
                history_scenes = scenes[:idx]
                existing_topics = list(topic_map.values())

                topic_request = TopicExtractRequest(
                    history_scene_list=history_scenes,
                    new_scene=new_scene,
                    existing_topics=existing_topics
                )

                topic_result = await topic_extractor.extract_topic(topic_request)

                if topic_result:
                    for topic in topic_result.topics:
                        topic_map[topic.topic_id] = topic

                success = True

            except Exception as e:
                retry_count += 1
                if retry_count < MAX_EXTRACTION_RETRIES:
                    console.print(f"[yellow][!] Topic extraction failed (idx={idx}, retry {retry_count}/{MAX_EXTRACTION_RETRIES}): {e}[/yellow]")
                else:
                    console.print(f"[red][X] Topic extraction failed (idx={idx}), max retries reached: {e}[/red]")
                    break

    if progress and task_id is not None:
        progress.update(task_id, completed=len(scenes) - 1)

    if topic_map:
        return TopicExtractResult(
            topics=list(topic_map.values()),
            action="merged",
        )

    return None


async def extract_items_for_topics(
    topics: List[Topic],
    scenes: List[Scene],
    llm_provider: LLMProvider,
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None
) -> List[MemoryItemExtractResult]:
    """Extract memory items per topic; returns one MemoryItemExtractResult per topic."""
    item_extractor = MemoryItemExtractor(llm_provider=llm_provider)

    scene_map: Dict[str, Scene] = {sc.scene_id: sc for sc in scenes}

    item_results: List[MemoryItemExtractResult] = []

    for idx, topic in enumerate(topics):
        if progress and task_id is not None:
            progress.update(task_id, completed=idx)

        topic_scenes = [
            scene_map[sc_id]
            for sc_id in topic.scene_ids
            if sc_id in scene_map
        ]

        if not topic_scenes:
            console.print(f"  [yellow][!] Topic {topic.topic_id} has no associated scenes, skipping[/yellow]")
            continue

        retry_count = 0
        success = False

        while retry_count < MAX_EXTRACTION_RETRIES and not success:
            try:
                item_result = await item_extractor.extract_items(
                    topic=topic,
                    scenes=topic_scenes
                )

                if item_result:
                    item_results.append(item_result)

                success = True

            except Exception as e:
                retry_count += 1
                if retry_count < MAX_EXTRACTION_RETRIES:
                    console.print(f"[yellow][!] Topic {topic.topic_id} item extraction failed (retry {retry_count}/{MAX_EXTRACTION_RETRIES}): {e}[/yellow]")
                else:
                    console.print(f"[red][X] Topic {topic.topic_id} item extraction failed, max retries reached: {e}[/red]")
                    break

    if progress and task_id is not None:
        progress.update(task_id, completed=len(topics))

    return item_results


async def process_single_conversation(
    conv_id: str,
    scenes_file: Path,
    save_dir: Path,
    llm_provider: LLMProvider,
    progress: Optional[Progress] = None,
    conv_task_id: Optional[int] = None,
    skip_existing: bool = True,
    items_dir: Optional[Path] = None,
) -> Optional[MemoryGraph]:
    """Memory-graph extraction for a single conversation: topics → items → graph."""
    try:
        from T_mem.utils.cost_ledger import set_conv as _cost_set_conv
        _cost_set_conv(conv_id)
        output_file = save_dir / f"memory_graph_conv_{conv_id}.json"
        console.print(f"  [dim]Conversation {conv_id}: checking path {output_file}[/dim]")

        if skip_existing and output_file.exists():
            console.print(f"  [cyan][√] Conversation {conv_id}: memory graph already exists, skipping[/cyan]")
            if progress and conv_task_id is not None:
                progress.update(conv_task_id, description=f"[cyan]Conversation {conv_id}[/cyan]", status="[cyan]exists[/cyan]", completed=1)

            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    memory_graph_dict = json.load(f)
                return MemoryGraph.from_dict(memory_graph_dict)
            except Exception as e:
                console.print(f"  [yellow][!] Conversation {conv_id}: failed to load existing memory graph: {e}, will reprocess[/yellow]")

        if progress and conv_task_id is not None:
            progress.update(conv_task_id, description=f"[cyan]Conversation {conv_id}[/cyan]", status="loading data")

        # Step 1: Load Scenes
        scenes = load_scenes_from_json(str(scenes_file))
        console.print(f"  [+] Conversation {conv_id}: loaded {len(scenes)} Scenes")

        if not scenes:
            console.print(f"  [yellow][!] Conversation {conv_id}: no Scenes, skipping[/yellow]")
            return None

        # Step 2: Extract topics.
        # Topics are fully embedded inside memory_graph_conv_*.json; no
        # standalone topic cache is kept or consulted.
        if progress and conv_task_id is not None:
            progress.update(conv_task_id, status="extract topic", total=len(scenes) - 1, completed=0)

        topic_token_stats = None

        console.print(f"  [yellow]→ Conversation {conv_id}: starting topic extraction...[/yellow]")
        try:
            llm_provider.reset_accumulated_stats()

            topic_result = await extract_topics_for_scenes(
                scenes=scenes,
                llm_provider=llm_provider,
                progress=progress,
                task_id=conv_task_id
            )

            topic_token_stats = llm_provider.get_accumulated_stats()
            if topic_token_stats:
                console.print(f"  [dim]   └─ Topic extraction Token: prompt={topic_token_stats['prompt_tokens']:,}, "
                              f"completion={topic_token_stats['completion_tokens']:,}, "
                              f"total={topic_token_stats['total_tokens']:,}[/dim]")

            num_topics = len(topic_result.topics) if topic_result else 0
            console.print(f"  [+] Conversation {conv_id}: extracted {num_topics} topics")
        except Exception as topic_extract_error:
            console.print(f"[red][-] Conversation {conv_id}: topic extraction failed: {topic_extract_error}[/red]")
            import traceback
            traceback.print_exc()
            raise

        # Step 3: Extract items per topic.
        if progress and conv_task_id is not None:
            progress.update(conv_task_id, status="checking item cache", total=1, completed=0)

        item_token_stats = None

        cached_items = None
        if items_dir and items_dir.exists():
            item_cache_file = items_dir / f"items_conv_{conv_id}.json"
            if item_cache_file.exists():
                console.print(f"  [cyan]✓ Found item cache file: {item_cache_file.name}[/cyan]")
                try:
                    cached_items = load_item_results(conv_id, items_dir)
                    if cached_items is not None:
                        console.print("  [green]✓ Item cache loaded successfully[/green]")
                except Exception as e:
                    console.print(f"  [yellow]⚠ Item cache loading error: {e}, will re-extract[/yellow]")
                    cached_items = None

        if cached_items is not None:
            console.print(f"  [cyan][√] Conversation {conv_id}: using cached item results[/cyan]")
            item_results = cached_items
        else:
            topics = topic_result.topics if topic_result else []

            if not topics:
                console.print(f"  [yellow][!] Conversation {conv_id}: no topics, skipping item extraction[/yellow]")
                item_results = []
            else:
                if progress and conv_task_id is not None:
                    progress.update(conv_task_id, status="extract item", total=len(topics), completed=0)

                console.print(f"  [yellow]→ Conversation {conv_id}: extracting items based on {len(topics)} topics...[/yellow]")

                llm_provider.reset_accumulated_stats()

                item_results = await extract_items_for_topics(
                    topics=topics,
                    scenes=scenes,
                    llm_provider=llm_provider,
                    progress=progress,
                    task_id=conv_task_id
                )

                item_token_stats = llm_provider.get_accumulated_stats()
                if item_token_stats:
                    console.print(f"  [dim]   └─ Item extraction Token: prompt={item_token_stats['prompt_tokens']:,}, "
                                  f"completion={item_token_stats['completion_tokens']:,}, "
                                  f"total={item_token_stats['total_tokens']:,}[/dim]")

                total_items = sum(len(r.items) for r in item_results)
                if total_items > 0:
                    console.print(f"  [+] Conversation {conv_id}: extracted {total_items} items")
                else:
                    console.print(f"  [red][-] Conversation {conv_id}: item extraction returned 0 items[/red]")
                    if progress and conv_task_id is not None:
                        progress.update(conv_task_id, status="[yellow]items=0[/yellow]")

            if items_dir:
                save_item_results(conv_id, item_results, items_dir)
                console.print(f"  [+] Conversation {conv_id}: saved item results to items/")

        # Step 4: Build memory graph.
        if progress and conv_task_id is not None:
            progress.update(conv_task_id, status="building memory graph", total=1, completed=0)

        memory_graph_extractor = MemoryGraphExtractor()
        memory_graph = memory_graph_extractor.build_memory_graph(
            scenes=scenes,
            item_results=item_results,
            topic_extract_result=topic_result,
        )

        stats = memory_graph.get_stats()
        console.print(f"  [+] Conversation {conv_id}: built memory graph - {stats}")

        # Step 5: Save memory graph
        if progress and conv_task_id is not None:
            progress.update(conv_task_id, status="saving memory graph", completed=1)

        memory_graph_dict = memory_graph.to_dict()
        output_file = save_dir / f"memory_graph_conv_{conv_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(memory_graph_dict, f, ensure_ascii=False, indent=2)

        console.print(f"  [+] Conversation {conv_id}: saved memory graph to {output_file.name}")

        # Step 6: Save token statistics (if available).
        if topic_token_stats or item_token_stats:
            token_stats_dir = save_dir.parent / "token_stats"
            save_token_stats(conv_id, topic_token_stats, item_token_stats, token_stats_dir)

            total_prompt = (topic_token_stats or {}).get('prompt_tokens', 0) + (item_token_stats or {}).get('prompt_tokens', 0)
            total_completion = (topic_token_stats or {}).get('completion_tokens', 0) + (item_token_stats or {}).get('completion_tokens', 0)
            total_tokens = total_prompt + total_completion
            console.print(f"  [dim]   └─ Total Token: prompt={total_prompt:,}, completion={total_completion:,}, total={total_tokens:,}[/dim]")

        if progress and conv_task_id is not None:
            total_items = sum(len(r.items) for r in item_results) if item_results else 0
            if total_items > 0:
                progress.update(conv_task_id, status="[green]done[/green]", completed=1)
            else:
                progress.update(conv_task_id, status="[yellow]done (items=0)[/yellow]", completed=1)

        return memory_graph

    except Exception as e:
        console.print(f"[red][-] Conversation {conv_id} processing failed: {e}[/red]")
        if progress and conv_task_id is not None:
            progress.update(conv_task_id, status="[red]failed[/red]")
        import traceback
        traceback.print_exc()
        return None


async def process_conversation_with_semaphore(
    semaphore: asyncio.Semaphore,
    conv_id: str,
    scenes_file: Path,
    save_dir: Path,
    llm_provider: LLMProvider,
    progress: Optional[Progress] = None,
    conv_task_id: Optional[int] = None,
    skip_existing: bool = True,
    items_dir: Optional[Path] = None,
) -> tuple[str, Optional[MemoryGraph]]:
    """Conversation processing with semaphore-bounded concurrency."""
    async with semaphore:
        if progress and conv_task_id is not None:
            progress.start_task(conv_task_id)

        memory_graph = await process_single_conversation(
            conv_id=conv_id,
            scenes_file=scenes_file,
            save_dir=save_dir,
            llm_provider=llm_provider,
            progress=progress,
            conv_task_id=conv_task_id,
            skip_existing=skip_existing,
            items_dir=items_dir,
        )

        return (conv_id, memory_graph)


async def main():
    """Batch memory-graph extraction for all conversations."""
    config = ExperimentConfig()

    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Stage 2: Memory Graph Extraction[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")

    scenes_dir = config.scenes_dir()
    memory_graph_dir = config.memory_graph_dir()
    items_dir = config.items_dir()
    token_stats_dir = config.token_stats_dir()

    memory_graph_dir.mkdir(parents=True, exist_ok=True)
    items_dir.mkdir(parents=True, exist_ok=True)
    token_stats_dir.mkdir(parents=True, exist_ok=True)

    # Inter-conv concurrency: default 14 in parallel. In-conv work (TopicExtractor /
    # MemoryItemExtractor awaits) is already sequential, so this caps Stage-2
    # LLM peak. Env override to throttle under load.
    max_concurrent_tasks = int(
        os.environ.get("T_MEM_MAX_CONCURRENCY", "")
        or 14
    )
    max_concurrent_tasks = max(1, max_concurrent_tasks)
    skip_existing = True

    console.print(f"[bold]Experiment name:[/bold] {config.experiment_name}")
    console.print(f"[bold]Number of conversations:[/bold] {config.num_conv}")
    console.print(f"[bold]Concurrency:[/bold] {max_concurrent_tasks}")
    console.print(f"[bold]Skip existing:[/bold] {'Yes' if skip_existing else 'No'}")
    console.print("[bold]Pipeline:[/bold] scene → topic → item → memory graph\n")

    llm_config = config.llm_config[config.llm_service].copy()
    llm_config.pop('llm_provider', None)  # not a LLMProvider kwarg
    llm_config['enable_stats'] = True
    llm_provider = LLMProvider(**llm_config)

    console.print("[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Final Configuration[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print(f"[bold]Scene directory:[/bold] {scenes_dir}")
    console.print(f"[bold]Memory graph save directory:[/bold] {memory_graph_dir}")
    console.print(f"[bold]Item cache directory:[/bold] {items_dir}")
    console.print(f"[bold]Token statistics directory:[/bold] {token_stats_dir}")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")

    conv_files = []
    for i in range(config.num_conv):
        scene_file = scenes_dir / f"scene_list_conv_{i}.json"
        if scene_file.exists():
            conv_files.append((str(i), scene_file))
        else:
            console.print(f"[yellow][!] File not found: {scene_file}[/yellow]")

    if not conv_files:
        console.print("[red][-] No Scene files found[/red]")
        return

    console.print(f"[green][OK][/green] Found {len(conv_files)} conversation files\n")

    semaphore = asyncio.Semaphore(max_concurrent_tasks)

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

        tasks = []
        for conv_id, scene_file in conv_files:
            task_id = progress.add_task(
                f"[cyan]Conversation {conv_id}[/cyan]",
                total=1,
                status="waiting",
                start=False
            )
            tasks.append((conv_id, scene_file, task_id))

        coroutines = [
            process_conversation_with_semaphore(
                semaphore=semaphore,
                conv_id=conv_id,
                scenes_file=scene_file,
                save_dir=memory_graph_dir,
                llm_provider=llm_provider,
                progress=progress,
                conv_task_id=task_id,
                skip_existing=skip_existing,
                items_dir=items_dir,
            )
            for conv_id, scene_file, task_id in tasks
        ]

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                conv_id = tasks[i][0]
                console.print(f"[red][-] Conversation {conv_id} processing error: {result}[/red]")
                processed_results.append((conv_id, None))
            else:
                processed_results.append(result)

    successful = sum(1 for _, hg in processed_results if hg is not None)
    failed = len(processed_results) - successful

    total_token_summary = {
        'topic_extraction': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'call_count': 0, 'total_duration': 0.0},
        'item_extraction': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'call_count': 0, 'total_duration': 0.0},
        'total': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'call_count': 0, 'total_duration': 0.0}
    }

    for conv_id, _ in processed_results:
        token_stats = load_token_stats(conv_id, token_stats_dir)
        if token_stats:
            for stage in ['topic_extraction', 'item_extraction']:
                if token_stats.get(stage):
                    for key in ['prompt_tokens', 'completion_tokens', 'total_tokens', 'call_count']:
                        total_token_summary[stage][key] += token_stats[stage].get(key, 0)
                        total_token_summary['total'][key] += token_stats[stage].get(key, 0)
                    total_token_summary[stage]['total_duration'] += token_stats[stage].get('total_duration', 0.0)
                    total_token_summary['total']['total_duration'] += token_stats[stage].get('total_duration', 0.0)

    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Processing Complete[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    console.print(f"[bold green][OK] Succeeded:[/bold green] {successful}/{len(processed_results)} conversations")
    if failed > 0:
        console.print(f"[bold red][X] Failed:[/bold red] {failed}/{len(processed_results)} conversations")

    if total_token_summary['total']['total_tokens'] > 0:
        console.print("\n[bold yellow]Token Usage Summary:[/bold yellow]")
        console.print(f"  Topic extraction: prompt={total_token_summary['topic_extraction']['prompt_tokens']:,}, "
                      f"completion={total_token_summary['topic_extraction']['completion_tokens']:,}, "
                      f"total={total_token_summary['topic_extraction']['total_tokens']:,}, "
                      f"calls={total_token_summary['topic_extraction']['call_count']}")
        console.print(f"  Item extraction: prompt={total_token_summary['item_extraction']['prompt_tokens']:,}, "
                      f"completion={total_token_summary['item_extraction']['completion_tokens']:,}, "
                      f"total={total_token_summary['item_extraction']['total_tokens']:,}, "
                      f"calls={total_token_summary['item_extraction']['call_count']}")
        console.print(f"  [bold]Total: prompt={total_token_summary['total']['prompt_tokens']:,}, "
                      f"completion={total_token_summary['total']['completion_tokens']:,}, "
                      f"total={total_token_summary['total']['total_tokens']:,}, "
                      f"calls={total_token_summary['total']['call_count']}[/bold]")

        summary_file = token_stats_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "experiment_name": config.experiment_name,
                "num_conversations": len(processed_results),
                "successful": successful,
                "failed": failed,
                "token_usage": total_token_summary
            }, f, ensure_ascii=False, indent=2)
        console.print(f"\n[dim]Token statistics summary saved to: {summary_file}[/dim]")

    console.print(f"\n[bold]Memory graphs saved at:[/bold] {memory_graph_dir}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
