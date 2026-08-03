"""Stage 1: Memory extraction (scene boundary detection + scene memory generation)."""

import json
import sys
import uuid
import asyncio
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta

from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from T_mem.bootstrap import patch_providers  # noqa: E402
patch_providers()

from T_mem.config import ExperimentConfig
from T_mem.llm.llm_provider import LLMProvider
from T_mem.types import RawDataType, Scene
from T_mem.extractors.scene_extractor import RawData
from T_mem.extractors.conv_scene_extractor import (
    ConvSceneExtractor, ConvSceneExtractRequest
)
from T_mem.utils.datetime_utils import to_iso_format, from_iso_format, get_now_with_timezone

console = Console()


def parse_locomo_timestamp(timestamp_str: str) -> datetime:
    """Parse LoCoMo timestamp format (e.g. '3:00 PM on 14 March, 2024') to datetime."""
    timestamp_str = timestamp_str.replace("\\s+", " ").strip()
    return datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")


def load_locomo_raw_data(locomo_data_path: str) -> Dict[str, list]:
    """Load LoCoMo dataset; returns {conv_id: list_of_message_dicts}."""
    with open(locomo_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_data_dict = {}
    conversations = [data[i]['conversation'] for i in range(len(data))]
    print(f"   [INFO] Found {len(conversations)} conversations")

    for con_id, conversation in enumerate(conversations):
        messages = []
        session_keys = sorted(
            [key for key in conversation
             if key.startswith("session_") and not key.endswith("_date_time")]
        )

        print(f"   [INFO] Found {len(session_keys)} sessions")
        print(f"   [INFO] Speakers: {conversation.get('speaker_a', 'Unknown')} & {conversation.get('speaker_b', 'Unknown')}")

        speaker_name_to_id = {}
        for session_key in session_keys:
            session_messages = conversation[session_key]
            session_time_key = f"{session_key}_date_time"

            if session_time_key in conversation:
                session_time = parse_locomo_timestamp(conversation[session_time_key])

                for i, msg in enumerate(session_messages):
                    msg_timestamp = session_time + timedelta(seconds=i * 30)
                    iso_timestamp = to_iso_format(msg_timestamp)

                    speaker_name = msg["speaker"]
                    if speaker_name not in speaker_name_to_id:
                        speaker_name_to_id[speaker_name] = f"{speaker_name.lower().replace(' ', '_')}_{con_id}"

                    content = msg["text"]
                    if msg.get("img_url"):
                        blip_caption = msg.get("blip_caption", "an image")
                        content = f"[{speaker_name} shared an image: {blip_caption}] {content}"

                    message = {
                        "speaker_id": speaker_name_to_id[speaker_name],
                        "user_name": speaker_name,
                        "speaker_name": speaker_name,
                        "content": content,
                        "timestamp": iso_timestamp,
                        "original_timestamp": conversation[session_time_key],
                        "dia_id": msg["dia_id"],
                        "session": session_key,
                    }
                    for optional_field in ["img_url", "blip_caption", "query"]:
                        if optional_field in msg:
                            message[optional_field] = msg[optional_field]
                    messages.append(message)

        raw_data_dict[str(con_id)] = messages
        print(f"   [SUCCESS] Converted {len(messages)} messages from {len(session_keys)} sessions")

    return raw_data_dict


def convert_conversation_to_raw_data_list(conversation: list) -> List[RawData]:
    return [RawData(content=msg, data_id=str(uuid.uuid4())) for msg in conversation]


async def scene_extraction_from_conversation(
    raw_data_list: List[RawData],
    llm_provider: LLMProvider = None,
    scene_extractor: ConvSceneExtractor = None,
    smart_mask: bool = True,
    conv_id: str = None,
    progress: Progress = None,
    task_id: int = None,
) -> list:
    """Run boundary detection on a conversation and extract Scenes."""
    if scene_extractor is None:
        scene_extractor = ConvSceneExtractor(llm_provider=llm_provider)

    scene_list = []
    speakers = {
        raw_data.content["speaker_id"]
        for raw_data in raw_data_list
        if isinstance(raw_data.content, dict) and "speaker_id" in raw_data.content
    }
    history_raw_data_list = []

    total_messages = len(raw_data_list)
    smart_mask_flag = False

    for idx, raw_data in enumerate(raw_data_list):
        if progress and task_id is not None:
            progress.update(task_id, completed=idx)

        if history_raw_data_list == [] or len(history_raw_data_list) == 1:
            history_raw_data_list.append(raw_data)
            continue

        if smart_mask and len(history_raw_data_list) > 5:
            smart_mask_flag = True
        else:
            smart_mask_flag = False

        request = ConvSceneExtractRequest(
            history_raw_data_list=history_raw_data_list,
            new_raw_data_list=[raw_data],
            user_id_list=list(speakers),
            smart_mask_flag=smart_mask_flag,
        )

        for i in range(5):
            try:
                result = await scene_extractor.extract_scene(request)
                break
            except Exception as e:
                console.print(f"  [yellow][!] Conv-{conv_id} msg {idx}: retry {i+1}/5: {e}[/yellow]")
                if i == 4:
                    raise RuntimeError("Scene extraction failed after 5 retries")
                continue

        scene_result = result[0]

        if scene_result is None:
            history_raw_data_list.append(raw_data)
        elif isinstance(scene_result, Scene):
            if smart_mask_flag:
                history_raw_data_list = [history_raw_data_list[-1], raw_data]
            else:
                history_raw_data_list = [raw_data]
            scene_list.append(scene_result)
        else:
            console.print(f"  [red][ERROR] Unexpected result type: {scene_result}[/red]")
            raise RuntimeError("Scene extraction returned unexpected result")

    if progress and task_id is not None:
        progress.update(task_id, completed=total_messages)

    if history_raw_data_list:
        scene = Scene(
            type=RawDataType.CONVERSATION,
            scene_id=str(uuid.uuid4()),
            user_id_list=list(speakers),
            original_data=history_raw_data_list,
            timestamp=(scene_list[-1].timestamp) if scene_list else datetime.now(),
            summary="(pending)",
        )

        try:
            processed_data = [scene_extractor._data_process(rd) for rd in history_raw_data_list]
            processed_data = [d for d in processed_data if d is not None]
            scene = await scene_extractor._generate_scene_memory(scene, processed_data)
            scene.original_data = processed_data
        except Exception as e:
            console.print(f"  [yellow][!] Final segment scene memory generation failed: {e}[/yellow]")
            scene.original_data = [scene_extractor._data_process(rd) for rd in history_raw_data_list]
            scene.original_data = [d for d in scene.original_data if d is not None]

        scene_list.append(scene)

    return scene_list


async def process_single_conversation(
    conv_id: str,
    conversation: list,
    save_dir: Path,
    llm_provider: LLMProvider = None,
    progress_counter: dict = None,
    progress: Progress = None,
    task_id: int = None,
) -> tuple:
    """Process a single conversation and return results."""
    try:
        from T_mem.utils.cost_ledger import set_conv as _cost_set_conv
        _cost_set_conv(conv_id)
        # Conv-level idempotent skip (mirrors stage2:393 / stage3:165 idiom).
        # Enables `--resume <exp_dir>` after a crash to leave already-finished
        # conv outputs untouched and skip their LLM scene extraction. We do
        # NOT deserialise the existing scene_list_conv_*.json into Scene
        # objects because stage 2/4/5/7 read that file directly from disk;
        # only stage1 main's summary cares about the in-memory `sc_list`,
        # and a `[]` return there is harmless (the on-disk file is the
        # source of truth for downstream stages).
        output_file = save_dir / f"scene_list_conv_{conv_id}.json"
        if output_file.exists():
            console.print(
                f"  [cyan][√] Conv-{conv_id}: scene_list already exists, skipping[/cyan]"
            )
            if progress and task_id is not None:
                progress.update(
                    task_id,
                    status="[cyan]exists[/cyan]",
                    completed=len(conversation),
                )
            if progress_counter:
                progress_counter['completed'] += 1
            return conv_id, []

        if progress and task_id is not None:
            progress.update(task_id, status="Processing")

        raw_data_list = convert_conversation_to_raw_data_list(conversation)
        scene_extractor = ConvSceneExtractor(llm_provider=llm_provider)
        scene_list = await scene_extraction_from_conversation(
            raw_data_list,
            llm_provider=llm_provider,
            scene_extractor=scene_extractor,
            conv_id=conv_id,
            progress=progress,
            task_id=task_id,
        )

        for sc in scene_list:
            if hasattr(sc, 'timestamp'):
                ts = sc.timestamp
                if isinstance(ts, (int, float)):
                    sc.timestamp = datetime.fromtimestamp(ts)
                elif isinstance(ts, str):
                    sc.timestamp = from_iso_format(ts)
                elif not isinstance(ts, datetime):
                    sc.timestamp = get_now_with_timezone()

        scene_dicts = [sc.to_dict() for sc in scene_list]
        output_file = save_dir / f"scene_list_conv_{conv_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(scene_dicts, f, ensure_ascii=False, indent=2)

        if progress_counter:
            progress_counter['completed'] += 1

        return conv_id, scene_list

    except Exception as e:
        console.print(f"\n[red][ERROR] Conversation {conv_id} failed: {e}[/red]")
        if progress_counter:
            progress_counter['completed'] += 1
            progress_counter['failed'] += 1
        import traceback
        traceback.print_exc()
        return conv_id, []


async def main():
    config = ExperimentConfig()
    llm_service = config.llm_service
    dataset_path = config.dataset_path
    raw_data_dict = load_locomo_raw_data(dataset_path)

    save_dir = config.scenes_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]" + "=" * 80 + "[/bold cyan]")
    console.print("[bold cyan]Stage 1: Memory Extraction (Scene Boundary Detection)[/bold cyan]")
    console.print("[bold cyan]" + "=" * 80 + "[/bold cyan]\n")

    console.print(f"[INFO] Total conversations: {len(raw_data_dict)}", style="bold cyan")
    total_messages = sum(len(conv) for conv in raw_data_dict.values())
    console.print(f"[INFO] Total messages: {total_messages}", style="bold blue")
    console.print(f"[INFO] Save directory: {save_dir}", style="bold green")

    console.print("[INFO] Initializing LLM Provider...", style="yellow")
    console.print(f"   Model: {config.llm_config[llm_service]['model']}", style="dim")

    shared_llm_provider = LLMProvider(
        provider_type="openai",
        model=config.llm_config[llm_service]["model"],
        temperature=config.llm_config[llm_service]["temperature"],
        max_tokens=config.llm_config[llm_service]["max_tokens"],
    )

    progress_counter = {
        'total': len(raw_data_dict),
        'completed': 0,
        'failed': 0
    }

    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("*"),
        TaskProgressColumn(),
        TextColumn("*"),
        TimeElapsedColumn(),
        TextColumn("*"),
        TimeRemainingColumn(),
        TextColumn("*"),
        TextColumn("[bold blue]{task.fields[status]}"),
        console=console,
        transient=False,
        refresh_per_second=1,
    ) as progress:
        main_task = progress.add_task(
            "[bold cyan][MAIN] Total Progress",
            total=len(raw_data_dict),
            completed=0,
            status="Processing",
        )

        conversation_tasks = {}
        updated_tasks = []

        for conv_id, conversation in raw_data_dict.items():
            conv_task_id = progress.add_task(
                f"[yellow]Conv-{conv_id}",
                total=len(conversation),
                completed=0,
                status="Waiting",
            )
            conversation_tasks[conv_id] = conv_task_id

            task = process_single_conversation(
                conv_id,
                conversation,
                save_dir,
                llm_provider=shared_llm_provider,
                progress_counter=progress_counter,
                progress=progress,
                task_id=conv_task_id,
            )
            updated_tasks.append(task)

        # CRITICAL: caps the global Stage-1 LLM peak (in-conv work is already
        # serial inside ConvSceneExtractor). Env override to throttle under load.
        import os as _os
        _s1c = int(_os.environ.get("T_MEM_VENUS_MAX_WORKERS", "").strip() or 14)
        stage1_sem = asyncio.Semaphore(max(1, _s1c))

        async def run_with_completion(task, conv_id):
            async with stage1_sem:
                result = await task
                progress.update(conversation_tasks[conv_id],
                                status="[green]Done[/green]",
                                completed=progress.tasks[conversation_tasks[conv_id]].total)
                progress.update(main_task, advance=1)
                return result

        results = await asyncio.gather(*[
            run_with_completion(task, conv_id)
            for (conv_id, _), task in zip(raw_data_dict.items(), updated_tasks)
        ])

        progress.update(main_task, status="[green]Completed[/green]")

    elapsed = time.time() - start_time

    all_scenes = []
    successful = 0
    for conv_id, sc_list in results:
        if sc_list:
            successful += 1
            all_scenes.extend(sc_list)

    console.print("\n" + "=" * 60, style="dim")
    console.print("[STATS] Processing Statistics:", style="bold")
    console.print(f"   [SUCCESS] Conversations: {successful}/{len(raw_data_dict)}", style="green")
    console.print(f"   [INFO] Total scenes: {len(all_scenes)}", style="blue")
    console.print(f"   [TIME] Elapsed: {elapsed:.2f}s", style="yellow")
    console.print(f"   [TIME] Average per conversation: {elapsed / len(raw_data_dict):.2f}s", style="cyan")
    console.print("=" * 60, style="dim")

    all_dicts = [sc.to_dict() for sc in all_scenes]
    summary_file = save_dir / "scene_list_all.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_dicts, f, ensure_ascii=False, indent=2)
    console.print(f"\n[SAVE] Summary saved to: {summary_file}", style="green")

    summary = {
        "total_conversations": len(raw_data_dict),
        "successful_conversations": successful,
        "total_scenes": len(all_scenes),
        "processing_time_seconds": elapsed,
        "average_time_per_conversation": elapsed / len(raw_data_dict),
        "conversation_results": {
            conv_id: len(sc_list) for conv_id, sc_list in results
        }
    }
    summary_info_file = save_dir / "processing_summary.json"
    with open(summary_info_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    console.print(f"[SAVE] Processing summary saved to: {summary_info_file}\n", style="green")


if __name__ == "__main__":
    asyncio.run(main())
