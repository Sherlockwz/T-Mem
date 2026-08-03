"""Two-layer graph builder: aggregates scenes, topics and memory items into a MemoryGraph."""

from typing import List, Optional, Dict, Any
from T_mem.utils.logger import get_logger
from T_mem.types import Scene
from T_mem.structure import (
    MemoryGraph,
    MemoryItemNode,
    SceneNode,
    TopicNode,
)
from T_mem.extractors.memory_item_extractor import MemoryItemExtractResult
from T_mem.extractors.topic_extractor import TopicExtractResult

logger = get_logger(__name__)


class MemoryGraphExtractor:
    """Aggregates Scenes, Topics and MemoryItems into a MemoryGraph container."""

    def __init__(self):
        pass

    def build_memory_graph(
        self,
        scenes: List[Scene],
        item_results: List[MemoryItemExtractResult],
        topic_extract_result: Optional[TopicExtractResult],
    ) -> MemoryGraph:
        logger.info("[MemoryGraphExtractor] Building two-layer graph")

        memory_graph = MemoryGraph()

        for scene in scenes:
            scene_node = SceneNode.from_scene(scene=scene, item_ids=[])
            memory_graph.scenes[scene.scene_id] = scene_node
        logger.info(f"  Scene layer: Added {len(scenes)} Scene nodes")

        topics = topic_extract_result.topics if topic_extract_result else []
        for topic in topics:
            topic_node = TopicNode.from_topic(topic=topic)
            memory_graph.topics[topic.topic_id] = topic_node
        logger.info(f"  Topic layer: Added {len(topics)} topic nodes")

        all_items = []
        for result in item_results:
            all_items.extend(result.items)

        for item in all_items:
            item_node = MemoryItemNode.from_item(item=item)
            memory_graph.items[item.item_id] = item_node

            for scene_id in item.scene_ids:
                scene_node = memory_graph.scenes.get(scene_id)
                if scene_node is not None and item.item_id not in scene_node.item_ids:
                    scene_node.item_ids.append(item.item_id)

        logger.info(f"  Item layer: Added {len(all_items)} item nodes")
        logger.info(f"[MemoryGraphExtractor] Graph built - {memory_graph.get_stats()}")

        return memory_graph

    def get_memory_graph_summary(self, memory_graph: MemoryGraph) -> Dict[str, Any]:
        stats = memory_graph.get_stats()
        scene_fanout = [len(sc.item_ids) for sc in memory_graph.scenes.values()]
        topic_fanout = [len(t.scene_ids) for t in memory_graph.topics.values()]
        return {
            "stats": stats,
            "avg_items_per_scene": (sum(scene_fanout) / len(scene_fanout)) if scene_fanout else 0.0,
            "avg_scenes_per_topic": (sum(topic_fanout) / len(topic_fanout)) if topic_fanout else 0.0,
        }

    def print_memory_graph_structure(self, memory_graph: MemoryGraph):
        logger.info("=" * 80)
        logger.info("Graph Structure Summary")
        logger.info("=" * 80)

        summary = self.get_memory_graph_summary(memory_graph)

        logger.info("Node statistics:")
        logger.info(f"  Items:    {summary['stats']['items']}")
        logger.info(f"  Scenes:   {summary['stats']['scenes']}")
        logger.info(f"  Topics:   {summary['stats']['topics']}")

        logger.info("Fan-out:")
        logger.info(f"  avg items/scene:    {summary['avg_items_per_scene']:.2f}")
        logger.info(f"  avg scenes/topic:   {summary['avg_scenes_per_topic']:.2f}")

        logger.info("=" * 80)
