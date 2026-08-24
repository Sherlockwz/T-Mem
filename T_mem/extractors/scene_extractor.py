"""Base classes for scene boundary detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from T_mem.llm.llm_provider import LLMProvider
from T_mem.types import RawDataType, Scene


@dataclass
class RawData:
    content: dict[str, Any]
    data_id: str
    data_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SceneExtractRequest:
    history_raw_data_list: List[RawData]
    new_raw_data_list: List[RawData]
    user_id_list: List[str]
    smart_mask_flag: Optional[bool] = False


@dataclass
class StatusResult:
    should_wait: bool


class SceneExtractor(ABC):
    def __init__(self, raw_data_type: RawDataType, llm_provider=LLMProvider, **llm_kwargs):
        self.raw_data_type = raw_data_type
        self.llm_kwargs = llm_kwargs
        self._llm_provider = llm_provider

    @abstractmethod
    async def extract_scene(self, request: SceneExtractRequest) -> tuple[Optional[Scene], Optional[StatusResult]]:
        pass
