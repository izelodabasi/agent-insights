from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Kind(StrEnum):
    PROMPT = "prompt"
    ASSISTANT = "assistant"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    REJECTION = "rejection"
    INTERRUPT = "interrupt"
    COMMAND = "command"
    NOTIFICATION = "notification"


class Event(BaseModel):
    kind: Kind
    timestamp: datetime
    text: str = ""
    tool: str = ""
    tool_id: str = ""
    tool_input: dict = Field(default_factory=dict)
    is_error: bool = False
    sidechain: bool = False
    tags: set[str] = Field(default_factory=set)


class Usage(BaseModel):
    model: str
    timestamp: datetime | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = Field(0, description="subset of cache_write_tokens on the 1h TTL")
    cache_read_tokens: int = 0
    cache_write_requires_explicit_price: bool = False


class Session(BaseModel):
    agent: str
    session_id: str
    project: str
    cwd: str
    title: str = ""
    ai_title: str = ""
    branches: set[str] = Field(default_factory=set)
    reasoning_efforts: set[str] = Field(default_factory=set)
    events: list[Event] = Field(default_factory=list)
    usage: dict[str, Usage] = Field(default_factory=dict)

    @property
    def start(self) -> datetime:
        return self.events[0].timestamp

    @property
    def end(self) -> datetime:
        return self.events[-1].timestamp

    @property
    def tags(self) -> set[str]:
        return {t for e in self.events for t in e.tags}

    def of(self, *kinds: Kind) -> list[Event]:
        return [e for e in self.events if e.kind in kinds]

    def user_messages(self) -> list[Event]:
        """What the user typed: prompts, and the reasons given when rejecting a tool call."""
        return [e for e in self.of(Kind.PROMPT, Kind.REJECTION) if e.text.strip()]
