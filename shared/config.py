"""Shared configuration for V4Pro Engineering Runtime."""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class RuntimeConfig:
    """Global runtime configuration."""

    # Project paths
    project_path: str = field(default_factory=lambda: os.getcwd())
    state_dir: str = ""

    # Model backend
    model_backend: str = "deepseek-v4-pro"
    context_limit: int = 1_000_000  # V4 Pro 1M context

    # V3: Memory Cascade
    cascade_l0_max_turns: int = 50     # L0 → L1 after 50 turns
    cascade_l1_max_turns: int = 200    # L1 → L2 after 200 turns
    cascade_l2_max_turns: int = 500    # L2 → L3 after 500 turns
    error_loop_threshold: int = 3      # 连续同样错误触发 reflect

    # V3: Reflection
    reflect_cooldown_turns: int = 10   # 反思最小间隔
    reflect_on_error_loop: bool = True
    reflect_on_phase_change: bool = True

    # H1: Divergence
    walk_default_steps: int = 4
    walk_max_steps: int = 10
    injection_threshold: float = 0.35
    injection_cooldown_turns: int = 5

    # Search
    search_provider: str = "duckduckgo"  # default, free, no API key
    search_max_results: int = 5

    def __post_init__(self):
        if not self.state_dir:
            self.state_dir = os.path.join(self.project_path, ".v4pro-runtime")
        Path(self.state_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        return cls(
            project_path=os.environ.get("PROJECT_PATH", os.getcwd()),
            model_backend=os.environ.get("MODEL_BACKEND", "deepseek-v4-pro"),
            search_provider=os.environ.get("SEARCH_PROVIDER", "duckduckgo"),
        )

    def save(self):
        path = os.path.join(self.state_dir, "config.json")
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    @classmethod
    def load(cls, state_dir: str) -> "RuntimeConfig":
        path = os.path.join(state_dir, "config.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        return cls(state_dir=state_dir)
