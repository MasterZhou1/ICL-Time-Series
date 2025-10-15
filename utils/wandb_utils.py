"""
Lightweight Weights & Biases integration with .env support.

Usage:
    from utils.wandb_utils import WandbRun
    with WandbRun(project="icl-time-series", config=my_config_dict) as wb:
        wb.log({"train/loss": 0.123})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os


def _maybe_load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        # dotenv is optional; ignore if unavailable
        pass


@dataclass
class WandbRun:
    project: str
    config: Optional[Dict[str, Any]] = None
    group: Optional[str] = None
    name: Optional[str] = None
    mode: Optional[str] = None  # "online" | "offline" | "disabled"

    def __post_init__(self) -> None:
        _maybe_load_env()
        self._wb = None
        # Default to disabled unless explicitly enabled with valid API key
        env_mode = os.getenv("WANDB_MODE")
        api_key = os.getenv("WANDB_API_KEY")
        if self.mode is None:
            self.mode = "online" if (env_mode == "online" and api_key) else "disabled"
        # Silence any prompts/output
        os.environ.setdefault("WANDB_SILENT", "true")
        os.environ.setdefault("WANDB_CONSOLE", "off")
        os.environ.setdefault("WANDB_START_METHOD", "thread")
        if self.mode == "disabled":
            os.environ["WANDB_DISABLED"] = "true"

    def __enter__(self):
        try:
            import wandb
        except Exception:
            return self  # wandb optional; no-op

        try:
            self._wb = wandb.init(
                project=self.project,
                config=self.config,
                group=self.group,
                name=self.name,
                mode=self.mode,
                settings=wandb.Settings(start_method="thread")
            )
        except Exception:
            # Fall back to disabled mode if auth/network fails
            try:
                self._wb = wandb.init(project=self.project, config=self.config, mode="disabled")
            except Exception:
                self._wb = None
        return self

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._wb is None:
            return
        try:
            import wandb
            wandb.log(data, step=step)
        except Exception:
            pass

    def __exit__(self, exc_type, exc, tb):
        if self._wb is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        self._wb = None


