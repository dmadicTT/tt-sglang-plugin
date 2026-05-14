"""Helpers for the bundled DeepSeek R1 Tenstorrent mock model."""

import os
from importlib.resources import files

DEEPSEEK_R1_0528_MODEL_ID = "deepseek-ai/DeepSeek-R1-0528"
DEEPSEEK_R1_0528_MOCK_MODEL_DIR = "mock_model"


def deepseek_r1_0528_mock_model_path() -> str:
    return str(files("sglang_tenstorrent").joinpath(DEEPSEEK_R1_0528_MOCK_MODEL_DIR))


def is_deepseek_r1_0528_mock_model_path(path: str) -> bool:
    return os.path.abspath(path) == os.path.abspath(deepseek_r1_0528_mock_model_path())


mock_model_path = deepseek_r1_0528_mock_model_path
is_mock_model_path = is_deepseek_r1_0528_mock_model_path
