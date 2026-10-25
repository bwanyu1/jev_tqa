import json
import random

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from djev import schema
from djev.model import DiffusionLM
from djev.sampler import SamplerConfig
from djev.sdrl import DecisionTask, SudokuTask, rollouts, score
from djev.sft import collate, diffusion_loss, encode_example
from djev.sudoku import grid, messages, parse

SOL = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
PUZ = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")


def test_sudoku_reward_and_parse():
    task = SudokuTask([])
    row = {"messages": messages(PUZ) + [{"role": "assistant", "content": grid(SOL)}]}
    assert task.reward(row, grid(SOL)) == pytest.approx(2.0)
    wrong = SOL[:2] + ("1" if SOL[2] != "1" else "2") + SOL[3:]  # cell 2 is blank in PUZ
    assert 0.9 < task.reward(row, grid(wrong)) < 1.0
    assert parse(grid(SOL)) == SOL


def test_decision_reward():
    task = DecisionTask([])
    row = {"state": "s", "questions": [{"type": "noul", "id": "a", "question": "?"}, {"type": "choice", "id": "b", "question": "?", "options": ["x", "y"]}],
           "answers": {"a": "true", "b": "2"}}
    assert task.reward(row, '{"a": true, "b": 2}') == pytest.approx(1.1)
    assert task.reward(row, '{"a": false, "b": 2') == pytest.approx(0.5)


def test_one_sdrl_iteration_runs(tok):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=2048)
    m = DiffusionLM(Qwen3ForCausalLM._from_config(cfg, attn_implementation="sdpa"))
    row = {"messages": messages(PUZ) + [{"role": "assistant", "content": grid(SOL)}]}
    task = SudokuTask([row])
    sampler = SamplerConfig(canvas_length=32, max_canvases=1, max_steps=3)
    stops = schema.stop_ids(tok)
    kept, mean_r, mean_steps = rollouts(m, tok, task, [row], 2, sampler, stops)
    assert 1 <= mean_steps <= 3 and mean_r >= 0
    # force one kept example to exercise the update path
    e = encode_example(tok, task.target(row, grid(SOL)), 256, 1, 256)
    loss, _ = diffusion_loss(m, collate([e]), 256, 1, 256, encoder_loss=False, t_low=0.5)
    loss.backward()
    assert torch.isfinite(loss)
    s = score(m, tok, task, [row], sampler, stops)
    assert set(s) == {"reward", "steps_per_row", "tpf"}
