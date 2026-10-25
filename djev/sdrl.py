"""Simplified SD·RL (DiffusionGemma §5): online sampler distillation + reward maximisation.

The paper describes the stage but not its loss. This follows its description with the methods it
cites, in the simplest form that fits one 16 GB GPU:

  1. online teacher  : the current model samples G completions per prompt with a generous budget
                       (N = --teacher-steps) and the Algorithm 1 sampler (stochastic, so G differ)
  2. reward          : a verifiable reward per completion (Sudoku grid / decision answers); the best of G
                       is kept (reward-ranked finetuning, RAFT), ties going to the completion that
                       converged in FEWER denoising steps, which pushes towards short trajectories
  3. distillation    : the kept completions are trained with the diffusion loss, the noise level drawn
                       from U[--t-low, 1] so the model learns to finish from heavily-noised canvases,
                       i.e. in few steps (self-distillation through time)
  4. repeat          : as the model sharpens, adaptive stopping shortens the teacher's own trajectories
                       (the paper's implicit curriculum)

Every --eval-every iterations the student is scored with a small budget (N = --student-steps).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time

import torch

from . import schema
from .io import load, save
from .sampler import SamplerConfig, generate
from .sft import collate, diffusion_loss, encode_example, parse_questions


# ----------------------------------------------------------------------------- tasks
class SudokuTask:
    """Rows: chat rows from djev.sudoku build (the assistant turn holds the solution, used only as reward)."""

    def __init__(self, rows):
        self.rows = rows

    def prompt(self, tok, row):
        *ctx, _ = row["messages"]
        return tok.apply_chat_template(ctx, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)

    def reward(self, row, text):
        *ctx, gold = row["messages"]
        sol = "".join(re.findall(r"[1-9]", gold["content"]))
        puzzle = ctx[-1]["content"].replace(" ", "")
        pred = "".join(re.findall(r"[1-9]", text))[:81]
        blanks = [i for i, c in enumerate(puzzle) if c == "0"]
        cells = sum(1 for i in blanks if i < len(pred) and pred[i] == sol[i]) / max(len(blanks), 1)
        return cells + (1.0 if pred == sol else 0.0)  # partial credit per cell, bonus for a solved grid

    def target(self, row, text):
        *ctx, _ = row["messages"]
        return {"messages": ctx + [{"role": "assistant", "content": text.strip()}]}


class DecisionTask:
    """Rows: decision rows with gold answers (reward = share of questions answered correctly, 0 if no JSON)."""

    def __init__(self, rows):
        self.rows = rows

    def prompt(self, tok, row):
        return schema.prompt_ids(tok, row["state"], parse_questions(row["questions"]))

    def reward(self, row, text):
        got = 0
        for q in row["questions"]:
            m = re.search(r'"%s"\s*:\s*(true|false|\d)\b' % re.escape(q["id"]), text)
            got += bool(m) and m.group(1) == str(row["answers"][q["id"]]).lower()
        return got / len(row["questions"]) + (0.1 if text.strip().endswith("}") else 0.0)

    def target(self, row, text):
        return {"state": row["state"], "questions": row["questions"], "answers": row["answers"], "_text": text}


# ----------------------------------------------------------------------------- loop
@torch.no_grad()
def rollouts(model, tok, task, rows, G, cfg, stops):
    model.eval()
    kept, rewards, steps = [], [], []
    for row in rows:
        ids = torch.tensor([task.prompt(tok, row)])
        cands = []
        for _ in range(G):
            g = generate(model, ids, stops, cfg)
            text = tok.decode(g.tokens)
            cands.append((task.reward(row, text), -g.total_steps, text))
            rewards.append(cands[-1][0])
            steps.append(g.total_steps)
        best = max(cands)
        if best[0] > 0:
            kept.append(task.target(row, best[2]))
    model.train()
    return kept, sum(rewards) / len(rewards), sum(steps) / len(steps)


@torch.no_grad()
def score(model, tok, task, rows, cfg, stops):
    model.eval()
    r, s, toks = 0.0, 0, 0
    for row in rows:
        g = generate(model, torch.tensor([task.prompt(tok, row)]), stops, cfg)
        r += task.reward(row, tok.decode(g.tokens))
        s += g.total_steps + len(g.canvases) - 1
        toks += len(g.tokens)
    model.train()
    return {"reward": round(r / len(rows), 4), "steps_per_row": round(s / len(rows), 2), "tpf": round(toks / max(s, 1), 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--init", required=True, help="diffusion checkpoint to improve (e.g. runs/sudoku_diffusion)")
    ap.add_argument("--task", choices=["sudoku", "decisions"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--prompts-per-iter", type=int, default=16)
    ap.add_argument("--group", type=int, default=4, help="G completions per prompt")
    ap.add_argument("--updates-per-iter", type=int, default=16)
    ap.add_argument("--teacher-steps", type=int, default=192)
    ap.add_argument("--student-steps", type=int, default=48)
    ap.add_argument("--t-low", type=float, default=0.5)
    ap.add_argument("--canvas", type=int, default=256)
    ap.add_argument("--canvases", type=int, default=1)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    model, tok, _ = load(a.backbone, a.init)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    from peft import LoraConfig, get_peft_model

    model.backbone = get_peft_model(model.backbone, LoraConfig(r=a.lora_rank, lora_alpha=2 * a.lora_rank, target_modules="all-linear",
                                                               exclude_modules=r".*(vision|audio)\w*\..*"))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    ev = [json.loads(l) for l in open(a.eval_data, encoding="utf-8")][: a.eval_n]
    Task = SudokuTask if a.task == "sudoku" else DecisionTask
    task, ev_task = Task(rows), Task(ev)
    if a.task == "sudoku" and "messages" not in ev[0]:  # djev.sudoku eval rows -> chat rows
        from .sudoku import grid, messages

        ev_task = Task([{"messages": messages(r["puzzle"]) + [{"role": "assistant", "content": grid(r["solution"])}]} for r in ev])
    stops = schema.stop_ids(tok)
    teacher = SamplerConfig(canvas_length=a.canvas, max_canvases=a.canvases, max_steps=a.teacher_steps)
    student = SamplerConfig(canvas_length=a.canvas, max_canvases=a.canvases, max_steps=a.student_steps)
    log = open(f"{a.out}.log.jsonl", "a")

    def record(d):
        d["time"] = time.strftime("%F %T")
        print(json.dumps(d), flush=True)
        log.write(json.dumps(d) + "\n"); log.flush()

    record({"iter": 0, "eval_student": score(model, tok, ev_task, ev_task.rows, student, stops),
            "eval_teacher": score(model, tok, ev_task, ev_task.rows[:10], teacher, stops)})
    for it in range(1, a.iterations + 1):
        batch_rows = random.sample(task.rows, a.prompts_per_iter)
        kept, mean_r, mean_steps = rollouts(model, tok, task, batch_rows, a.group, teacher, stops)
        enc = []
        for k in kept:
            if a.task == "decisions":  # train on the teacher's own JSON text
                text = k.pop("_text")
                k = {"messages": schema.messages(k["state"], parse_questions(k["questions"])) + [{"role": "assistant", "content": text.strip()}]}
            e = encode_example(tok, k, a.prompt_len, a.canvases, a.canvas)
            if e:
                enc.append(e)
        losses = []
        for u in range(a.updates_per_iter if enc else 0):
            loss, _ = diffusion_loss(model, collate(random.sample(enc, min(a.batch, len(enc)))), a.prompt_len, a.canvases,
                                     a.canvas, encoder_loss=False, t_low=a.t_low)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        rec = {"iter": it, "teacher_reward": round(mean_r, 4), "teacher_steps": round(mean_steps, 1), "kept": len(enc),
               "loss": round(sum(losses) / max(len(losses), 1), 4)}
        if it % a.eval_every == 0 or it == a.iterations:
            rec["eval_student"] = score(model, tok, ev_task, ev_task.rows, student, stops)
            save(model, tok, f"{a.out}/iter_{it}", {"backbone": a.backbone, "canvas_length": a.canvas, "args": {**vars(a)}, "iter": it})
        record(rec)
    save(model, tok, a.out, {"backbone": a.backbone, "canvas_length": a.canvas, "args": vars(a)})


if __name__ == "__main__":
    main()
