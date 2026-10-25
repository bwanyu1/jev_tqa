"""Run user-authored decision requests through the final infill checkpoint.

This records actual outputs and latency; it does not assign benchmark scores.
Run from the project root: python -m scripts.try_practical --requests ... --out ...
"""
import argparse
import json
import time
from pathlib import Path

import torch

from djev.infill import diffusion_decide
from djev.io import load
from djev.sampler import SamplerConfig
from djev.sft import parse_questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--checkpoint', default='runs/jev_infill')
    args = parser.parse_args()
    torch.manual_seed(20261025)
    started = time.perf_counter()
    model, tok, processor = load('Qwen/Qwen3-VL-4B-Instruct', args.checkpoint)
    model.eval()
    cfg = SamplerConfig(canvas_length=128, max_steps=48)
    print(json.dumps({'load_seconds': time.perf_counter() - started}), flush=True)
    rows = [json.loads(line) for line in Path(args.requests).read_text().splitlines() if line.strip()]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as sink, torch.inference_mode():
        for i, row in enumerate(rows):
            questions = parse_questions(row['questions'])
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                decisions, steps = diffusion_decide(model, tok, row['state'], questions, cfg,
                                                    row.get('media'), processor)
            except (ValueError, IndexError, RuntimeError) as exc:
                result = {'id': row['id'], 'input': row, 'checkpoint': args.checkpoint,
                          'seconds': time.perf_counter() - start,
                          'error': f'{type(exc).__name__}: {exc}'}
                sink.write(json.dumps(result, ensure_ascii=False) + '\n')
                sink.flush()
                print(json.dumps({'id': row['id'], 'error': result['error']}, ensure_ascii=False), flush=True)
                continue
            torch.cuda.synchronize()
            result = {'id': row['id'], 'input': row, 'checkpoint': args.checkpoint,
                      'seconds': time.perf_counter() - start, 'steps': steps,
                      'cold_first_request': i == 0, 'decisions': decisions}
            sink.write(json.dumps(result, ensure_ascii=False) + '\n')
            sink.flush()
            print(json.dumps({'id': row['id'], 'seconds': result['seconds'], 'steps': steps,
                              'answers': {k: {'answer': v['answer'], 'probs': v['probs']}
                                          for k, v in decisions.items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
