"""Manual video annotation trial with sampling audit, stage timing and an AR control."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import DynamicCache
from transformers.video_utils import load_video

from djev import schema
from djev.infill import ar_decide, denoise_infill, readout
from djev.io import load
from djev.sampler import SamplerConfig
from djev.sft import parse_questions


def video_prompt_inputs(tokenizer, processor, state, questions, media):
    if not media or not any(m['type'] == 'video' for m in media):
        return schema.prompt_inputs(tokenizer, processor, state, questions, media)
    if any(m['type'] != 'video' for m in media):
        raise ValueError('This trial expects video-only media')
    decoded = [load_video(m['path'], backend='pyav') for m in media]
    text = processor.apply_chat_template(schema.messages(state, questions, media),
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    out = processor(text=[text], videos=[v for v, metadata in decoded],
                    video_metadata=[metadata for v, metadata in decoded],
                    do_sample_frames=True, return_tensors='pt')
    return out['input_ids'][0].tolist(), {k: out[k] for k in schema.MM_KEYS if k in out}


def sync():
    torch.cuda.synchronize()


def ar_cached(model, ids, mm, template, slots, candidates, questions):
    """Same fixed-template AR readout, with a single multimodal prefill."""
    cache = DynamicCache()
    seq = ids + template[:slots[0]]
    extra = dict(mm or {})
    if 'mm_token_type_ids' in extra:
        types = extra['mm_token_type_ids']
        extra['mm_token_type_ids'] = torch.cat(
            [types, types.new_zeros(1, len(seq) - types.shape[1])], 1)
    x = torch.tensor([seq], device=model.device)
    pos = model.rope_positions(x, torch.ones_like(x, dtype=torch.bool), extra)
    logits = model.backbone(input_ids=x, position_ids=pos, past_key_values=cache,
                            use_cache=True, **extra).logits[0, -1]
    next_pos = int(pos.max()) + 1
    decisions = {}
    for i, (q, cand) in enumerate(zip(questions, candidates)):
        probs = readout(logits, cand)
        answer = max(probs, key=probs.get)
        decisions[q.id] = {'answer': answer, 'probs': probs}
        if i + 1 < len(slots):
            chunk = [cand[answer]] + template[slots[i] + 1:slots[i + 1]]
            x = torch.tensor([chunk], device=model.device)
            pos = torch.arange(next_pos, next_pos + len(chunk), device=model.device)[None]
            logits = model.backbone(input_ids=x, position_ids=pos, past_key_values=cache,
                                    use_cache=True).logits[0, -1]
            next_pos += len(chunk)
    return decisions


def verify_text_ar(model, tokenizer):
    questions = [schema.Noul('a', 'Does the sentence mention a cat?'),
                 schema.Noul('b', 'Does the sentence mention a dog?')]
    state = 'A cat is sleeping.'
    reference, _ = ar_decide(model.backbone, tokenizer, state, questions)
    template, slots, candidates = schema.infill_template(tokenizer, state, questions)
    got = ar_cached(model, schema.prompt_ids(tokenizer, state, questions), None,
                    template, slots, candidates, questions)
    error = max(abs(got[q.id]['probs'][v] - reference[q.id]['probs'][v])
                for q in questions for v in q.values())
    if error > 1e-3:
        raise RuntimeError(f'Cached AR text-path mismatch: {error}')
    print(json.dumps({'ar_text_path_check_max_abs_error': error}), flush=True)


@torch.inference_mode()
def run_case(model, tokenizer, processor, row, mode, first_call):
    qs = parse_questions(row['questions'])
    template, slots, candidates = schema.infill_template(tokenizer, row['state'], qs)
    if len(template) > 128 or max(slots) >= 128:
        raise ValueError(f'Answer template exceeds canvas: {len(template)} tokens')
    sample_audit = []
    vp = processor.video_processor
    config = row['video_config']
    vp.fps, vp.max_frames = config['fps'], config['max_frames']
    vp.max_video_tokens = config['max_video_tokens']
    original_sample = vp.sample_frames

    def sample(metadata, **kwargs):
        indices = original_sample(metadata, **kwargs)
        sample_audit.append({'source_fps': metadata.fps,
                             'total_source_frames': metadata.total_num_frames,
                             'sampled_indices': indices.tolist(),
                             'sampled_seconds': (indices / metadata.fps).tolist()})
        return indices

    vp.sample_frames = sample
    vision_times, vision_started = [], []
    visual = model.backbone.model.visual

    def before(*unused):
        sync()
        vision_started.append(time.perf_counter())

    def after(*unused):
        sync()
        vision_times.append(time.perf_counter() - vision_started.pop())

    hooks = [visual.register_forward_pre_hook(before), visual.register_forward_hook(after)]
    try:
        sync()
        started = time.perf_counter()
        ids, mm = video_prompt_inputs(tokenizer, processor, row['state'], qs, row.get('media'))
        sync()
        prepared = time.perf_counter()
        grids = {k: mm[k].tolist() for k in ['image_grid_thw', 'video_grid_thw'] if mm and k in mm}
        mm = {k: v.to(model.device) for k, v in mm.items()} if mm else None
        sync()
        if mode == 'diffusion':
            ctx = model.encode(torch.tensor([ids], device=model.device), mm=mm)
            sync()
            encoded = time.perf_counter()
            _, first, final, steps = denoise_infill(
                model, ctx, template, slots, SamplerConfig(canvas_length=128, max_steps=48))
            sync()
            finished = time.perf_counter()
            decisions = {}
            for q, cand, initial, last in zip(qs, candidates, first, final):
                p = readout(initial, cand)
                pf = readout(last, cand)
                decisions[q.id] = {'answer': max(p, key=p.get), 'probs': p,
                                   'final_answer': max(pf, key=pf.get), 'final_probs': pf}
            timing = {'prepare_s': prepared-started, 'encode_s': encoded-prepared,
                      'denoise_s': finished-encoded}
        else:
            decisions = ar_cached(model, ids, mm, template, slots, candidates, qs)
            sync()
            finished = time.perf_counter()
            steps = len(qs)
            timing = {'prepare_s': prepared-started, 'prefill_and_labels_s': finished-prepared}
        timing.update(total_s=finished-started, vision_forward_s=sum(vision_times))
        return {'id': row['id'], 'mode': mode, 'input': row,
                'cold_first_request': first_call, 'prompt_tokens': len(ids),
                'answer_template_tokens': len(template), 'grids': grids,
                'sample_audit': sample_audit, 'steps_or_ar_forwards': steps,
                'timing': timing, 'decisions': decisions}
    finally:
        vp.sample_frames = original_sample
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mode', choices=['diffusion', 'ar'], default='diffusion')
    args = parser.parse_args()
    torch.manual_seed(20261025)
    ckpt = 'runs/jev_infill' if args.mode == 'diffusion' else 'runs/ar_sft_v5'
    started = time.perf_counter()
    model, tokenizer, processor = load('Qwen/Qwen3-VL-4B-Instruct', ckpt)
    print(json.dumps({'mode': args.mode, 'load_seconds': time.perf_counter()-started}), flush=True)
    if args.mode == 'ar':
        with torch.inference_mode():
            verify_text_ar(model, tokenizer)
    rows = [json.loads(line) for line in Path(args.requests).read_text().splitlines() if line.strip()]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as sink:
        for i, row in enumerate(rows):
            try:
                result = run_case(model, tokenizer, processor, row, args.mode, i == 0)
                result['checkpoint'] = ckpt
            except Exception as exc:
                result = {'id': row['id'], 'mode': args.mode, 'checkpoint': ckpt,
                          'input': row, 'error': f'{type(exc).__name__}: {exc}'}
            sink.write(json.dumps(result, ensure_ascii=False)+'\n')
            sink.flush()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if 'device-side assert' in result.get('error', ''):
                raise RuntimeError('CUDA context failed; stop this worker')
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
