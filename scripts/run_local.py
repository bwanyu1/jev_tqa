"""Use the bundled final Jev model or the general diffusion model offline."""
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['jev', 'diffusion_only'], required=True)
    parser.add_argument('--request', default='examples/request.json')
    parser.add_argument('--prompt', default='Explain why the sky is blue in two sentences.')
    parser.add_argument('--device', default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    import torch
    from djev import schema
    from djev.infill import denoise_infill, readout
    from djev.sampler import SamplerConfig, generate
    from djev.sft import parse_questions
    from models.final_diffusion.load_models import load_model

    row = questions = None
    if args.model == 'jev':
        row = json.loads(Path(args.request).read_text())
        questions = parse_questions(row['questions'])
        schema.validate(questions)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    model, tokenizer, processor = load_model(args.model, device=args.device)
    def sync():
        if model.device.type == 'cuda':
            torch.cuda.synchronize()
        elif model.device.type == 'mps':
            torch.mps.synchronize()
    sync()
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with torch.inference_mode():
        if args.model == 'jev':
            template, slots, candidates = schema.infill_template(tokenizer, row['state'], questions)
            if len(template) > 128 or max(slots) >= 128:
                raise ValueError('Answer template exceeds 128 tokens; shorten ids or split questions.')
            if any(m['type'] == 'video' for m in row.get('media', [])):
                from scripts.try_video_practical import video_prompt_inputs
                processor.video_processor.fps = 2
                processor.video_processor.max_frames = 32
                ids, mm = video_prompt_inputs(tokenizer, processor, row['state'], questions, row['media'])
            else:
                ids, mm = schema.prompt_inputs(tokenizer, processor, row['state'], questions, row.get('media'))
            mm = {k: v.to(model.device) for k, v in mm.items()} if mm else None
            context = model.encode(torch.tensor([ids], device=model.device), mm=mm)
            _, first, _, steps = denoise_infill(model, context, template, slots,
                SamplerConfig(canvas_length=128, max_steps=args.max_steps or 48))
            decisions = {}
            for q, cand, logits in zip(questions, candidates, first):
                probs = readout(logits, cand)
                decisions[q.id] = {'answer': max(probs, key=probs.get), 'probs': probs}
            result = {'decisions': decisions, 'steps': steps, 'readout': 'first'}
        else:
            ids = tokenizer.apply_chat_template([{'role':'user', 'content':args.prompt}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors='pt', return_dict=False)
            if isinstance(ids, dict):
                ids = ids['input_ids']
            stop_ids = {tokenizer.eos_token_id}
            im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
            if im_end is not None and im_end != tokenizer.unk_token_id:
                stop_ids.add(im_end)
            result_gen = generate(model, ids, {x for x in stop_ids if x is not None},
                SamplerConfig(canvas_length=256, max_steps=args.max_steps or 192, max_canvases=2))
            result = {'text':tokenizer.decode(result_gen.tokens, skip_special_tokens=True)}
    sync()
    result.update(model=args.model, load_seconds=load_seconds,
                  processing_seconds=time.perf_counter()-started)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
