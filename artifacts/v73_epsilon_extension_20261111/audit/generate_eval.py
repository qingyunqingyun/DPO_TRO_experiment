#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--pairs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--seed', type=int, default=2101)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--max-model-len', type=int, default=4096)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    a = p.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    rows = [json.loads(x) for x in Path(a.pairs).open() if x.strip()]
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True, trust_remote_code=True)
    prompts = [tok.apply_chat_template(x['prompt'], add_generation_prompt=True, tokenize=False) for x in rows]
    llm = LLM(model=a.model, seed=a.seed, trust_remote_code=True, task='generate',
              max_num_seqs=256, enable_prefix_caching=True,
              gpu_memory_utilization=a.gpu_memory_utilization,
              max_model_len=a.max_model_len)
    params = SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p,
                            max_tokens=a.max_new_tokens, min_tokens=1,
                            skip_special_tokens=True)
    outputs = llm.generate(prompts, params)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        for row, result in zip(rows, outputs):
            for j, completion in enumerate(result.outputs):
                f.write(json.dumps({'pair_id': row.get('pair_id'), 'prompt_id': row.get('prompt_id'),
                                    'prompt': row['prompt'], 'sample_id': j,
                                    'response': [{'role':'assistant','content': completion.text}],
                                    'seed': a.seed}, ensure_ascii=False) + '\n')
    print(json.dumps({'prompts': len(rows), 'samples': len(rows)*a.k, 'output': str(out)}))
if __name__ == '__main__':
    main()
