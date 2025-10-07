import argparse
import pandas as pd
from tqdm import tqdm
from pipeline import AgenticPipeline
import json

def main(args):
    df = pd.read_csv(args.input)
    pipeline = AgenticPipeline(train_path=args.train, use_llm=not args.no_llm, llm_model_name=args.llm_model)
    results = []
    traces = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        topic = str(row.get("topic", ""))
        question = str(row.get("problem_statement", ""))
        options = [str(row.get(f"answer_option_{i}", "")) for i in range(1, 6)]
        # new: run_mcq_with_trace should return trace
        res = pipeline.run_mcq(question, options)  # if you updated pipeline.run_mcq to include trace
        results.append({
            "topic": topic,
            "problem_statement": question,
            "solution": res["solution"],
            "correct option": res["correct_option"]
        })
        traces.append({
            "topic": topic,
            "problem_statement": question,
            "trace": res.get("trace", {})
        })
    pd.DataFrame(results).to_csv(args.output, index=False)
    with open(args.traces, 'w', encoding='utf-8') as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\\n")
    print(f"✅ Results saved to {args.output} and traces to {args.traces}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--traces", default="traces.jsonl")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-model", default="sshleifer/tiny-gpt2")
    args = parser.parse_args()
    main(args)