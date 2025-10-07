import re
import pandas as pd
from tools import calc_expression, safe_eval_expr
from model_wrapper import SmallLLM
from utils import simple_sentence_split

class AgenticPipeline:
    def __init__(self, train_path=None, use_llm=True, llm_model_name=None):
        self.use_llm = use_llm
        self.llm = SmallLLM(llm_model_name) if (use_llm and llm_model_name) else None
        self.memory = {}
        if train_path:
            self._load_train(train_path)

    def _load_train(self, path):
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                q = str(row.get("problem_statement", "")).lower()
                ans = str(row.get("answer_option_" + str(row.get("correct option", "")), ""))
                self.memory[q] = ans
        except Exception as e:
            print("⚠️ Training data not loaded:", e)

    def retrieve_similar(self, question):
        question = question.lower()
        for q in self.memory.keys():
            if q.split("?")[0][:20] in question or question.split("?")[0][:20] in q:
                return self.memory[q]
        return None

    def decompose(self, question: str):
        sentences = simple_sentence_split(question)
        subtasks = []
        math_pattern = re.compile(r'[\d\.\(\)\+\-\*\/\^=]|sum|product|calculate|solve|what is')
        for s in sentences:
            s_strip = s.strip()
            if not s_strip:
                continue
            if math_pattern.search(s_strip.lower()):
                subtasks.append({'type': 'calc', 'text': s_strip})
            else:
                subtasks.append({'type': 'reason', 'text': s_strip})
        if not subtasks:
            subtasks.append({'type': 'reason', 'text': question})
        return subtasks

    def choose_tool(self, subtask):
        if subtask['type'] == 'calc':
            return 'calculator'
        if subtask['type'] == 'reason':
            return 'llm' if (self.use_llm and self.llm) else 'python_eval'
        return 'python_eval'

    def execute_subtask(self, subtask):
        tool = self.choose_tool(subtask)
        text = subtask['text']
        if tool == 'calculator':
            expr = re.sub(r'(?i)(calculate|compute|what is|find|=)', '', text)
            try:
                value = calc_expression(expr)
                return {'tool': 'calculator', 'ok': True, 'result': str(value)}
            except Exception:
                try:
                    value = safe_eval_expr(expr)
                    return {'tool': 'safe_eval', 'ok': True, 'result': str(value)}
                except Exception as e:
                    return {'tool': tool, 'ok': False, 'error': str(e)}
        elif tool == 'python_eval':
            try:
                v = safe_eval_expr(text)
                return {'tool': 'safe_eval', 'ok': True, 'result': str(v)}
            except Exception as e:
                return {'tool': 'safe_eval', 'ok': False, 'error': str(e)}
        elif tool == 'llm':
            prompt = f"Solve or explain logically: {text}\\nGive reasoning and numeric result if any."
            resp = self.llm.generate(prompt)
            return {'tool': 'llm', 'ok': True, 'result': resp}
        return {'tool': 'none', 'ok': False, 'error': 'no tool'}

    def run_mcq(self, question: str, options: list):
        known_answer = self.retrieve_similar(question)
        if known_answer:
            try:
                idx = options.index(known_answer) + 1
                return {"solution": f"Found similar in training: {known_answer}", "correct_option": idx}
            except:
                pass

        subtasks = self.decompose(question)
        reasoning = []
        for st in subtasks:
            res = self.execute_subtask(st)
            if res.get('ok'):
                reasoning.append(res['result'])

        scores = []
        for i, opt in enumerate(options, start=1):
            score = sum(str(r).lower() in opt.lower() for r in reasoning)
            nums_q = re.findall(r'[-+]?\d*\.?\d+', " ".join(reasoning))
            nums_o = re.findall(r'[-+]?\d*\.?\d+', opt)
            score += len(set(nums_q) & set(nums_o))
            scores.append((i, score))

        best = max(scores, key=lambda x: x[1])[0] if scores else 1
        reason_text = " | ".join(reasoning) if reasoning else "No reasoning"
        return {"solution": reason_text, "correct_option": best}
