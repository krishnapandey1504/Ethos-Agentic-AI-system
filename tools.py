import ast
import operator as op
from sympy import sympify, N

ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: lambda x: -x,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}

def _eval_ast(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        return ALLOWED_OPERATORS[type(node.op)](operand)
    raise ValueError("Unsupported expression")

def safe_eval_expr(expr: str):
    expr = expr.strip()
    tree = ast.parse(expr, mode='eval')
    return _eval_ast(tree.body)

def calc_expression(expr: str):
    e = sympify(expr, evaluate=True)
    try:
        return float(N(e))
    except Exception:
        return str(e)
