"""
compute.py -- deterministic formula evaluation (the math "plug in values" feature).

The model NEVER does the arithmetic (it hallucinates numbers). Instead it (or the
user) supplies a FORMULA and the variable VALUES; we parse the formula with sympy
and evaluate it exactly, returning an auditable breakdown (parsed form, the
substitution, the numeric result). Pure-Python + sympy: local, air-gap-safe,
reproducible -- the same defensibility stance as our citations.

Input may be a plain expression ("pi/2 * f * Q * W") or light LaTeX
("\\frac{\\pi}{2} f Q W"); we normalise the common LaTeX the VLM emits into a
sympy-parseable string. Parsing is SANDBOXED: unknown names become free Symbols
(variables), and only a fixed whitelist of math functions / constants is exposed --
there is no eval() and no path to an arbitrary callable.
"""

import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

# implicit_multiplication: "2x" -> 2*x ; convert_xor: "a^2" -> a**2 ;
# standard_transformations includes auto_symbol, so undefined names -> Symbol.
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

# The ONLY names the parser may resolve to something callable/constant. Everything
# else becomes a free Symbol, so a formula can never reach an arbitrary function.
_ALLOWED = {
    "pi": sp.pi, "E": sp.E, "I": sp.I,
    "sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "Abs": sp.Abs, "floor": sp.floor, "ceiling": sp.ceiling,
}

# The sandbox namespace handed to parse_expr's internal eval. It exposes the sympy
# CONSTRUCTORS the generated code needs (Symbol/Integer/Float/Rational) plus the
# whitelisted functions -- and "__builtins__": {} blocks all of Python's builtins,
# so even a crafted string cannot reach __import__, open, eval, etc.
_GLOBALS = {"__builtins__": {}, "Symbol": sp.Symbol, "Integer": sp.Integer,
            "Float": sp.Float, "Rational": sp.Rational}
_GLOBALS.update(_ALLOWED)

# Greek LaTeX names -> plain ASCII (so the catch-all "\command" stripper below does
# NOT delete them). "pi" is included here for the \pi -> pi rewrite even though it is
# also an _ALLOWED constant -- the two lists serve different layers (normalise vs parse).
_GREEK = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
          "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau",
          "upsilon", "phi", "chi", "psi", "omega")


def latex_to_expr(s):
    """Best-effort normalise common LaTeX into a sympy-parseable string. NOT a full
    LaTeX parser -- it handles the constructs the VLM actually emits ($$, \\frac,
    \\sqrt, \\cdot, ^{}, _{}, Greek names). A genuine parser is a later upgrade."""
    s = (s or "").strip()
    s = re.sub(r"\$\$?|\\\[|\\\]|\\\(|\\\)", " ", s)          # strip $, $$, \[ \] \( \)
    s = re.sub(r"\\[,;!:> ]", " ", s)                         # LaTeX spacing: \, \; \!
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    frac = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")  # \frac{a}{b} -> ((a)/(b))
    while frac.search(s):
        s = frac.sub(r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\^\s*\{([^{}]*)\}", r"**(\1)", s)            # ^{...} -> **(...)
    s = re.sub(r"_\s*\{([^{}]*)\}", r"_\1", s)                # f_{n} -> f_n (one symbol)
    for g in _GREEK:                                          # \alpha -> alpha, ...
        s = s.replace("\\" + g, g)
    s = s.replace("{", "(").replace("}", ")")                # leftover braces -> parens
    s = re.sub(r"\\[a-zA-Z]+", "", s)                         # drop any other \command
    return s.strip()


def parse_formula(text):
    """Parse a formula string (plain or LaTeX) into a sympy expression, sandboxed:
    unknown names -> free Symbols; only _ALLOWED is callable (global_dict={})."""
    raw = text or ""
    s = latex_to_expr(raw) if ("\\" in raw or "$" in raw) else raw.strip()
    if s.count("=") == 1:                # "G = pi/2 ..." -> evaluate the RHS only
        s = s.split("=", 1)[1]
    return parse_expr(s, local_dict={}, global_dict=_GLOBALS,
                      transformations=_TRANSFORMS, evaluate=True)


def evaluate(formula, values=None, digits=6):
    """Evaluate `formula` with `values` ({var_name: number}). Returns an auditable dict:
      ok          -- True iff parsed, every variable supplied, and a number resulted
      expr        -- parsed formula (str)
      symbols     -- variable names the formula needs
      missing     -- variables with no supplied value
      substituted -- the expression after substitution (str)
      result      -- float (or a still-symbolic str if vars are missing)
      error       -- parse/eval failure message, else None
    Never raises: failures come back as {'ok': False, 'error': ...}."""
    values = values or {}
    out = {"ok": False, "expr": None, "symbols": [], "missing": [],
           "substituted": None, "result": None, "error": None}
    try:
        expr = parse_formula(formula)
    except Exception as e:
        out["error"] = f"could not parse formula: {type(e).__name__}: {e}"
        return out

    out["expr"] = str(expr)
    syms = sorted(expr.free_symbols, key=lambda x: x.name)
    out["symbols"] = [s.name for s in syms]
    out["missing"] = [s.name for s in syms if s.name not in values]

    subs = {}
    for s in syms:
        if s.name in values:
            try:
                subs[s] = sp.Float(float(values[s.name]))
            except (TypeError, ValueError):
                out["error"] = f"value for '{s.name}' is not a number: {values[s.name]!r}"
                return out

    sub_expr = expr.subs(subs)
    out["substituted"] = str(sub_expr)
    try:
        val = sp.N(sub_expr, digits)
    except Exception as e:
        out["error"] = f"could not evaluate: {type(e).__name__}: {e}"
        return out

    if getattr(val, "free_symbols", set()):      # still has variables -> not a number
        out["result"] = str(val)
        out["ok"] = False                        # need the missing values first
    else:
        out["result"] = float(val)
        out["ok"] = True
    return out


def format_result(res):
    """One-line human summary of an evaluate() result (for the chat / CLI)."""
    if res["error"]:
        return f"⚠ {res['error']}"
    if res["missing"]:
        return (f"Need values for: {', '.join(res['missing'])}.  "
                f"Formula: {res['expr']}")
    return f"{res['expr']}  =  {res['result']:g}   (with {res['substituted']})"


if __name__ == "__main__":
    tests = [
        ("pi/2 * f * Q * W", {"f": 100, "Q": 10, "W": 0.04}),
        ("\\frac{\\pi}{2} f Q W", {"f": 100, "Q": 10, "W": 0.04}),
        ("a^2 + b^2", {"a": 3, "b": 4}),
        ("\\sqrt{a^2 + b^2}", {"a": 3, "b": 4}),
        ("G = sqrt(k*T/m)", {"k": 1.38e-23, "T": 300, "m": 4.65e-26}),
        ("f_n * 2", {"f_n": 10}),
        ("x + 1", {}),                 # missing var
        ("2 +", {}),                   # parse error
    ]
    for f, v in tests:
        print(f"{f!r}  {v}")
        print("   ", format_result(evaluate(f, v)))
