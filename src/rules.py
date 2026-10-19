"""rule-lab.html 의 규칙 엔진(RuleCore)을 Python 으로 옮긴 것.

조합 문자열 예:
    "Q:first > I:add,c=3 > B:left,k=8 > R:add"

`규칙코드:모드,파라미터=값` 을 `>` 로 잇는다. 모드·파라미터를 생략하면 rule-lab 기본값을 쓴다.
AO 의 주기 상수는 `period=2/5/3` 처럼 `/` 로 구분한다.

값은 1..n 의 양의 정수이고, 모든 규칙 뒤에 ((x - 1) mod p) + 1 을 적용한다 (기본 p = n).
범위 안의 값(1..p)은 그대로 두고, 넘은 값만 1..p 로 감는다.
규칙 M 과 조건 변환의 `mod` 는 rule-lab 과 같이 항상 n 을 쓴다.
"""

import hashlib
import math
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

MAX_DIGITS = 10_000
_DIGIT_LIMIT = 10**MAX_DIGITS
_POW_BIT_LIMIT = 33_220
_FACTOR_TICKS = 300_000

# 자릿수 규칙(P)은 최대 10,000자리 정수를 문자열로 바꾼다.
if hasattr(sys, "get_int_max_str_digits"):
    if 0 < sys.get_int_max_str_digits() < MAX_DIGITS:
        sys.set_int_max_str_digits(MAX_DIGITS)

# code -> (모드 목록, 기본 모드). 순서와 기본값은 rule-lab 의 RULES 와 같다.
RULE_MODES: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "A": (("keep", "asc", "desc", "high", "low"), "high"),
    "B": (("keep", "left", "right"), "left"),
    "C": (("keep", "reverse"), "reverse"),
    "D": (("keep", "odd", "even"), "odd"),
    "E": (("keep", "shuffle", "inverse"), "shuffle"),
    "I": (("keep", "add"), "add"),
    "J": (("keep", "mul"), "mul"),
    "K": (("keep", "square", "cube", "pow"), "square"),
    "M": (("keep", "mod"), "mod"),
    "P": (("keep", "reverse", "sum", "product"), "reverse"),
    "Q": (("first", "last", "min", "max", "kth"), "first"),
    "R": (("keep", "add", "mul"), "add"),
    "S": (("keep", "sum", "min", "max"), "sum"),
    "AF": (("keep", "sum", "product", "diff"), "diff"),
    "AG": (("keep", "swap"), "swap"),
    "X": (("keep", "add", "mul", "pow"), "add"),
    "Y": (("keep", "add", "mul"), "add"),
    "Z": (("keep", "even", "odd"), "even"),
    "AB": (("keep", "above", "below", "previous"), "above"),
    "AJ": (
        ("all", "oddIndex", "evenIndex", "oddValue", "evenValue", "threshold"),
        "oddIndex",
    ),
    "AO": (("keep", "add", "mul"), "add"),
    "BA": (("keep", "largest", "smallest", "omega", "distinct"), "largest"),
    "BC": (("keep", "addG", "mulG", "addL", "mulL"), "addG"),
}

# Z / AB / AJ 가 선택한 원소에 적용하는 변환
FUNCTIONS = ("keep", "add", "mul", "pow", "mod", "reverse", "sum", "product")

_DEFAULT_PARAMS = {
    "c": "2",
    "k": "2",
    "a": "2",
    "b": "3",
    "t": "10",
    "period": "2/5/3",
    "fn": "add",
    "fc": "2",
}


@dataclass
class Step:
    code: str
    mode: str
    params: Dict[str, object] = field(default_factory=dict)


# ------------------------------------------------------------------ parsing

def _param_names(code: str, mode: str, fn: str) -> List[str]:
    """이 규칙·모드가 실제로 쓰는 파라미터 (rule-lab 화면에 보이는 입력칸과 같다)."""
    if mode == "keep":
        return []
    names: List[str] = []
    if code in ("I", "J"):
        names.append("c")
    if code == "B" or (code == "K" and mode == "pow") or (code == "Q" and mode == "kth"):
        names.append("k")
    if code == "Y":
        names += ["a", "b"]
    if code == "AO":
        names.append("period")
    if code == "AJ" and mode == "threshold":
        names.append("t")
    if code in ("Z", "AB", "AJ"):
        names.append("fn")
        if fn in ("add", "mul", "pow"):
            names.append("fc")
    return names


def _parse_int(raw: object, label: str, minimum: int) -> int:
    text = str(raw).strip()
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"{label} must be an integer >= {minimum}, got '{raw}'")
    value = int(text)
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {value}")
    return value


def _parse_param(code: str, name: str, raw: object) -> object:
    label = f"{code}.{name}"
    if name == "fn":
        if raw not in FUNCTIONS:
            raise ValueError(f"{label} must be one of {list(FUNCTIONS)}, got '{raw}'")
        return raw
    if name == "period":
        parts = raw if isinstance(raw, (list, tuple)) else re.split(r"[/\s,]+", str(raw).strip())
        values = tuple(_parse_int(v, label, 1) for v in parts if str(v) != "")
        if not values:
            raise ValueError(f"{label} needs at least one value")
        return values
    # B 의 이동량만 0 을 허용한다.
    return _parse_int(raw, label, 0 if (code, name) == ("B", "k") else 1)


def make_step(code: str, mode: Optional[str] = None, fields: Optional[dict] = None) -> Step:
    """규칙 하나를 만든다. fields 에서 이 규칙이 쓰지 않는 값은 무시한다."""
    if code not in RULE_MODES:
        raise ValueError(f"unknown rule '{code}', expected one of {list(RULE_MODES)}")
    modes, default_mode = RULE_MODES[code]
    mode = default_mode if mode is None else mode
    if mode not in modes:
        raise ValueError(f"rule {code}: unknown mode '{mode}', expected one of {list(modes)}")
    fields = fields or {}
    fn = fields.get("fn", _DEFAULT_PARAMS["fn"])
    params: Dict[str, object] = {}
    for name in _param_names(code, mode, fn):
        default = "1" if (code, name) == ("B", "k") else _DEFAULT_PARAMS[name]
        params[name] = _parse_param(code, name, fields.get(name, default))
    return Step(code, mode, params)


def parse_rules(spec: str) -> List[Step]:
    """'Q:first > I:add,c=3 > R:add' -> [Step, ...]"""
    pieces = [p.strip() for p in str(spec).split(">")]
    if not pieces or any(not p for p in pieces):
        raise ValueError(f"empty rule in '{spec}'")
    steps: List[Step] = []
    for piece in pieces:
        code, _, rest = piece.partition(":")
        code = code.strip().upper()
        items = [x.strip() for x in rest.split(",")] if rest.strip() else []
        mode = None
        if items and "=" not in items[0]:
            mode = items.pop(0)
        fields: Dict[str, str] = {}
        for item in items:
            name, eq, value = item.partition("=")
            name = name.strip()
            if not eq or not name:
                raise ValueError(f"rule {code}: expected name=value, got '{item}'")
            if name in fields:
                raise ValueError(f"rule {code}: duplicate parameter '{name}'")
            fields[name] = value.strip()
        step = make_step(code, mode, fields)
        unused = sorted(set(fields) - set(step.params))
        if unused:
            raise ValueError(
                f"rule {step.code}:{step.mode} does not use parameter(s) {unused}"
            )
        steps.append(step)
    return steps


def format_step(step: Step) -> str:
    out = f"{step.code}:{step.mode}"
    for name, value in step.params.items():
        if name == "period":
            value = "/".join(str(v) for v in value)
        out += f",{name}={value}"
    return out


def format_rules(steps: Sequence[Step]) -> str:
    """정규화된 조합 문자열. 같은 조합은 항상 같은 문자열이 된다."""
    return " > ".join(format_step(s) for s in steps)


def rules_hash(steps: Sequence[Step]) -> str:
    return hashlib.sha1(format_rules(steps).encode("utf-8")).hexdigest()[:8]


# --------------------------------------------------------------- arithmetic

def _checked(x: int) -> int:
    if x >= _DIGIT_LIMIT:
        raise ValueError(f"intermediate value exceeds {MAX_DIGITS} digits")
    return x


def _pow(x: int, e: int) -> int:
    if x == 1 or e == 1:
        return x
    if (x.bit_length() - 1) * e > _POW_BIT_LIMIT:
        raise ValueError("power result exceeds the computation limit")
    return _checked(x**e)


def _digits(x: int, mode: str) -> int:
    chars = str(x)
    if mode == "reverse":
        return int(chars[::-1])
    if mode == "sum":
        return sum(int(d) for d in chars)
    out = 1
    for d in chars:
        if d != "0":
            out = _checked(out * int(d))
    return out


def _factors(x: int) -> Dict[str, int]:
    if x == 1:
        return {"smallest": 1, "largest": 1, "omega": 1, "distinct": 1}
    rest, d = x, 2
    first, last, count, distinct, ticks = 0, 1, 0, 0, 0
    while d * d <= rest:
        ticks += 1
        if ticks > _FACTOR_TICKS:
            raise ValueError("prime factorization is too large")
        if rest % d == 0:
            distinct += 1
            if not first:
                first = d
            last = d
            while rest % d == 0:
                rest //= d
                count += 1
        d = 3 if d == 2 else d + 2
    if rest > 1:
        if not first:
            first = rest
        last = rest
        count += 1
        distinct += 1
    return {"smallest": first, "largest": last, "omega": count + 1, "distinct": distinct + 1}


def _function(step: Step, n: int):
    fn = step.params["fn"]
    c = step.params.get("fc", 1)

    def apply(x: int) -> int:
        if fn == "keep":
            return x
        if fn == "add":
            return _checked(x + c)
        if fn == "mul":
            return _checked(x * c)
        if fn == "pow":
            return _pow(x, c)
        if fn == "mod":
            return (x - 1) % n + 1
        return _digits(x, fn)

    return apply


def _alternate_ends(sorted_xs: List[int], take_high: bool) -> List[int]:
    out: List[int] = []
    lo, hi = 0, len(sorted_xs) - 1
    while lo <= hi:
        if take_high:
            out.append(sorted_xs[hi])
            hi -= 1
        else:
            out.append(sorted_xs[lo])
            lo += 1
        take_high = not take_high
    return out


# ---------------------------------------------------------------- execution

def apply_step(
    xs: List[int], step: Step, n: int, r: Optional[int]
) -> Tuple[List[int], Optional[int]]:
    """규칙 하나를 적용한다. (새 배열, 저장값 r) 을 돌려준다."""
    code, mode, p = step.code, step.mode, step.params
    m = len(xs)
    if mode == "keep":
        return list(xs), r
    ys = list(xs)

    if code == "A":
        s = sorted(xs)
        if mode == "asc":
            ys = s
        elif mode == "desc":
            ys = s[::-1]
        else:
            ys = _alternate_ends(s, mode == "high")
    elif code == "B":
        k = p["k"] % m
        shift = k if mode == "left" else (m - k) % m
        ys = xs[shift:] + xs[:shift]
    elif code == "C":
        ys = xs[::-1]
    elif code == "D":
        idx = [i for i in range(m) if i % 2 == (0 if mode == "odd" else 1)]
        for i, v in zip(idx, [xs[i] for i in idx][::-1]):
            ys[i] = v
    elif code == "E":
        if mode == "inverse":
            ys = xs[0::2] + xs[1::2]
        else:
            half = (m + 1) // 2
            ys = []
            for i in range(half):
                ys.append(xs[i])
                if i + half < m:
                    ys.append(xs[i + half])
    elif code == "I":
        ys = [_checked(x + p["c"]) for x in xs]
    elif code == "J":
        ys = [_checked(x * p["c"]) for x in xs]
    elif code == "K":
        e = 2 if mode == "square" else 3 if mode == "cube" else p["k"]
        ys = [_pow(x, e) for x in xs]
    elif code == "M":
        ys = [(x - 1) % n + 1 for x in xs]
    elif code == "P":
        ys = [_digits(x, mode) for x in xs]
    elif code == "Q":
        if mode == "first":
            r = xs[0]
        elif mode == "last":
            r = xs[-1]
        elif mode == "min":
            r = min(xs)
        elif mode == "max":
            r = max(xs)
        else:
            if p["k"] > m:
                raise ValueError(f"Q.k must be <= array length {m}, got {p['k']}")
            r = xs[p["k"] - 1]
    elif code == "R":
        if r is None:
            raise ValueError("R needs a preceding Q")
        ys = [_checked(x + r if mode == "add" else x * r) for x in xs]
    elif code == "S":
        v = _checked(sum(xs)) if mode == "sum" else min(xs) if mode == "min" else max(xs)
        ys = [_checked(x + v) for x in xs]
    elif code == "AF":
        ys = []
        for i, x in enumerate(xs):
            y = xs[(i + 1) % m]
            if mode == "sum":
                ys.append(_checked(x + y))
            elif mode == "product":
                ys.append(_checked(x * y))
            else:
                ys.append(_checked((y - x) ** 2 + 1))
    elif code == "AG":
        for i in range(0, m - 1, 2):
            ys[i], ys[i + 1] = ys[i + 1], ys[i]
    elif code == "X":
        if mode == "pow":
            ys = [_pow(x, i + 1) for i, x in enumerate(xs)]
        elif mode == "add":
            ys = [_checked(x + i + 1) for i, x in enumerate(xs)]
        else:
            ys = [_checked(x * (i + 1)) for i, x in enumerate(xs)]
    elif code == "Y":
        a, b = p["a"], p["b"]
        ys = [
            _checked(x + (a if i % 2 == 0 else b) if mode == "add" else x * (a if i % 2 == 0 else b))
            for i, x in enumerate(xs)
        ]
    elif code in ("Z", "AB", "AJ"):
        transform = _function(step, n)
        if code == "Z":
            mask = [x % 2 == (0 if mode == "even" else 1) for x in xs]
        elif code == "AB":
            total = _checked(sum(xs))
            if mode == "above":
                mask = [x * m > total for x in xs]
            elif mode == "below":
                mask = [x * m < total for x in xs]
            else:
                mask = [x > xs[(i - 1) % m] for i, x in enumerate(xs)]
        else:
            select = {
                "all": lambda i, x: True,
                "oddIndex": lambda i, x: i % 2 == 0,
                "evenIndex": lambda i, x: i % 2 == 1,
                "oddValue": lambda i, x: x % 2 == 1,
                "evenValue": lambda i, x: x % 2 == 0,
                "threshold": lambda i, x: x >= p["t"],
            }[mode]
            mask = [select(i, x) for i, x in enumerate(xs)]
        ys = [transform(x) if keep else x for x, keep in zip(xs, mask)]
    elif code == "AO":
        values = p["period"]
        ys = [
            _checked(x + values[i % len(values)] if mode == "add" else x * values[i % len(values)])
            for i, x in enumerate(xs)
        ]
    elif code == "BA":
        memo: Dict[int, Dict[str, int]] = {}
        ys = []
        for x in xs:
            if x not in memo:
                memo[x] = _factors(x)
            ys.append(memo[x][mode])
    elif code == "BC":
        if mode.endswith("G"):
            v = math.gcd(*xs)
        else:
            v = xs[0]
            for x in xs[1:]:
                v = _checked(v // math.gcd(v, x) * x)
        ys = [_checked(x + v if mode.startswith("add") else x * v) for x in xs]

    if any(y < 1 for y in ys):
        raise ValueError("result left the positive integers")
    return ys, r


def run_rules(
    values: Sequence[int], steps: Sequence[Step], n: int, modulus: Optional[int] = None
) -> List[int]:
    """values (1..n) 에 규칙을 차례로 적용하고 ((x - 1) mod p) + 1 을 돌려준다 (값 1..p)."""
    p = n if modulus is None else modulus
    if p <= 0:
        raise ValueError(f"modulus must be positive, got {p}")
    if not values:
        raise ValueError("rules require at least one input value")
    xs = [int(v) for v in values]
    r: Optional[int] = None
    for i, step in enumerate(steps):
        try:
            xs, r = apply_step(xs, step, n, r)
        except ValueError as e:
            raise ValueError(f"step {i + 1} ({format_step(step)}): {e}") from e
    return [(x - 1) % p + 1 for x in xs]
