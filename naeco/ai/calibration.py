"""Display-only probability calibration (stdlib; no engine or network imports).

The same increasing Platt map can be applied to actor values and completed Q.
Only actor-value calibration is validated by the Phase 9 held-out data.
"""
import json
import math
from dataclasses import dataclass
from typing import Optional

_EPS = 1e-6


def _probability(p):
    if isinstance(p, bool) or not isinstance(p, (int, float)) or \
            not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    return float(p)


def _logit(p):
    p = max(_EPS, min(1.0 - _EPS, _probability(p)))
    return math.log(p) - math.log1p(-p)


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _validated_pairs(pairs):
    result = []
    for p, y in pairs:
        p = _probability(p)
        if isinstance(y, bool) or not isinstance(y, (int, float)) or \
                y not in (0.0, 0.5, 1.0):
            raise ValueError("target must be 0, 0.5 or 1")
        result.append((p, float(y)))
    return result


@dataclass(frozen=True)
class Identity:
    def apply(self, p):
        return _probability(p)

    def describe(self):
        return {"kind": "identity", "a": 1.0, "b": 0.0, "source": None}


@dataclass(frozen=True)
class Platt:
    a: float
    b: float
    source: Optional[dict] = None

    def __post_init__(self):
        for value in (self.a, self.b):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                    not math.isfinite(value):
                raise ValueError("Platt coefficients must be finite numbers")
        if self.a <= 0:
            raise ValueError("Platt slope must be positive (increasing map)")

    def apply(self, p):
        return _sigmoid(self.a * _logit(p) + self.b)

    def describe(self):
        return {"kind": "platt", "a": self.a, "b": self.b,
                "source": self.source}


def fit_platt(pairs, iters=50):
    """Deterministic Newton/IRLS fit of Bernoulli cross-entropy.

    A backtracking line search keeps the map increasing and decreases loss.
    The tiny Hessian ridge only stabilizes the solve; it is not a loss penalty.
    """
    if type(iters) is not int or iters <= 0:
        raise ValueError("iters must be a positive integer")
    data = [(_logit(p), y) for p, y in _validated_pairs(pairs)]
    if not data or min(x for x, _ in data) == max(x for x, _ in data):
        raise ValueError("fit needs nonempty data with distinct probabilities")
    n = len(data)

    def loss(a, b):
        total = 0.0
        for x, y in data:
            z = a * x + b
            total += max(z, 0.0) - y * z + math.log1p(math.exp(-abs(z)))
        return total / n

    a, b = 1.0, 0.0
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in data:
            q = _sigmoid(a * x + b)
            error = q - y
            weight = q * (1.0 - q)
            ga += error * x
            gb += error
            haa += weight * x * x
            hab += weight * x
            hbb += weight
        ga, gb = ga / n, gb / n
        haa, hab, hbb = haa / n + 1e-9, hab / n, hbb / n + 1e-9
        determinant = haa * hbb - hab * hab
        da = (hbb * ga - hab * gb) / determinant
        db = (haa * gb - hab * ga) / determinant
        old_loss = loss(a, b)
        step = 1.0
        for _ in range(30):
            next_a, next_b = a - step * da, b - step * db
            if next_a > 0 and loss(next_a, next_b) <= old_loss:
                break
            step *= 0.5
        else:
            break
        a, b = next_a, next_b
        if max(abs(step * da), abs(step * db)) < 1e-9:
            break
    return Platt(a, b)


def ece_with_bins(pairs, bins=10):
    """Return equal-width ECE and the nonempty diagnostic bucket table.

    p=1 belongs to the last bucket. Empty data returns (0.0, []).
    """
    if type(bins) is not int or bins <= 0:
        raise ValueError("bins must be a positive integer")
    data = _validated_pairs(pairs)
    buckets = [[] for _ in range(bins)]
    for p, y in data:
        buckets[min(bins - 1, int(p * bins))].append((p, y))
    result = 0.0
    table = []
    for b, items in enumerate(buckets):
        if items:
            mp = sum(p for p, _ in items) / len(items)
            my = sum(y for _, y in items) / len(items)
            result += abs(mp - my) * len(items) / len(data)
            table.append({"bin": b, "n": len(items), "mean_p": mp, "mean_y": my})
    return result, table


def ece(pairs, bins=10):
    """Equal-width ECE; use ece_with_bins for the diagnostic bucket table."""
    return ece_with_bins(pairs, bins)[0]


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and \
        all(c in "0123456789abcdef" for c in value)


def load_calibrator(path, expected_weights_sha256):
    """Load a schema-1 Platt file, or an exact identity with a short reason.

    The input digest is provenance; loading never reads the diagnostic dataset.
    The champion digest binds the map to the model used to fit it.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except OSError:
        return Identity(), "calibration file missing or unreadable"
    except (ValueError, UnicodeError):
        return Identity(), "calibration file corrupt: invalid JSON"
    try:
        if not isinstance(raw, dict) or type(raw.get("schema")) is not int or \
                raw["schema"] != 1 or raw.get("kind") != "platt":
            raise ValueError("unsupported schema or kind")
        source = raw["source"]
        if not isinstance(source, dict) or not _is_sha256(source.get("file_sha256")):
            raise ValueError("invalid source digest")
        if not _is_sha256(raw.get("champion_weights_sha256")):
            raise ValueError("invalid champion digest")
        calibrator = Platt(raw["a"], raw["b"], source=dict(source))
    except (KeyError, TypeError, ValueError) as exc:
        return Identity(), "calibration file corrupt: %s" % exc
    if raw["champion_weights_sha256"] != expected_weights_sha256:
        return Identity(), "calibration champion weights SHA-256 mismatch"
    return calibrator, None
