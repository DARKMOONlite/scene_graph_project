"""
Functions for computing spatial/topological predicates between two binary masks
produced by SAM2 (shape: H x W, dtype bool or uint8).

Each function takes two np.ndarray masks (A, B) and returns a bool.

Predicates
----------
Topological  : disjoint, touching, inside
Positional   : left_of, right_of, above, below
Proximal     : near
Size         : smaller
"""

import numpy as np
from scipy.ndimage import binary_dilation
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class BasicPredicate():
    mask1:np.ndarray
    mask2:np.ndarray
    disjoint:bool = True
    touching:bool = False
    inside:bool = False
    _b_inside_a:bool = False
    left_of:bool = False
    right_of:bool = False
    above:bool = False
    below:bool = False
    near:bool = False
    # smaller:bool = False
    def __init__(self, mask1: np.ndarray, mask2: np.ndarray):
        self.mask1 = mask1
        self.mask2 = mask2

        # Compute centroids once — reused for near + all positional predicates
        cx1, cy1 = get_centroid(mask1)
        cx2, cy2 = get_centroid(mask2)

        h, w = mask1.shape[-2], mask1.shape[-1]
        threshold = 0.10 * np.sqrt(h ** 2 + w ** 2)
        self.near = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) < threshold
        if not self.near:
            return

        # Positional: derived directly from centroids, no extra function calls
        self.left_of = cx1 < cx2
        self.right_of = not self.left_of
        self.above = cy1 < cy2
        self.below = not self.above

        # Compute pixel overlap once — shared by disjoint, touching, inside
        a = _as_bool(mask1).squeeze()
        b = _as_bool(mask2).squeeze()
        overlapping = bool(np.any(a & b))

        if overlapping:
            # Overlapping: can't be disjoint or touching; check containment
            self.disjoint = False
            self.touching = False
            self.inside = bool(np.all(b[a])) if a.any() else False
            self._b_inside_a = bool(np.all(a[b])) if b.any() else False
        else:
            # Not overlapping: one dilation to determine touching vs disjoint
            adj = bool(np.any(_dilate(a, 3) & b))
            self.touching = adj
            self.disjoint = not adj
            self.inside = False
            self._b_inside_a = False

        # self.smaller = smaller(mask1, mask2)

    def reverse(self) -> "BasicPredicate":
        return BasicPredicate(self.mask2,self.mask1)

    def primary_predicates(self) -> list[str]:
        """Return the most salient combination of predicates.

        - If inside: return ["inside"] alone (containment implies position).
        - Otherwise: combine one topological/proximal predicate with one
          positional predicate, e.g. ["touching", "left_of"] or ["near", "above"].
        - Falls back to ["disjoint"] if nothing else applies.
        """
        if self.inside:
            return ["inside"]

        result = []

        # Topological / proximal (pick most specific)
        if self.touching:
            result.append("touching")
        elif self.near:
            result.append("near")
        else:
            result.append("disjoint")

        # Positional (pick the true one; above/below take priority over left/right)
        if self.above:
            result.append("above")
        elif self.below:
            result.append("below")
        elif self.left_of:
            result.append("left_of")
        elif self.right_of:
            result.append("right_of")

        return result

def _as_bool(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool)


def get_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Return (x, y) centroid of a boolean mask (image coordinates)."""
    coords = np.argwhere(_as_bool(mask).squeeze())   # shape (N, 2): (row, col) = (y, x)
    if len(coords) == 0:
        return (0.0, 0.0)
    # print(coords.mean(axis=0))
    y, x = coords.mean(axis=0)
    return (float(x), float(y))


def _area(mask: np.ndarray) -> int:
    return int(_as_bool(mask).sum())


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Morphologically dilate a boolean mask by `radius` pixels."""
    m = _as_bool(mask).squeeze()  # handles (1, H, W) SAM2 masks -> (H, W)
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return binary_dilation(m, structure=struct)


# # ---------------------------------------------------------------------------
# # Topological
# # ---------------------------------------------------------------------------

# def disjoint(A: np.ndarray, B: np.ndarray) -> bool:
#     """True if A and B share no pixels AND do not touch at their boundaries."""
#     a, b = _as_bool(A), _as_bool(B)
#     if np.any(a & b):          # overlapping — not disjoint
#         return False
#     # check adjacency: dilate A by 1 pixel; if it reaches B they are touching
#     return not np.any(_dilate(a) & b)


# def touching(A: np.ndarray, B: np.ndarray) -> bool:
#     """True if A and B do not overlap but share a boundary (are adjacent)."""
#     a, b = _as_bool(A), _as_bool(B)
#     if np.any(a & b):          # overlapping — not merely touching
#         return False
#     return np.any(_dilate(a,3) & b)


# def inside(A: np.ndarray, B: np.ndarray) -> bool:
#     """True if every pixel of A is also in B (A is fully contained by B)."""
#     a, b = _as_bool(A), _as_bool(B)
#     if _area(a) == 0:
#         return False
#     return bool(np.all(b[a]))


# # ---------------------------------------------------------------------------
# # Positional  (centroid-based; image origin is top-left)
# # ---------------------------------------------------------------------------

# def left_of(A: np.ndarray, B: np.ndarray) -> bool:
#     """True if the centroid of A is to the left of the centroid of B."""
#     return near(A,B) and get_centroid(A)[0] < get_centroid(B)[0]


# def right_of(A: np.ndarray, B: np.ndarray) -> bool:
#     """True if the centroid of A is to the right of the centroid of B."""
#     return near(A,B) and get_centroid(A)[0] > get_centroid(B)[0]


# def above(A: np.ndarray, B: np.ndarray) -> bool:
#     """True if the centroid of A is above the centroid of B.
#     In image coordinates y increases downward, so above means smaller y."""
#     return near(A,B) and get_centroid(A)[1] < get_centroid(B)[1]


# def below(A: np.ndarray, B: np.ndarray) -> bool:
#     """True if the centroid of A is below the centroid of B."""
#     return near(A,B) and get_centroid(A)[1] > get_centroid(B)[1]


# # ---------------------------------------------------------------------------
# # Proximal
# # ---------------------------------------------------------------------------

# def near(A: np.ndarray, B: np.ndarray, threshold: float | None = None) -> bool:
#     """True if the centroids of A and B are within `threshold` pixels.

#     If `threshold` is None it defaults to 10 % of the image diagonal,
#     which gives a scale-invariant measure regardless of image resolution.
#     """
#     if threshold is None:
#         h, w = A.shape[:2]
#         threshold = 0.10 * np.sqrt(h ** 2 + w ** 2)
#     ax, ay = get_centroid(A)
#     bx, by = get_centroid(B)
#     dist = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
#     return dist < threshold


# # ---------------------------------------------------------------------------
# # Size
# # ---------------------------------------------------------------------------

# def smaller(A: np.ndarray, B: np.ndarray) -> bool:
#     """True if A covers fewer pixels than B."""
#     return _area(A) < _area(B)
