"""
Xenonite pattern generator — correct approach.

The xenonite pattern is a SURFACE LATTICE: take the input mesh's surface,
scatter Voronoi seed points across it, then keep only the faces that lie
near a Voronoi cell boundary (= the "wires"). Faces inside Voronoi cells
are removed, leaving holes. The result looks like the original shape built
from an interconnected lattice of lines with circular/hexagonal voids.

Algorithm:
  1. Load mesh, extract face centroids.
  2. Scatter N seed points randomly across the surface (proportional to area).
  3. For each face centroid, find its 2 nearest seeds. The boundary between
     those two seeds is the Voronoi edge passing through that face.
  4. Compute the face's distance to that boundary line. Faces within
     wire_width of a boundary are KEPT; faces further away are REMOVED.
  5. Optionally also keep faces near a 3rd seed (triple-point nodes = circles).
  6. Export the masked mesh as a binary STL.
"""

import io
import numpy as np
import trimesh
from scipy.spatial import cKDTree


# ── tuneable defaults ─────────────────────────────────────────────────────────
SEED_DENSITY    = 0.0012   # seeds per unit² of surface area
MIN_SEEDS       = 80
MAX_SEEDS       = 800
WIRE_WIDTH      = 0.22     # fraction of avg cell radius kept as wire
NODE_FACTOR     = 1.6      # triple-points (circle nodes) are wider
# ─────────────────────────────────────────────────────────────────────────────


def _dist_to_midplane(points: np.ndarray,
                      a: np.ndarray,
                      b: np.ndarray) -> np.ndarray:
    """
    Voronoi boundary between seed arrays a[i] and b[i] is the perpendicular
    bisector plane for each pair. Return each point's signed distance to its
    pair's plane. a, b, points all have shape (N, 3).
    """
    mid    = (a + b) / 2.0          # (N, 3)
    ab     = b - a                   # (N, 3)
    ab_len = np.linalg.norm(ab, axis=1, keepdims=True)  # (N, 1)
    ab_len = np.where(ab_len < 1e-9, 1.0, ab_len)
    normal = ab / ab_len             # (N, 3)
    diff   = points - mid            # (N, 3)
    return np.abs((diff * normal).sum(axis=1))  # (N,)


def generate_xenonite(stl_bytes: bytes,
                      seed_density: float = SEED_DENSITY,
                      wire_width: float = WIRE_WIDTH) -> bytes:
    """
    Accept raw STL bytes, return xenonite-patterned STL bytes.
    """
    mesh = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Could not parse mesh from uploaded file.")

    # ── 1. Sample seeds weighted by face area ─────────────────────────────────
    face_centers = mesh.triangles_center          # (F, 3)
    face_areas   = mesh.area_faces                # (F,)
    total_area   = float(face_areas.sum())

    n_seeds = int(np.clip(
        total_area * seed_density,
        MIN_SEEDS, MAX_SEEDS
    ))

    rng   = np.random.default_rng(42)
    probs = face_areas / total_area
    idx   = rng.choice(len(face_centers), size=n_seeds, replace=False, p=probs)
    seeds = face_centers[idx]                     # (S, 3)

    # ── 2. For every face find its 3 nearest seeds ────────────────────────────
    tree      = cKDTree(seeds)
    dists, nn = tree.query(face_centers, k=min(3, n_seeds))
    # nn shape: (F, k)  — indices into seeds

    # ── 3. Compute distance of each face centroid to Voronoi edge (1st & 2nd) ─
    s1 = seeds[nn[:, 0]]        # nearest seed
    s2 = seeds[nn[:, 1]]        # 2nd nearest

    edge_dist = _dist_to_midplane(face_centers, s1, s2)

    # ── 4. Adaptive wire width ────────────────────────────────────────────────
    # Use the distance between a face's 1st and 2nd seed as cell-local scale
    cell_scale = dists[:, 1]          # distance to 2nd seed ≈ half cell diameter
    threshold  = wire_width * cell_scale

    on_wire = edge_dist < threshold

    # ── 5. Also keep triple-point nodes (where 3 cells almost meet) ──────────
    if n_seeds >= 3:
        s3        = seeds[nn[:, 2]]
        edge_dist2 = _dist_to_midplane(face_centers, s1, s3)
        edge_dist3 = _dist_to_midplane(face_centers, s2, s3)
        # A triple point is where all three edges are close
        near_triple = (
            (edge_dist  < NODE_FACTOR * threshold) &
            (edge_dist2 < NODE_FACTOR * threshold) &
            (edge_dist3 < NODE_FACTOR * threshold)
        )
        on_wire |= near_triple

    # ── 6. Build masked mesh ──────────────────────────────────────────────────
    kept_faces = mesh.faces[on_wire]
    if len(kept_faces) == 0:
        raise ValueError("No faces passed the wire filter — try increasing wire_width.")

    result = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=kept_faces,
        process=False,
    )
    result.remove_unreferenced_vertices()

    out = io.BytesIO()
    result.export(out, file_type="stl")
    return out.getvalue()
