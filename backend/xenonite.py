"""
Xenonite pattern generator — v4: vertex displacement.

Instead of boolean subtraction (requires watertight input) or face removal
(creates non-manifold edges), this version displaces vertices along their
normals using a Voronoi ridge profile:

  - Vertices near a Voronoi cell boundary → pushed outward  (the raised wire)
  - Vertices far from any boundary (cell centre) → pushed inward (the recess)

This produces a clean embossed surface identical in look to the reference,
works on any input mesh, and always outputs a valid manifold mesh.

Algorithm:
  1. Scatter N Voronoi seed points on the surface (Poisson-disk style).
  2. For each vertex, find its 2 nearest seeds and compute distance to their
     bisector plane (= distance to nearest Voronoi edge).
  3. Apply a smooth ridge profile: Gaussian peak at the edge, valley at centre.
  4. Displace each vertex along its normal by the profile value.
  5. Also add spherical "node" bumps at triple-point junctions.
"""

import io
import numpy as np
import trimesh
from scipy.spatial import cKDTree


# ── tuneable defaults ─────────────────────────────────────────────────────────
SEED_DENSITY   = 0.008     # seeds per unit² of surface area
MIN_SEEDS      = 150
MAX_SEEDS      = 1200

# Ridge profile  (all in fraction of avg cell radius)
RIDGE_HEIGHT   = 1.8       # outward displacement at wire centre
VALLEY_DEPTH   = 0.6       # inward displacement at cell centre
RIDGE_SIGMA    = 0.18      # Gaussian width of the ridge (fraction of cell radius)
NODE_SIGMA     = 0.28      # width of the triple-point node bump
NODE_HEIGHT    = 2.2       # height of node bump (slightly taller than wire)
# ─────────────────────────────────────────────────────────────────────────────


def _dist_to_midplane_verts(verts: np.ndarray,
                             a: np.ndarray,
                             b: np.ndarray) -> np.ndarray:
    """Distance of each vertex to the bisector plane between its seed pair."""
    mid    = (a + b) / 2.0
    ab     = b - a
    ab_len = np.linalg.norm(ab, axis=1, keepdims=True)
    ab_len = np.where(ab_len < 1e-9, 1.0, ab_len)
    normal = ab / ab_len
    return np.abs(((verts - mid) * normal).sum(axis=1))


def _gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (x / sigma) ** 2)


def generate_xenonite(stl_bytes: bytes,
                      seed_density: float = SEED_DENSITY,
                      ridge_height: float = RIDGE_HEIGHT,
                      valley_depth: float = VALLEY_DEPTH) -> bytes:
    mesh = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Could not parse mesh from uploaded file.")

    # ── 1. Poisson-disk-like seed placement (rejection sampling) ─────────────
    face_areas   = mesh.area_faces
    total_area   = float(face_areas.sum())

    n_seeds = int(np.clip(total_area * seed_density, MIN_SEEDS, MAX_SEEDS))

    # Sample seed positions on the surface (uniform area weighting)
    seed_pts, _ = trimesh.sample.sample_surface(mesh, n_seeds)

    # Light Poisson-disk relaxation: remove seeds too close to each other
    expected_spacing = np.sqrt(total_area / n_seeds)
    tree_s = cKDTree(seed_pts)
    pairs  = tree_s.query_pairs(r=expected_spacing * 0.5)
    remove = set()
    for a, b in pairs:
        if a not in remove:
            remove.add(b)
    keep_mask = np.ones(n_seeds, dtype=bool)
    keep_mask[list(remove)] = False
    seeds = seed_pts[keep_mask]
    n_seeds = len(seeds)

    # ── 2. Per-vertex distance to nearest Voronoi boundaries ──────────────────
    k    = min(3, n_seeds)
    tree = cKDTree(seeds)
    verts = np.array(mesh.vertices, dtype=np.float64)

    dists, nn = tree.query(verts, k=k)
    # dists[:, 0] = dist to nearest seed  (≈ cell radius)
    # dists[:, 1] = dist to 2nd seed

    s1 = seeds[nn[:, 0]]
    s2 = seeds[nn[:, 1]]

    # Distance to Voronoi edge (bisector between 1st and 2nd seed)
    edge_dist  = _dist_to_midplane_verts(verts, s1, s2)
    cell_scale = dists[:, 1]   # ≈ half-cell diameter, used for normalisation

    norm_dist = edge_dist / np.maximum(cell_scale, 1e-6)   # 0=at wire, 1=cell centre

    # ── 3. Ridge profile ──────────────────────────────────────────────────────
    # At norm_dist≈0 (Voronoi edge): ridge_height outward
    # At norm_dist≈1 (cell centre): valley_depth inward
    ridge_val  = ridge_height * _gaussian(norm_dist, RIDGE_SIGMA)
    valley_val = -valley_depth * (1.0 - _gaussian(norm_dist, 0.5))
    displacement = ridge_val + valley_val

    # ── 4. Triple-point node bumps ────────────────────────────────────────────
    if k >= 3:
        s3   = seeds[nn[:, 2]]
        ed2  = _dist_to_midplane_verts(verts, s1, s3)
        ed3  = _dist_to_midplane_verts(verts, s2, s3)
        nd2  = ed2 / np.maximum(cell_scale, 1e-6)
        nd3  = ed3 / np.maximum(cell_scale, 1e-6)

        # Proximity to triple point = all three edges close simultaneously
        triple_proximity = (
            _gaussian(norm_dist, NODE_SIGMA) *
            _gaussian(nd2, NODE_SIGMA) *
            _gaussian(nd3, NODE_SIGMA)
        )
        node_bump = NODE_HEIGHT * triple_proximity
        displacement = np.maximum(displacement, node_bump)

    # ── 5. Displace vertices along normals ────────────────────────────────────
    # Smooth the displacement field to avoid high-frequency faceting artefacts
    # by averaging each vertex's value with its neighbours
    vertex_normals = mesh.vertex_normals

    # Build vertex adjacency for smoothing
    edges = mesh.edges_unique
    smooth_disp = displacement.copy()
    for _ in range(3):
        acc   = smooth_disp.copy()
        count = np.ones(len(verts))
        np.add.at(acc,   edges[:, 0], smooth_disp[edges[:, 1]])
        np.add.at(acc,   edges[:, 1], smooth_disp[edges[:, 0]])
        np.add.at(count, edges[:, 0], 1)
        np.add.at(count, edges[:, 1], 1)
        smooth_disp = acc / count

    new_verts = verts + vertex_normals * smooth_disp[:, np.newaxis]

    result = trimesh.Trimesh(
        vertices=new_verts,
        faces=mesh.faces.copy(),
        process=False,
    )

    out = io.BytesIO()
    result.export(out, file_type="stl")
    return out.getvalue()
