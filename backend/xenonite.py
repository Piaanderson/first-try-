"""
Xenonite pattern generator.

Algorithm:
  1. Load the input mesh and extract all surface triangles.
  2. Sample a seed point per N triangles (controls pattern density).
  3. Build a Delaunay triangulation of the seed points projected into a
     local tangent frame — this gives us the connectivity graph whose
     dual is the Voronoi diagram.
  4. For every Delaunay edge whose two seed points are "close enough" on
     the surface, create a thin tube (cylinder) between them.
  5. Add a small sphere at every junction node.
  6. Merge all geometry and export as a binary STL.
"""

import io
import numpy as np
import trimesh
from scipy.spatial import Delaunay, cKDTree


# ── tuneable parameters ───────────────────────────────────────────────────────
SEED_RATIO      = 0.04   # fraction of faces that get a seed point (capped by MAX_SEEDS)
MAX_SEEDS       = 600    # hard cap — keeps output file size manageable
TUBE_RADIUS     = 0.18   # radius of connecting tubes (in model units)
NODE_RADIUS     = 0.32   # radius of junction spheres
MAX_EDGE_FACTOR = 3.5    # edges longer than factor*median_edge are pruned
TUBE_SEGMENTS   = 6      # cross-section segments per tube (keep low for speed)
NODE_SEGMENTS   = 1      # subdivision level for node spheres (1=80 faces, 3=1280)
# ─────────────────────────────────────────────────────────────────────────────


def _cylinder_between(p1: np.ndarray, p2: np.ndarray,
                      radius: float, sections: int) -> trimesh.Trimesh:
    """Return a cylinder mesh connecting two 3-D points."""
    vec   = p2 - p1
    length = float(np.linalg.norm(vec))
    if length < 1e-8:
        return trimesh.Trimesh()

    cyl = trimesh.creation.cylinder(
        radius=radius, height=length, sections=sections
    )
    # cylinder is centred at origin along Z; rotate then translate
    z_axis = np.array([0.0, 0.0, 1.0])
    axis   = vec / length
    cross  = np.cross(z_axis, axis)
    dot    = np.dot(z_axis, axis)

    if np.linalg.norm(cross) < 1e-8:
        # parallel or anti-parallel
        if dot < 0:
            cyl.apply_transform(trimesh.transformations.rotation_matrix(
                np.pi, [1, 0, 0]))
    else:
        angle = np.arccos(np.clip(dot, -1, 1))
        cyl.apply_transform(
            trimesh.transformations.rotation_matrix(angle, cross / np.linalg.norm(cross))
        )

    mid = (p1 + p2) / 2.0
    cyl.apply_translation(mid)
    return cyl


def _sphere_at(center: np.ndarray, radius: float, subdivisions: int) -> trimesh.Trimesh:
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    sphere.apply_translation(center)
    return sphere


def generate_xenonite(stl_bytes: bytes,
                      seed_ratio: float = SEED_RATIO,
                      tube_radius: float = TUBE_RADIUS,
                      node_radius: float = NODE_RADIUS) -> bytes:
    """
    Accept raw STL bytes, return xenonite-pattern STL bytes.
    """
    mesh = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Could not parse mesh from uploaded file.")

    # ── 1. Sample seed points on face centroids ───────────────────────────────
    face_centers = mesh.triangles_center          # (N, 3)
    n_faces      = len(face_centers)
    n_seeds      = max(50, min(MAX_SEEDS, int(n_faces * seed_ratio)))

    rng     = np.random.default_rng(42)
    indices = rng.choice(n_faces, size=n_seeds, replace=False)
    seeds   = face_centers[indices]               # (n_seeds, 3)

    # ── 2. Delaunay triangulation of seeds ────────────────────────────────────
    # Project to 2-D for triangulation using PCA of seed cloud
    centroid = seeds.mean(axis=0)
    _, _, Vt = np.linalg.svd(seeds - centroid)
    seeds_2d = (seeds - centroid) @ Vt[:2].T      # (n_seeds, 2)

    tri = Delaunay(seeds_2d)

    # Collect unique edges from Delaunay simplices
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
            edges.add((min(a, b), max(a, b)))

    # ── 3. Prune long edges ───────────────────────────────────────────────────
    edge_list    = np.array(list(edges))
    edge_lengths = np.linalg.norm(seeds[edge_list[:, 0]] - seeds[edge_list[:, 1]], axis=1)
    median_len   = float(np.median(edge_lengths))
    keep         = edge_lengths < MAX_EDGE_FACTOR * median_len
    edge_list    = edge_list[keep]

    # ── 4. Build tubes and nodes ──────────────────────────────────────────────
    parts: list[trimesh.Trimesh] = []

    used_nodes: set[int] = set()
    for a, b in edge_list:
        cyl = _cylinder_between(seeds[a], seeds[b], tube_radius, TUBE_SEGMENTS)
        if len(cyl.vertices):
            parts.append(cyl)
        used_nodes.add(a)
        used_nodes.add(b)

    for idx in used_nodes:
        sphere = _sphere_at(seeds[idx], node_radius, NODE_SEGMENTS)
        parts.append(sphere)

    if not parts:
        raise ValueError("No geometry was generated — try adjusting density.")

    # ── 5. Merge and export ───────────────────────────────────────────────────
    result = trimesh.util.concatenate(parts)
    out    = io.BytesIO()
    result.export(out, file_type="stl")
    return out.getvalue()
