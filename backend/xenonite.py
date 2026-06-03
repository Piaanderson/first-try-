"""
Xenonite pattern generator — v3.

Fixes vs v2:
  - 4× seed density so cells are smaller and detail is preserved
  - Connectivity preservation: after masking, detect disconnected components
    and add back the minimum bridging faces needed to reconnect them to the
    main body (no more floating limbs)
  - Manifold repair: dilate the wire mask by one face ring before cutting,
    so strip edges are clean and non-manifold T-junctions don't form
  - Tiny fragment removal: floating chips smaller than MIN_FRAG_FACES are pruned

Algorithm:
  1. Scatter Voronoi seeds on face centroids (weighted by area).
  2. For each face, find 2 nearest seeds and compute distance to their
     bisector plane → faces within wire_width of a bisector are KEPT.
  3. Also keep faces near triple-point junctions (the circular nodes).
  4. Dilate the keep-mask by one adjacency ring to clean up edges.
  5. Rebuild mesh, remove fragments < MIN_FRAG_FACES.
  6. Bridge any disconnected component back to the main body.
"""

import io
import numpy as np
import trimesh
import trimesh.graph
from scipy.sparse.csgraph import connected_components as sparse_cc
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


# ── tuneable defaults ─────────────────────────────────────────────────────────
SEED_DENSITY    = 0.005    # seeds per unit² of surface area (4× denser than v2)
MIN_SEEDS       = 200
MAX_SEEDS       = 2000
WIRE_WIDTH      = 0.28     # fraction of avg cell radius kept as wire
NODE_FACTOR     = 1.8      # triple-point nodes are wider
DILATE_RINGS    = 1        # adjacency rings to dilate before cutting
MIN_FRAG_FACES  = 30       # fragments smaller than this are pruned entirely
# ─────────────────────────────────────────────────────────────────────────────


def _dist_to_midplane(points: np.ndarray,
                      a: np.ndarray,
                      b: np.ndarray) -> np.ndarray:
    """Per-point distance to the bisector plane between seed pairs a[i], b[i]."""
    mid    = (a + b) / 2.0
    ab     = b - a
    ab_len = np.linalg.norm(ab, axis=1, keepdims=True)
    ab_len = np.where(ab_len < 1e-9, 1.0, ab_len)
    normal = ab / ab_len
    return np.abs(((points - mid) * normal).sum(axis=1))


def _build_face_adjacency_matrix(mesh: trimesh.Trimesh) -> csr_matrix:
    """Sparse adjacency matrix over faces (faces sharing an edge are neighbours)."""
    n    = len(mesh.faces)
    adj  = mesh.face_adjacency          # (E, 2) pairs of adjacent face indices
    data = np.ones(len(adj), dtype=np.uint8)
    mat  = csr_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([adj[:, 0], adj[:, 1]]),
          np.concatenate([adj[:, 1], adj[:, 0]]))),
        shape=(n, n),
    )
    return mat


def _dilate_mask(mask: np.ndarray, adj: csr_matrix, rings: int) -> np.ndarray:
    """Expand a boolean face mask by `rings` adjacency steps (sparse, no dense conversion)."""
    m = mask.copy()
    for _ in range(rings):
        kept_idx = np.where(m)[0]
        if len(kept_idx) == 0:
            break
        # For each kept face, get its neighbours without densifying
        for start in range(0, len(kept_idx), 4096):
            chunk = kept_idx[start:start + 4096]
            rows  = adj[chunk]           # sparse slice, shape (chunk, n)
            # nonzero column indices = neighbour face indices
            _, cols = rows.nonzero()
            m[cols] = True
    return m


def _reconnect(mesh: trimesh.Trimesh,
               face_mask: np.ndarray,
               min_frag: int) -> np.ndarray:
    """
    Remove tiny floating fragments and reconnect any large disconnected
    component to the main body by adding back the shortest face-path between
    them in the ORIGINAL mesh adjacency graph.
    """
    n_faces = len(mesh.faces)
    orig_adj = _build_face_adjacency_matrix(mesh)

    def get_components(mask):
        sub = orig_adj[np.ix_(mask, mask)]
        n_comp, labels = sparse_cc(sub, directed=False)
        return n_comp, labels

    kept_idx = np.where(face_mask)[0]
    if len(kept_idx) == 0:
        return face_mask

    n_comp, labels = get_components(kept_idx)

    if n_comp == 1:
        # Already connected — just prune tiny fragments
        return face_mask

    # Compute component sizes
    comp_sizes = np.bincount(labels)
    main_comp  = int(np.argmax(comp_sizes))

    # --- prune fragments smaller than min_frag ---
    for ci in range(n_comp):
        if comp_sizes[ci] < min_frag and ci != main_comp:
            face_mask[kept_idx[labels == ci]] = False

    # --- reconnect surviving large components ---
    kept_idx = np.where(face_mask)[0]
    n_comp, labels = get_components(kept_idx)
    comp_sizes = np.bincount(labels)
    main_comp  = int(np.argmax(comp_sizes))
    main_faces = set(kept_idx[labels == main_comp].tolist())

    for ci in range(n_comp):
        if ci == main_comp or comp_sizes[ci] < min_frag:
            continue
        # BFS in original face adjacency from this component to main
        orphan_faces = kept_idx[labels == ci]
        visited = {f: None for f in orphan_faces}
        queue = list(orphan_faces)
        found_path = None

        while queue and found_path is None:
            nxt = []
            for f in queue:
                nbrs = orig_adj.getrow(f).indices
                for nb in nbrs:
                    if nb not in visited:
                        visited[nb] = f
                        if nb in main_faces:
                            # Trace path back
                            path = [nb]
                            cur  = f
                            while cur is not None:
                                path.append(cur)
                                cur = visited.get(cur)
                            found_path = path
                            break
                        nxt.append(nb)
                if found_path:
                    break
            queue = nxt

        if found_path:
            for f in found_path:
                face_mask[f] = True
                main_faces.add(f)

    return face_mask


def generate_xenonite(stl_bytes: bytes,
                      seed_density: float = SEED_DENSITY,
                      wire_width: float = WIRE_WIDTH) -> bytes:
    mesh = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Could not parse mesh from uploaded file.")

    # ── 1. Sample seeds weighted by face area ─────────────────────────────────
    face_centers = mesh.triangles_center
    face_areas   = mesh.area_faces
    total_area   = float(face_areas.sum())

    n_seeds = int(np.clip(
        total_area * seed_density,
        MIN_SEEDS, MAX_SEEDS
    ))

    rng   = np.random.default_rng(42)
    probs = face_areas / total_area
    seed_face_idx = rng.choice(len(face_centers), size=n_seeds, replace=False, p=probs)
    seeds = face_centers[seed_face_idx]

    # ── 2. Per-face distance to nearest Voronoi boundary ──────────────────────
    k     = min(3, n_seeds)
    tree  = cKDTree(seeds)
    dists, nn = tree.query(face_centers, k=k)

    s1 = seeds[nn[:, 0]]
    s2 = seeds[nn[:, 1]]

    edge_dist  = _dist_to_midplane(face_centers, s1, s2)
    cell_scale = dists[:, 1]
    threshold  = wire_width * cell_scale

    face_mask = edge_dist < threshold

    # ── 3. Add triple-point nodes ─────────────────────────────────────────────
    if k >= 3:
        s3         = seeds[nn[:, 2]]
        ed2        = _dist_to_midplane(face_centers, s1, s3)
        ed3        = _dist_to_midplane(face_centers, s2, s3)
        node_thresh = NODE_FACTOR * threshold
        face_mask  |= (edge_dist < node_thresh) & (ed2 < node_thresh) & (ed3 < node_thresh)

    # ── 4. Dilate mask to clean up jagged edges & prevent non-manifold strips ─
    adj_mat   = _build_face_adjacency_matrix(mesh)
    face_mask = _dilate_mask(face_mask, adj_mat, DILATE_RINGS)

    # ── 5. Reconnect floating pieces & prune tiny fragments ───────────────────
    face_mask = _reconnect(mesh, face_mask, MIN_FRAG_FACES)

    # ── 6. Build result mesh ──────────────────────────────────────────────────
    kept = np.where(face_mask)[0]
    if len(kept) == 0:
        raise ValueError("No faces survived the filter — try increasing wire_width.")

    result = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces[kept],
        process=False,
    )
    result.remove_unreferenced_vertices()

    # Basic cleanup — merge duplicate vertices, fix winding consistency
    result.merge_vertices()
    trimesh.repair.fix_normals(result)

    # Remove any remaining non-manifold faces (edges shared by >2 faces)
    from collections import Counter as _Counter
    face_arr = result.faces
    edge_face: dict = {}
    for fi, face in enumerate(face_arr):
        for i in range(3):
            e = tuple(sorted([int(face[i]), int(face[(i + 1) % 3])]))
            edge_face.setdefault(e, []).append(fi)
    bad = set()
    for fis in edge_face.values():
        if len(fis) > 2:
            for fi in fis[2:]:
                bad.add(fi)
    if bad:
        keep2 = np.array([i for i in range(len(face_arr)) if i not in bad])
        result = trimesh.Trimesh(vertices=result.vertices,
                                 faces=result.faces[keep2], process=False)
        result.remove_unreferenced_vertices()

    out = io.BytesIO()
    result.export(out, file_type="stl")
    return out.getvalue()
