"""
Xenonite pattern generator — v7: displacement + hollow cell removal.

Combines the two approaches:
  1. Displace vertices along normals (raised wire ridges, recessed cell panels)
  2. Remove faces in cell interiors (actual through-holes)
  3. Reconnect floating components via BFS through the original face graph

The displacement step runs first, giving the wires 3D structure BEFORE faces
are removed.  This means the remaining edge faces are proper raised ridges, not
flat surface strips — so the result looks like the reference.
"""

import io
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sparse_cc


# ── tuneable defaults ─────────────────────────────────────────────────────────
SEED_DENSITY    = 0.016    # seeds per unit² — doubled from v6 for more/smaller cells
MIN_SEEDS       = 200
MAX_SEEDS       = 2000

RIDGE_HEIGHT    = 0.45     # outward displacement fraction of cell radius
VALLEY_DEPTH    = 0.10     # inward fraction (kept low; holes do the real work now)
RIDGE_SIGMA     = 0.12     # Gaussian width for wire ridge
NODE_SIGMA      = 0.16     # width of triple-point node bump
NODE_HEIGHT     = 0.55     # height of junction node bumps

SMOOTH_PASSES   = 1        # 1 pass prevents spike artefacts without blurring

HOLLOW_THRESH   = 0.52     # faces with norm_dist > this are removed (the holes)
MIN_FRAG_FACES  = 40       # fragments smaller than this get pruned
# ─────────────────────────────────────────────────────────────────────────────


def _dist_to_midplane(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mid    = (a + b) / 2.0
    ab     = b - a
    ab_len = np.linalg.norm(ab, axis=1, keepdims=True)
    ab_len = np.where(ab_len < 1e-9, 1.0, ab_len)
    normal = ab / ab_len
    return np.abs(((pts - mid) * normal).sum(axis=1))


def _gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (x / sigma) ** 2)


def _build_adj(n_faces: int, face_adj_pairs: np.ndarray) -> csr_matrix:
    data = np.ones(len(face_adj_pairs), dtype=np.uint8)
    i, j = face_adj_pairs[:, 0], face_adj_pairs[:, 1]
    return csr_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([i, j]), np.concatenate([j, i]))),
        shape=(n_faces, n_faces),
    )


def _reconnect(orig_adj: csr_matrix, face_mask: np.ndarray,
               min_frag: int) -> np.ndarray:
    """Remove tiny fragments; BFS-reconnect surviving large components to main body."""
    kept = np.where(face_mask)[0]
    if len(kept) == 0:
        return face_mask

    sub  = orig_adj[np.ix_(kept, kept)]
    n_cc, labels = sparse_cc(sub, directed=False)
    if n_cc == 1:
        return face_mask

    sizes     = np.bincount(labels)
    main_comp = int(np.argmax(sizes))
    main_set  = set(kept[labels == main_comp].tolist())

    # prune tiny fragments
    for ci in range(n_cc):
        if ci != main_comp and sizes[ci] < min_frag:
            face_mask[kept[labels == ci]] = False

    # re-query after pruning
    kept  = np.where(face_mask)[0]
    sub   = orig_adj[np.ix_(kept, kept)]
    n_cc, labels = sparse_cc(sub, directed=False)
    if n_cc == 1:
        return face_mask

    sizes     = np.bincount(labels)
    main_comp = int(np.argmax(sizes))
    main_set  = set(kept[labels == main_comp].tolist())

    for ci in range(n_cc):
        if ci == main_comp or sizes[ci] < min_frag:
            continue
        orphans = kept[labels == ci]
        visited = {int(f): None for f in orphans}
        queue   = list(map(int, orphans))
        path    = None
        while queue and path is None:
            nxt = []
            for f in queue:
                for nb in orig_adj.getrow(f).indices:
                    if nb not in visited:
                        visited[nb] = f
                        if nb in main_set:
                            path = [nb]
                            cur  = f
                            while cur is not None:
                                path.append(cur)
                                cur = visited.get(cur)
                            break
                        nxt.append(nb)
                if path:
                    break
            queue = nxt
        if path:
            for f in path:
                face_mask[f] = True
                main_set.add(f)

    return face_mask


def generate_xenonite(stl_bytes: bytes,
                      seed_density: float = SEED_DENSITY,
                      ridge_height: float = RIDGE_HEIGHT,
                      valley_depth: float = VALLEY_DEPTH) -> bytes:

    mesh = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Could not parse mesh from uploaded file.")

    face_areas = mesh.area_faces
    total_area = float(face_areas.sum())

    # ── 1. Seed placement (Poisson-disk relaxation) ───────────────────────────
    n_seeds = int(np.clip(total_area * seed_density, MIN_SEEDS, MAX_SEEDS))
    seed_pts, _ = trimesh.sample.sample_surface(mesh, n_seeds)

    expected_spacing = np.sqrt(total_area / n_seeds)
    tree_s = cKDTree(seed_pts)
    pairs  = tree_s.query_pairs(r=expected_spacing * 0.5)
    remove = set()
    for a, b in pairs:
        if a not in remove:
            remove.add(b)
    keep_m = np.ones(n_seeds, dtype=bool)
    keep_m[list(remove)] = False
    seeds   = seed_pts[keep_m]
    n_seeds = len(seeds)

    tree = cKDTree(seeds)
    k    = min(3, n_seeds)

    # ── 2. Vertex displacement ────────────────────────────────────────────────
    verts  = np.array(mesh.vertices, dtype=np.float64)
    dists, nn = tree.query(verts, k=k)

    s1 = seeds[nn[:, 0]]
    s2 = seeds[nn[:, 1]]

    edge_dist  = _dist_to_midplane(verts, s1, s2)
    cell_scale = dists[:, 1]
    norm_dist  = edge_dist / np.maximum(cell_scale, 1e-6)

    actual_ridge  = ridge_height * cell_scale
    actual_valley = valley_depth * cell_scale

    disp = (actual_ridge  * _gaussian(norm_dist, RIDGE_SIGMA)
            - actual_valley * (1.0 - _gaussian(norm_dist, 0.45)))

    if k >= 3:
        s3  = seeds[nn[:, 2]]
        nd2 = _dist_to_midplane(verts, s1, s3) / np.maximum(cell_scale, 1e-6)
        nd3 = _dist_to_midplane(verts, s2, s3) / np.maximum(cell_scale, 1e-6)
        node_bump = (NODE_HEIGHT * cell_scale *
                     _gaussian(norm_dist, NODE_SIGMA) *
                     _gaussian(nd2,       NODE_SIGMA) *
                     _gaussian(nd3,       NODE_SIGMA))
        disp = np.maximum(disp, node_bump)

    if SMOOTH_PASSES > 0:
        eu = mesh.edges_unique
        for _ in range(SMOOTH_PASSES):
            acc   = disp.copy()
            cnt   = np.ones(len(verts))
            np.add.at(acc, eu[:, 0], disp[eu[:, 1]])
            np.add.at(acc, eu[:, 1], disp[eu[:, 0]])
            np.add.at(cnt, eu[:, 0], 1)
            np.add.at(cnt, eu[:, 1], 1)
            disp = acc / cnt

    new_verts = verts + mesh.vertex_normals * disp[:, np.newaxis]

    # ── 3. Face mask: remove cell-interior faces (the through-holes) ──────────
    fc    = mesh.triangles_center
    fd, fnn = tree.query(fc, k=k)
    fs1   = seeds[fnn[:, 0]]
    fs2   = seeds[fnn[:, 1]]
    f_nd  = (_dist_to_midplane(fc, fs1, fs2)
             / np.maximum(fd[:, 1], 1e-6))

    if k >= 3:
        fs3  = seeds[fnn[:, 2]]
        fnd2 = _dist_to_midplane(fc, fs1, fs3) / np.maximum(fd[:, 1], 1e-6)
        fnd3 = _dist_to_midplane(fc, fs2, fs3) / np.maximum(fd[:, 1], 1e-6)
        # triple-point nodes: keep regardless of primary edge distance
        at_node = ((_gaussian(f_nd,  NODE_SIGMA) *
                    _gaussian(fnd2, NODE_SIGMA) *
                    _gaussian(fnd3, NODE_SIGMA)) > 0.15)
    else:
        at_node = np.zeros(len(fc), dtype=bool)

    face_mask = (f_nd < HOLLOW_THRESH) | at_node

    # ── 4. Reconnect + fragment removal ───────────────────────────────────────
    orig_adj  = _build_adj(len(mesh.faces), mesh.face_adjacency)
    face_mask = _reconnect(orig_adj, face_mask, MIN_FRAG_FACES)

    kept = np.where(face_mask)[0]
    if len(kept) == 0:
        raise ValueError("No faces survived the filter.")

    # ── 5. Build result with displaced vertices ───────────────────────────────
    result = trimesh.Trimesh(
        vertices=new_verts,
        faces=mesh.faces[kept],
        process=False,
    )
    result.remove_unreferenced_vertices()

    # Remove any faces that create non-manifold edges (shared by >2 faces)
    fa = result.faces
    edge_map: dict = {}
    for fi, face in enumerate(fa):
        for i in range(3):
            e = (min(int(face[i]), int(face[(i+1)%3])),
                 max(int(face[i]), int(face[(i+1)%3])))
            edge_map.setdefault(e, []).append(fi)
    bad = {fi for fis in edge_map.values() if len(fis) > 2 for fi in fis[2:]}
    if bad:
        keep2 = np.array([i for i in range(len(fa)) if i not in bad])
        result = trimesh.Trimesh(vertices=result.vertices,
                                 faces=result.faces[keep2], process=False)
        result.remove_unreferenced_vertices()

    out = io.BytesIO()
    result.export(out, file_type="stl")
    return out.getvalue()
