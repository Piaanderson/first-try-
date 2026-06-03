"""
Xenonite pattern generator — v8: displacement + hollow + watertight shell.

After displacement and face removal, the wire surface has open boundary edges
(each hole edge is shared by only 1 face).  We close them by:
  1. Creating an inner surface (vertices offset inward by SHELL_THICKNESS)
  2. Adding wall quads (2 triangles each) along every boundary edge loop
This produces a closed, manifold, printable mesh.
"""

import io
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sparse_cc


# ── tuneable defaults ─────────────────────────────────────────────────────────
SEED_DENSITY    = 0.016
MIN_SEEDS       = 200
MAX_SEEDS       = 2000

RIDGE_HEIGHT    = 0.45
VALLEY_DEPTH    = 0.10
RIDGE_SIGMA     = 0.12
NODE_SIGMA      = 0.16
NODE_HEIGHT     = 0.55
SMOOTH_PASSES   = 1

HOLLOW_THRESH   = 0.52     # faces with norm_dist > this become holes
MIN_FRAG_FACES  = 40

SHELL_THICKNESS = 0.4      # mm — inner surface offset; gives wires wall geometry
# ─────────────────────────────────────────────────────────────────────────────


def _dist_to_midplane(pts, a, b):
    mid    = (a + b) / 2.0
    ab     = b - a
    ab_len = np.linalg.norm(ab, axis=1, keepdims=True)
    ab_len = np.where(ab_len < 1e-9, 1.0, ab_len)
    return np.abs(((pts - mid) * (ab / ab_len)).sum(axis=1))


def _gaussian(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2)


def _build_adj(n, pairs):
    d = np.ones(len(pairs), dtype=np.uint8)
    i, j = pairs[:, 0], pairs[:, 1]
    return csr_matrix(
        (np.concatenate([d, d]),
         (np.concatenate([i, j]), np.concatenate([j, i]))),
        shape=(n, n))


def _reconnect(orig_adj, face_mask, min_frag):
    kept = np.where(face_mask)[0]
    if not len(kept):
        return face_mask
    sub = orig_adj[np.ix_(kept, kept)]
    n_cc, labels = sparse_cc(sub, directed=False)
    if n_cc == 1:
        return face_mask
    sizes = np.bincount(labels)
    main_comp = int(np.argmax(sizes))
    main_set = set(kept[labels == main_comp].tolist())
    for ci in range(n_cc):
        if ci != main_comp and sizes[ci] < min_frag:
            face_mask[kept[labels == ci]] = False
    kept = np.where(face_mask)[0]
    sub = orig_adj[np.ix_(kept, kept)]
    n_cc, labels = sparse_cc(sub, directed=False)
    if n_cc == 1:
        return face_mask
    sizes = np.bincount(labels)
    main_comp = int(np.argmax(sizes))
    main_set = set(kept[labels == main_comp].tolist())
    for ci in range(n_cc):
        if ci == main_comp or sizes[ci] < min_frag:
            continue
        orphans = kept[labels == ci]
        visited = {int(f): None for f in orphans}
        queue = list(map(int, orphans))
        path = None
        while queue and path is None:
            nxt = []
            for f in queue:
                for nb in orig_adj.getrow(f).indices:
                    if nb not in visited:
                        visited[nb] = f
                        if nb in main_set:
                            path, cur = [nb], f
                            while cur is not None:
                                path.append(cur); cur = visited.get(cur)
                            break
                        nxt.append(nb)
                if path: break
            queue = nxt
        if path:
            for f in path:
                face_mask[f] = True; main_set.add(f)
    return face_mask


def _thicken(outer_verts, outer_normals, faces, thickness):
    """
    Build a watertight shell from an open surface mesh.

    For every boundary edge (shared by exactly 1 face), add two wall triangles
    connecting the outer and inner surfaces, making each edge shared by 2 faces.
    """
    n = len(outer_verts)
    inner_verts = outer_verts - outer_normals * thickness

    # ── find boundary edges ───────────────────────────────────────────────────
    from collections import defaultdict
    edge_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            edge_faces[(min(a, b), max(a, b))].append((fi, a, b))

    # boundary edge: appears in exactly 1 face → we know the winding (a→b)
    wall_tris = []
    for (va, vb), entries in edge_faces.items():
        if len(entries) != 1:
            continue
        _, a, b = entries[0]   # directed edge a→b as it appears in the outer face
        # Wall quad: outer_a, outer_b, inner_b, inner_a  (two triangles)
        # Winding chosen so normals point outward (away from the shell body)
        wall_tris.append([a,     b,     b + n])
        wall_tris.append([a,     b + n, a + n])

    # ── assemble ──────────────────────────────────────────────────────────────
    all_verts = np.vstack([outer_verts, inner_verts])
    inner_faces = faces[:, ::-1] + n          # flip winding so normals point inward
    all_faces = np.vstack([
        faces,
        inner_faces,
        np.array(wall_tris, dtype=np.int32) if wall_tris
            else np.empty((0, 3), dtype=np.int32),
    ])

    return trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)


def generate_xenonite(stl_bytes, seed_density=SEED_DENSITY,
                      ridge_height=RIDGE_HEIGHT, valley_depth=VALLEY_DEPTH):

    mesh = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Could not parse mesh.")

    face_areas = mesh.area_faces
    total_area = float(face_areas.sum())

    # ── 1. Seeds ──────────────────────────────────────────────────────────────
    n_seeds = int(np.clip(total_area * seed_density, MIN_SEEDS, MAX_SEEDS))
    seed_pts, _ = trimesh.sample.sample_surface(mesh, n_seeds)
    expected_spacing = np.sqrt(total_area / n_seeds)
    pairs = cKDTree(seed_pts).query_pairs(r=expected_spacing * 0.5)
    remove = set()
    for a, b in pairs:
        if a not in remove: remove.add(b)
    keep_m = np.ones(n_seeds, dtype=bool); keep_m[list(remove)] = False
    seeds = seed_pts[keep_m]; n_seeds = len(seeds)
    tree = cKDTree(seeds); k = min(3, n_seeds)

    # ── 2. Vertex displacement ────────────────────────────────────────────────
    verts = np.array(mesh.vertices, dtype=np.float64)
    dists, nn = tree.query(verts, k=k)
    s1, s2 = seeds[nn[:, 0]], seeds[nn[:, 1]]
    edge_dist  = _dist_to_midplane(verts, s1, s2)
    cell_scale = dists[:, 1]
    norm_dist  = edge_dist / np.maximum(cell_scale, 1e-6)

    disp = (ridge_height * cell_scale * _gaussian(norm_dist, RIDGE_SIGMA)
            - valley_depth * cell_scale * (1.0 - _gaussian(norm_dist, 0.45)))

    if k >= 3:
        s3  = seeds[nn[:, 2]]
        nd2 = _dist_to_midplane(verts, s1, s3) / np.maximum(cell_scale, 1e-6)
        nd3 = _dist_to_midplane(verts, s2, s3) / np.maximum(cell_scale, 1e-6)
        disp = np.maximum(disp,
               NODE_HEIGHT * cell_scale
               * _gaussian(norm_dist, NODE_SIGMA)
               * _gaussian(nd2, NODE_SIGMA)
               * _gaussian(nd3, NODE_SIGMA))

    if SMOOTH_PASSES > 0:
        eu = mesh.edges_unique
        for _ in range(SMOOTH_PASSES):
            acc = disp.copy(); cnt = np.ones(len(verts))
            np.add.at(acc, eu[:, 0], disp[eu[:, 1]])
            np.add.at(acc, eu[:, 1], disp[eu[:, 0]])
            np.add.at(cnt, eu[:, 0], 1); np.add.at(cnt, eu[:, 1], 1)
            disp = acc / cnt

    new_verts = verts + mesh.vertex_normals * disp[:, np.newaxis]

    # ── 3. Face mask (remove cell interiors) ──────────────────────────────────
    fc = mesh.triangles_center
    fd, fnn = tree.query(fc, k=k)
    f_nd = (_dist_to_midplane(fc, seeds[fnn[:, 0]], seeds[fnn[:, 1]])
            / np.maximum(fd[:, 1], 1e-6))

    at_node = np.zeros(len(fc), dtype=bool)
    if k >= 3:
        fnd2 = _dist_to_midplane(fc, seeds[fnn[:, 0]], seeds[fnn[:, 2]]) / np.maximum(fd[:, 1], 1e-6)
        fnd3 = _dist_to_midplane(fc, seeds[fnn[:, 1]], seeds[fnn[:, 2]]) / np.maximum(fd[:, 1], 1e-6)
        at_node = (_gaussian(f_nd, NODE_SIGMA)
                   * _gaussian(fnd2, NODE_SIGMA)
                   * _gaussian(fnd3, NODE_SIGMA)) > 0.15

    face_mask = (f_nd < HOLLOW_THRESH) | at_node

    orig_adj  = _build_adj(len(mesh.faces), mesh.face_adjacency)
    face_mask = _reconnect(orig_adj, face_mask, MIN_FRAG_FACES)

    kept = np.where(face_mask)[0]
    if not len(kept):
        raise ValueError("No faces survived the filter.")

    # ── 4. Build open surface with displaced vertices ─────────────────────────
    open_mesh = trimesh.Trimesh(vertices=new_verts, faces=mesh.faces[kept], process=False)
    open_mesh.remove_unreferenced_vertices()

    # ── 5. Thicken into watertight shell ──────────────────────────────────────
    result = _thicken(
        np.array(open_mesh.vertices),
        np.array(open_mesh.vertex_normals),
        np.array(open_mesh.faces),
        thickness=SHELL_THICKNESS,
    )
    result.remove_unreferenced_vertices()

    out = io.BytesIO()
    result.export(out, file_type="stl")
    return out.getvalue()
