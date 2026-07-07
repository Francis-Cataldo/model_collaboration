import numpy as np
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================

# Change process time below
LLM_COMPUTE_SEC = 0.0
RANKER_COMPUTE_SEC = 0.0
FUSER_COMPUTE_SEC = 0.0

R_EARTH = 6371.0
ALTITUDE = 530.0
ORBIT_R = R_EARTH + ALTITUDE

SATELLITE_SPEED = 8.0
SPEED_UNIT = "km/s"  # LEO speed is treated as 8 km/s
SPEED_KM_PER_SEC = SATELLITE_SPEED / 3600.0 if SPEED_UNIT == "km/h" else SATELLITE_SPEED
OMEGA = SPEED_KM_PER_SEC / ORBIT_R

INCLINATION_DEG = 53.0
N_PLANES = 72
M_PER_ORBIT = 20  # Number of satellites on one orbit
L = 3             # Number of pool LLMs

# This is now the direct physical staggering angle between neighboring planes.
# Example: 82 means plane p+1 is shifted by +82 degrees relative to plane p.
STAGGERING_ANGLE_DEG = 8.0

USER_LAT_DEG = 0.0
USER_LON_DEG = 90.0
CONE_HALF_ANGLE_DEG = 22.5

# More frames = longer animation if DT is unchanged.
# Total simulated time = N_FRAMES * DT seconds.
N_FRAMES = 300
DT = 3.0
FRAME_DURATION_MS = 100

C_LIGHT_KM_PER_SEC = 299792.458

DRAW_OPTIMAL_ROUTE = True
DRAW_ARROW_HEADS = False
ROUTE_LINE_WIDTH = 3
ROUTE_HEAD_SIZE = 45.0
ROUTE_SELECTED_SIZE = 10

EARTH_OPACITY = 1
ORBIT_OPACITY = 0.30
ORBIT_WIDTH = 1.2
USER_MARKER_SIZE = 7
OUTSIDE_SIZE = 3.2
INSIDE_SIZE = 6.5

RANDOM_SEED = 7
rng = np.random.default_rng(RANDOM_SEED)

# ============================================================
# DERIVED CONFIG
# ============================================================

if M_PER_ORBIT % (L + 2) != 0:
    raise ValueError(f"M_PER_ORBIT={M_PER_ORBIT} must be divisible by L+2={L+2}.")

K_BLOCKS = M_PER_ORBIT // (L + 2)
N_LLM_PER_ORBIT = L * K_BLOCKS
N_RANKER_PER_ORBIT = K_BLOCKS
N_FUSER_PER_ORBIT = K_BLOCKS

CENTER_PLANE_INDEX = N_PLANES // 2
SLOT_ANGLE_DEG = 360.0 / M_PER_ORBIT
SLOT_ANGLE_RAD = np.deg2rad(SLOT_ANGLE_DEG)
STAGGERING_ANGLE_RAD = np.deg2rad(STAGGERING_ANGLE_DEG)

# Center plane starts at 0 degrees.
# Each neighboring plane is physically shifted by STAGGERING_ANGLE_DEG.
PLANE_PHASE_OFFSETS = np.array([
    (p - CENTER_PLANE_INDEX) * STAGGERING_ANGLE_RAD
    for p in range(N_PLANES)
])

# ============================================================
# GEOMETRY
# ============================================================

def unit(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def raan_for_plane(p):
    # For inclined orbits, RAAN and RAAN+180 deg are different planes.
    # Therefore N_PLANES spans 0..360 deg.
    return 2 * np.pi * p / N_PLANES


def raan_spacing_deg():
    return 360.0 / N_PLANES


def ground_user_position(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return R_EARTH * np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ])


def orbit_position(r, inc_rad, raan_rad, u_rad):
    cO, sO = np.cos(raan_rad), np.sin(raan_rad)
    cu, su = np.cos(u_rad), np.sin(u_rad)
    ci, si = np.cos(inc_rad), np.sin(inc_rad)

    return np.array([
        r * (cO * cu - sO * su * ci),
        r * (sO * cu + cO * su * ci),
        r * (su * si)
    ])


def earth_surface(n_lon=80, n_lat=40):
    lon = np.linspace(0, 2 * np.pi, n_lon)
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    return (
        R_EARTH * np.cos(lat_grid) * np.cos(lon_grid),
        R_EARTH * np.cos(lat_grid) * np.sin(lon_grid),
        R_EARTH * np.sin(lat_grid)
    )


def orbit_curve(raan_rad, n_points=320):
    inc = np.deg2rad(INCLINATION_DEG)
    u_vals = np.linspace(0, 2 * np.pi, n_points)
    return np.array([orbit_position(ORBIT_R, inc, raan_rad, u) for u in u_vals])


def roles_for_plane(plane_index):
    """
    Dynamic role assignment for arbitrary L.

    Each block has length L+2:
        1 ranker
        L LLMs
        1 fuser

    Repeated K_BLOCKS times around the orbit.
    """

    block_len = L + 2

    if M_PER_ORBIT % block_len != 0:
        raise ValueError(
            f"M_PER_ORBIT={M_PER_ORBIT} must be divisible by L+2={block_len}. "
            f"Change M_PER_ORBIT or L."
        )

    roles = np.array(["LLM"] * M_PER_ORBIT, dtype=object)

    # Put ranker at the beginning of each block.
    # Put fuser roughly in the middle of each block.
    fuser_offset = int(np.ceil(block_len / 2))

    for start in range(0, M_PER_ORBIT, block_len):
        roles[start] = "R"
        roles[start + fuser_offset] = "F"

    return roles


def constellation_snapshot(t):
    inc = np.deg2rad(INCLINATION_DEG)
    positions, roles = [], []

    for p in range(N_PLANES):
        raan = raan_for_plane(p)
        plane_phase = PLANE_PHASE_OFFSETS[p]
        role_tags = roles_for_plane(p)

        for s in range(M_PER_ORBIT):
            u0 = 2 * np.pi * s / M_PER_ORBIT + plane_phase
            u = OMEGA * t + u0
            positions.append(orbit_position(ORBIT_R, inc, raan, u))
            roles.append(role_tags[s])

    return np.array(positions), np.array(roles, dtype=object)


def inside_centered_cone(user_pos, sat_positions, cone_half_angle_deg):
    """
    True if a satellite is inside the Earth-centered cone.

    The cone axis is the Earth-center-to-user direction.
    This exactly matches the gold cap boundary drawn on the orbital shell.
    """
    axis = unit(user_pos)
    cos_limit = np.cos(np.deg2rad(cone_half_angle_deg))
    sat_hats = sat_positions / np.linalg.norm(sat_positions, axis=1, keepdims=True)
    return (sat_hats @ axis) >= cos_limit


def cap_plane_basis(axis):
    temp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, temp)) > 0.9:
        temp = np.array([0.0, 1.0, 0.0])

    v1 = unit(np.cross(axis, temp))
    v2 = unit(np.cross(axis, v1))
    return v1, v2


def cap_circle_geometry(axis, half_angle_deg, radius=ORBIT_R):
    alpha = np.deg2rad(half_angle_deg)
    v1, v2 = cap_plane_basis(axis)
    center = radius * np.cos(alpha) * axis
    rho = radius * np.sin(alpha)
    return center, rho, v1, v2


def center_cone_mesh(axis, half_angle_deg, length=ORBIT_R * 1.03, n_h=24, n_phi=80):
    v1, v2 = cap_plane_basis(axis)
    alpha = np.deg2rad(half_angle_deg)

    h_vals = np.linspace(0, length, n_h)
    phi_vals = np.linspace(0, 2 * np.pi, n_phi)
    H, PHI = np.meshgrid(h_vals, phi_vals)

    radius = H * np.tan(alpha)

    X = H * axis[0] + radius * np.cos(PHI) * v1[0] + radius * np.sin(PHI) * v2[0]
    Y = H * axis[1] + radius * np.cos(PHI) * v1[1] + radius * np.sin(PHI) * v2[1]
    Z = H * axis[2] + radius * np.cos(PHI) * v1[2] + radius * np.sin(PHI) * v2[2]

    return X, Y, Z


def cap_boundary_circle(axis, half_angle_deg, radius=ORBIT_R, n_pts=240):
    center, rho, v1, v2 = cap_circle_geometry(axis, half_angle_deg, radius)
    theta = np.linspace(0, 2 * np.pi, n_pts)

    return (
        center[None, :]
        + rho * np.cos(theta)[:, None] * v1[None, :]
        + rho * np.sin(theta)[:, None] * v2[None, :]
    )

# ============================================================
# MASKS, COUNTS, AND ROUTE OPTIMIZATION
# ============================================================

def role_masks(roles, inside_cone):
    return {
        "llm_out": (roles == "LLM") & (~inside_cone),
        "llm_in":  (roles == "LLM") & inside_cone,
        "r_out":   (roles == "R") & (~inside_cone),
        "r_in":    (roles == "R") & inside_cone,
        "f_out":   (roles == "F") & (~inside_cone),
        "f_in":    (roles == "F") & inside_cone,
    }


def counts_from_masks(inside_cone, masks):
    return (
        int(np.sum(inside_cone)),
        int(np.sum(masks["llm_in"])),
        int(np.sum(masks["r_in"])),
        int(np.sum(masks["f_in"])),
    )


def pairwise_distance(a, b):
    return np.linalg.norm(a - b, axis=-1)


def optimize_llm_blender_route(user_pos, sat_positions, roles, inside_cone):
    """
    Solve the LLM-Blender routing problem inside the cone only.

    Objective:
        min_{R,F,S, |S|=L}
            max_{i in S} [d(U, i)/c + LLM_compute + d(i, R)/c]
            + RANKER_compute + d(R, F)/c
            + FUSER_compute + d(F, U)/c

    For a fixed ranker R, the best S is the L LLMs with the smallest
    User->LLM->Ranker latency. The bottleneck is the L-th smallest value.
    """

    llm_idx = np.where((roles == "LLM") & inside_cone)[0]
    ranker_idx = np.where((roles == "R") & inside_cone)[0]
    fuser_idx = np.where((roles == "F") & inside_cone)[0]

    if len(llm_idx) < L or len(ranker_idx) == 0 or len(fuser_idx) == 0:
        return None

    best = None

    user_to_llm = pairwise_distance(sat_positions[llm_idx], user_pos) / C_LIGHT_KM_PER_SEC
    fuser_to_user_all = pairwise_distance(sat_positions[fuser_idx], user_pos) / C_LIGHT_KM_PER_SEC

    for r_idx in ranker_idx:
        ranker_pos = sat_positions[r_idx]

        llm_to_ranker = pairwise_distance(sat_positions[llm_idx], ranker_pos) / C_LIGHT_KM_PER_SEC
        llm_stage_times = user_to_llm + LLM_COMPUTE_SEC + llm_to_ranker

        order = np.argsort(llm_stage_times)
        selected_local = order[:L]
        selected_llm_idx = llm_idx[selected_local]
        bottleneck_time = float(np.max(llm_stage_times[selected_local]))

        ranker_to_fuser_all = pairwise_distance(sat_positions[fuser_idx], ranker_pos) / C_LIGHT_KM_PER_SEC

        total_for_fusers = (
            bottleneck_time
            + RANKER_COMPUTE_SEC
            + ranker_to_fuser_all
            + FUSER_COMPUTE_SEC
            + fuser_to_user_all
        )

        best_f_local = int(np.argmin(total_for_fusers))
        total_sec = float(total_for_fusers[best_f_local])
        f_idx = int(fuser_idx[best_f_local])

        if best is None or total_sec < best["total_sec"]:
            best = {
                "total_sec": total_sec,
                "bottleneck_sec": bottleneck_time,
                "ranker_idx": int(r_idx),
                "fuser_idx": f_idx,
                "llm_indices": selected_llm_idx.astype(int),
                "region_n_llm": int(len(llm_idx)),
                "region_n_ranker": int(len(ranker_idx)),
                "region_n_fuser": int(len(fuser_idx)),
            }

    return best

# ============================================================
# PLOT HELPERS
# ============================================================

def segment_xyz(points_a, points_b):
    xs, ys, zs = [], [], []

    for a, b in zip(points_a, points_b):
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]

    return xs, ys, zs


def make_arrow_line_trace(points_a, points_b, color, name, width=ROUTE_LINE_WIDTH):
    x, y, z = segment_xyz(points_a, points_b)

    return go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        showlegend=False,
        hoverinfo="skip"
    )


def make_arrow_head_trace(points_a, points_b, color, name):
    if len(points_a) == 0:
        return go.Cone(
            x=[], y=[], z=[], u=[], v=[], w=[],
            showscale=False,
            showlegend=False,
            name=name,
            hoverinfo="skip"
        )

    starts = np.array(points_a)
    ends = np.array(points_b)
    dirs = ends - starts
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = np.divide(dirs, norms, out=np.zeros_like(dirs), where=norms > 0)

    return go.Cone(
        x=ends[:, 0], y=ends[:, 1], z=ends[:, 2],
        u=dirs[:, 0], v=dirs[:, 1], w=dirs[:, 2],
        sizemode="absolute",
        sizeref=ROUTE_HEAD_SIZE,
        anchor="tip",
        colorscale=[[0.0, color], [1.0, color]],
        opacity=0.65,
        showscale=False,
        showlegend=False,
        name=name,
        hoverinfo="skip"
    )


def build_route_traces(user_pos, sat_positions, route):
    empty_scatter = go.Scatter3d(
        x=[], y=[], z=[],
        mode="markers",
        showlegend=False,
        hoverinfo="skip"
    )
    empty_line = go.Scatter3d(
        x=[], y=[], z=[],
        mode="lines",
        showlegend=False,
        hoverinfo="skip"
    )
    empty_cone = go.Cone(
        x=[], y=[], z=[], u=[], v=[], w=[],
        showscale=False,
        showlegend=False,
        hoverinfo="skip"
    )

    if (not DRAW_OPTIMAL_ROUTE) or route is None:
        return [
            empty_scatter, empty_scatter, empty_scatter,
            empty_line, empty_line, empty_line, empty_line,
            empty_cone, empty_cone, empty_cone, empty_cone
        ]

    llm_ids = np.array(route["llm_indices"], dtype=int)
    r_idx = int(route["ranker_idx"])
    f_idx = int(route["fuser_idx"])

    llm_pts = sat_positions[llm_ids]
    r_pt = sat_positions[r_idx]
    f_pt = sat_positions[f_idx]

    llm_labels = [f"LLM{k + 1}" for k in range(len(llm_ids))]

    selected_llms = go.Scatter3d(
        x=llm_pts[:, 0], y=llm_pts[:, 1], z=llm_pts[:, 2],
        mode="markers+text",
        marker=dict(
            size=ROUTE_SELECTED_SIZE,
            color="royalblue",
            symbol="circle",
            line=dict(color="white", width=4)
        ),
        text=llm_labels,
        textposition="top center",
        name="Selected LLMs",
        showlegend=True,
        hoverinfo="skip"
    )

    selected_ranker = go.Scatter3d(
        x=[r_pt[0]], y=[r_pt[1]], z=[r_pt[2]],
        mode="markers+text",
        marker=dict(
            size=ROUTE_SELECTED_SIZE + 2,
            color="limegreen",
            symbol="square",
            line=dict(color="white", width=4)
        ),
        text=["Ranker*"],
        textposition="top center",
        name="Selected Ranker",
        showlegend=True,
        hoverinfo="skip"
    )

    selected_fuser = go.Scatter3d(
        x=[f_pt[0]], y=[f_pt[1]], z=[f_pt[2]],
        mode="markers+text",
        marker=dict(
            size=ROUTE_SELECTED_SIZE + 2,
            color="darkorchid",
            symbol="diamond",
            line=dict(color="white", width=4)
        ),
        text=["Fuser*"],
        textposition="top center",
        name="Selected Fuser",
        showlegend=True,
        hoverinfo="skip"
    )

    # User -> LLMs
    u_starts = [user_pos] * len(llm_pts)
    u_ends = list(llm_pts)

    # LLMs -> Ranker
    lr_starts = list(llm_pts)
    lr_ends = [r_pt] * len(llm_pts)

    # Ranker -> Fuser
    rf_starts = [r_pt]
    rf_ends = [f_pt]

    # Fuser -> User
    fu_starts = [f_pt]
    fu_ends = [user_pos]

    line_user_llms = make_arrow_line_trace(u_starts, u_ends, "blue", "User to LLMs", width=3)
    line_llms_ranker = make_arrow_line_trace(lr_starts, lr_ends, "teal", "LLMs to Ranker", width=3)
    line_ranker_fuser = make_arrow_line_trace(rf_starts, rf_ends, "green", "Ranker to Fuser", width=5)
    line_fuser_user = make_arrow_line_trace(fu_starts, fu_ends, "purple", "Fuser to User", width=5)

    if DRAW_ARROW_HEADS:
        head_user_llms = make_arrow_head_trace(u_starts, u_ends, "blue", "User to LLM arrowheads")
        head_llms_ranker = make_arrow_head_trace(lr_starts, lr_ends, "teal", "LLM to Ranker arrowheads")
        head_ranker_fuser = make_arrow_head_trace(rf_starts, rf_ends, "green", "Ranker to Fuser arrowhead")
        head_fuser_user = make_arrow_head_trace(fu_starts, fu_ends, "purple", "Fuser to User arrowhead")
    else:
        head_user_llms = empty_cone
        head_llms_ranker = empty_cone
        head_ranker_fuser = empty_cone
        head_fuser_user = empty_cone

    return [
        selected_llms, selected_ranker, selected_fuser,
        line_user_llms, line_llms_ranker, line_ranker_fuser, line_fuser_user,
        head_user_llms, head_llms_ranker, head_ranker_fuser, head_fuser_user
    ]


def route_annotation_text(route):
    if route is None:
        return "Optimal route: not enough LLM/ranker/fuser candidates inside the cone."

    return (
        f"<b>Optimal LLM-Blender route inside cone</b><br>"
        f"Selected: {L} LLMs + 1 Ranker + 1 Fuser<br>"
        f"Latency: {1000 * route['total_sec']:.3f} ms "
        f"(parallel LLM bottleneck {1000 * route['bottleneck_sec']:.3f} ms)<br>"
        f"Candidates: LLM {route['region_n_llm']}, "
        f"R {route['region_n_ranker']}, F {route['region_n_fuser']}<br>"
        f"Objective: min max[d(U,LLM)+d(LLM,R)] + d(R,F) + d(F,U)"
    )


def make_annotations(counts, route):
    return [
        dict(
            text=(
                f"Per orbit: {N_LLM_PER_ORBIT} LLMs, "
                f"{N_RANKER_PER_ORBIT} rankers, "
                f"{N_FUSER_PER_ORBIT} fusers. "
                f"Center plane starts at 0°; neighboring planes are physically offset by "
                f"±{STAGGERING_ANGLE_DEG:g}°."
            ),
            x=0.5,
            y=1.02,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=13)
        ),
        dict(
            text=f"Cone count: {counts[0]} (LLM {counts[1]}, R {counts[2]}, F {counts[3]})",
            x=0.985,
            y=0.98,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="top",
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=13)
        ),
        dict(
            text=route_annotation_text(route),
            x=0.985,
            y=0.84,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="top",
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="darkblue",
            borderwidth=1,
            font=dict(size=12)
        )
    ]


def title_text(t, counts):
    cc, cl, cr, cf = counts

    return (
        f"LLM / Ranker / Fuser simulation: L={L}, m={M_PER_ORBIT}, "
        f"{STAGGERING_ANGLE_DEG:g}° neighboring-plane offset<br>"
        f"Speed = {SATELLITE_SPEED:g} {SPEED_UNIT}, "
        f"1 frame = {DT:g} second, "
        f"t = {t:.0f} s, "
        f"planes = {N_PLANES}, "
        f"RAAN spacing = {raan_spacing_deg():.1f}°<br>"
        f"Inside cone = {cc} (LLM {cl}, R {cr}, F {cf})"
    )

# ============================================================
# BUILD STATIC OBJECTS
# ============================================================

user_pos = ground_user_position(USER_LAT_DEG, USER_LON_DEG)
axis = unit(user_pos)

xE, yE, zE = earth_surface()
earth_trace = go.Surface(
    x=xE,
    y=yE,
    z=zE,
    opacity=EARTH_OPACITY,
    colorscale=[[0.0, "#d9ecfa"], [1.0, "#8eb9dc"]],
    showscale=False,
    name="Earth",
    hoverinfo="skip",
    lighting=dict(ambient=0.95, diffuse=0.65, roughness=1.0, specular=0.02)
)

orbit_traces = []
for p in range(N_PLANES):
    pts = orbit_curve(raan_for_plane(p))
    orbit_traces.append(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="lines",
            line=dict(color="rgba(80,80,80,0.95)", width=ORBIT_WIDTH),
            opacity=ORBIT_OPACITY,
            showlegend=False,
            hoverinfo="skip",
            name=f"Orbit {p + 1}"
        )
    )

user_trace = go.Scatter3d(
    x=[user_pos[0]],
    y=[user_pos[1]],
    z=[user_pos[2]],
    mode="markers+text",
    marker=dict(size=USER_MARKER_SIZE, color="black", symbol="diamond"),
    text=["User"],
    textposition="top center",
    name="Ground user",
    hoverinfo="skip"
)

cone_X, cone_Y, cone_Z = center_cone_mesh(axis, CONE_HALF_ANGLE_DEG)
cone_trace = go.Surface(
    x=cone_X,
    y=cone_Y,
    z=cone_Z,
    opacity=0.16,
    showscale=False,
    colorscale=[[0.0, "#ffd94d"], [1.0, "#ffd94d"]],
    name=f"Earth-centered {CONE_HALF_ANGLE_DEG:g}° half-angle cone",
    hoverinfo="skip"
)

cap_pts = cap_boundary_circle(axis, CONE_HALF_ANGLE_DEG)
cap_trace = go.Scatter3d(
    x=cap_pts[:, 0],
    y=cap_pts[:, 1],
    z=cap_pts[:, 2],
    mode="lines",
    line=dict(color="gold", width=8),
    name="Cone boundary on orbital shell",
    hoverinfo="skip"
)

axis_line_trace = go.Scatter3d(
    x=[0.0, axis[0] * ORBIT_R * 1.02],
    y=[0.0, axis[1] * ORBIT_R * 1.02],
    z=[0.0, axis[2] * ORBIT_R * 1.02],
    mode="lines",
    line=dict(color="black", width=4, dash="dash"),
    name="Cone axis",
    hoverinfo="skip"
)

# ============================================================
# INITIAL FRAME
# ============================================================

pos0, roles0 = constellation_snapshot(0.0)
inside_cone0 = inside_centered_cone(user_pos, pos0, CONE_HALF_ANGLE_DEG)
m0 = role_masks(roles0, inside_cone0)
counts0 = counts_from_masks(inside_cone0, m0)
route0 = optimize_llm_blender_route(user_pos, pos0, roles0, inside_cone0)

role_specs = [
    ("LLMs outside cone", m0["llm_out"], dict(size=OUTSIDE_SIZE, color="royalblue", opacity=0.45)),
    ("LLMs inside cone", m0["llm_in"], dict(size=INSIDE_SIZE, color="royalblue", opacity=1.0, line=dict(color="white", width=1.0))),
    ("Rankers outside cone", m0["r_out"], dict(size=OUTSIDE_SIZE, color="seagreen", opacity=0.45, symbol="square")),
    ("Rankers inside cone", m0["r_in"], dict(size=INSIDE_SIZE, color="seagreen", opacity=1.0, symbol="square", line=dict(color="white", width=1.0))),
    ("Fusers outside cone", m0["f_out"], dict(size=OUTSIDE_SIZE, color="darkorchid", opacity=0.45, symbol="diamond")),
    ("Fusers inside cone", m0["f_in"], dict(size=INSIDE_SIZE, color="darkorchid", opacity=1.0, symbol="diamond", line=dict(color="white", width=1.0))),
]

role_data = [
    go.Scatter3d(
        x=pos0[mask, 0],
        y=pos0[mask, 1],
        z=pos0[mask, 2],
        mode="markers",
        marker=marker,
        name=name,
        hoverinfo="skip"
    )
    for name, mask, marker in role_specs
]

static_data = [
    earth_trace,
    *orbit_traces,
    user_trace,
    cone_trace,
    cap_trace,
    axis_line_trace
]

route_data = build_route_traces(user_pos, pos0, route0)

role_trace_offset = len(static_data)
fig = go.Figure(data=static_data + role_data + route_data)

# ============================================================
# ANIMATION FRAMES
# ============================================================

frames = []

for frame in range(N_FRAMES):
    t = frame * DT

    pos, roles = constellation_snapshot(t)
    inside_cone = inside_centered_cone(user_pos, pos, CONE_HALF_ANGLE_DEG)
    m = role_masks(roles, inside_cone)
    counts = counts_from_masks(inside_cone, m)
    route = optimize_llm_blender_route(user_pos, pos, roles, inside_cone)

    frame_traces = [
        go.Scatter3d(
            x=pos[m["llm_out"], 0],
            y=pos[m["llm_out"], 1],
            z=pos[m["llm_out"], 2],
            mode="markers",
            marker=dict(size=OUTSIDE_SIZE, color="royalblue", opacity=0.45)
        ),
        go.Scatter3d(
            x=pos[m["llm_in"], 0],
            y=pos[m["llm_in"], 1],
            z=pos[m["llm_in"], 2],
            mode="markers",
            marker=dict(size=INSIDE_SIZE, color="royalblue", opacity=1.0, line=dict(color="white", width=1.0))
        ),
        go.Scatter3d(
            x=pos[m["r_out"], 0],
            y=pos[m["r_out"], 1],
            z=pos[m["r_out"], 2],
            mode="markers",
            marker=dict(size=OUTSIDE_SIZE, color="seagreen", opacity=0.45, symbol="square")
        ),
        go.Scatter3d(
            x=pos[m["r_in"], 0],
            y=pos[m["r_in"], 1],
            z=pos[m["r_in"], 2],
            mode="markers",
            marker=dict(size=INSIDE_SIZE, color="seagreen", opacity=1.0, symbol="square", line=dict(color="white", width=1.0))
        ),
        go.Scatter3d(
            x=pos[m["f_out"], 0],
            y=pos[m["f_out"], 1],
            z=pos[m["f_out"], 2],
            mode="markers",
            marker=dict(size=OUTSIDE_SIZE, color="darkorchid", opacity=0.45, symbol="diamond")
        ),
        go.Scatter3d(
            x=pos[m["f_in"], 0],
            y=pos[m["f_in"], 1],
            z=pos[m["f_in"], 2],
            mode="markers",
            marker=dict(size=INSIDE_SIZE, color="darkorchid", opacity=1.0, symbol="diamond", line=dict(color="white", width=1.0))
        ),
    ]

    route_traces = build_route_traces(user_pos, pos, route)
    all_frame_traces = frame_traces + route_traces

    frames.append(
        go.Frame(
            data=all_frame_traces,
            traces=list(range(role_trace_offset, role_trace_offset + len(all_frame_traces))),
            name=str(frame),
            layout=go.Layout(
                title=title_text(t, counts),
                annotations=make_annotations(counts, route)
            )
        )
    )

fig.frames = frames

# ============================================================
# LAYOUT
# ============================================================

fig.update_layout(
    title=title_text(0.0, counts0),
    title_font=dict(size=18),
    width=1080,
    height=940,
    scene=dict(
        xaxis=dict(visible=False, showticklabels=False, title="", showgrid=False, zeroline=False, showbackground=False),
        yaxis=dict(visible=False, showticklabels=False, title="", showgrid=False, zeroline=False, showbackground=False),
        zaxis=dict(visible=False, showticklabels=False, title="", showgrid=False, zeroline=False, showbackground=False),
        aspectmode="cube",
        camera=dict(eye=dict(x=1.4, y=1.45, z=0.95))
    ),
    margin=dict(l=0, r=0, t=145, b=10),
    legend=dict(
        x=0.01,
        y=0.98,
        bgcolor="rgba(255,255,255,0.88)",
        font=dict(size=11)
    ),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            x=0.08,
            y=0.04,
            xanchor="left",
            yanchor="bottom",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=FRAME_DURATION_MS, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate"
                        )
                    ]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(frame=dict(duration=0, redraw=False), mode="immediate")
                    ]
                )
            ]
        )
    ],
    sliders=[
        dict(
            x=0.18,
            y=0.045,
            len=0.72,
            currentvalue=dict(prefix="Frame: "),
            steps=[
                dict(
                    method="animate",
                    args=[
                        [str(k)],
                        dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0)
                        )
                    ],
                    label=str(k)
                )
                for k in range(N_FRAMES)
            ]
        )
    ],
    annotations=make_annotations(counts0, route0)
)

fig.show()
