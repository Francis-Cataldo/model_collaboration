import numpy as np
import random
# ============================================================
# CONFIG
# ============================================================

# Change process compute times below.
# Units: seconds
LLM_COMPUTE_SEC = 
RANKER_COMPUTE_SEC = 2.0
FUSER_COMPUTE_SEC = 2.0

# Earth / orbit settings
R_EARTH = 6371.0
ALTITUDE = 530.0
ORBIT_R = R_EARTH + ALTITUDE

SATELLITE_SPEED = 8.0
SPEED_UNIT = "km/s"  # use "km/s" or "km/h"
SPEED_KM_PER_SEC = SATELLITE_SPEED / 3600.0 if SPEED_UNIT == "km/h" else SATELLITE_SPEED
OMEGA = SPEED_KM_PER_SEC / ORBIT_R

INCLINATION_DEG = 53.0
N_PLANES = 72
M_PER_ORBIT = 20        # Number of satellites per orbit
L = 3                   # Number of selected LLM satellites

FRACTIONAL_SLOT_OFFSET_STEP = 1.5

#random user positions. Also added inside functions
USER_LAT_DEG = random.uniform(-70.0, 70.0)
USER_LON_DEG = random.uniform(-180.0, 180.0)
CONE_HALF_ANGLE_DEG = 22.5

# Choose where route candidates are selected from:
# "cone" = all satellites inside the cone
# "diamond" = only satellites inside the inscribed diamond
ROUTE_REGION = "cone"

# Snapshot time
T_SNAPSHOT_SEC = 0.0

# Speed of light
C_LIGHT_KM_PER_SEC = 299792.458


# ============================================================
# BASIC CHECKS
# ============================================================

if M_PER_ORBIT % (L + 2) != 0:
    raise ValueError(
        f"M_PER_ORBIT={M_PER_ORBIT} must be divisible by L+2={L+2}. "
        f"Change M_PER_ORBIT or L."
    )

K_BLOCKS = M_PER_ORBIT // (L + 2)
N_LLM_PER_ORBIT = L * K_BLOCKS
N_RANKER_PER_ORBIT = K_BLOCKS
N_FUSER_PER_ORBIT = K_BLOCKS

CENTER_PLANE_INDEX = N_PLANES // 2
SLOT_ANGLE_RAD = 2 * np.pi / M_PER_ORBIT

PLANE_PHASE_OFFSETS = np.array([
    (p - CENTER_PLANE_INDEX) * FRACTIONAL_SLOT_OFFSET_STEP * SLOT_ANGLE_RAD
    for p in range(N_PLANES)
])


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def unit(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def raan_for_plane(p):
    return 2 * np.pi * p / N_PLANES


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


def pairwise_distance(a, b):
    return np.linalg.norm(a - b, axis=-1)


# ============================================================
# CONSTELLATION AND ROLE ASSIGNMENT
# ============================================================

def roles_for_plane(plane_index):
    """
    Dynamic role assignment for arbitrary L.

    Each block has length L+2:
        1 ranker
        L LLMs
        1 fuser

    The fuser is placed roughly in the middle of each block.
    """

    block_len = L + 2

    if M_PER_ORBIT % block_len != 0:
        raise ValueError(
            f"M_PER_ORBIT={M_PER_ORBIT} must be divisible by L+2={block_len}. "
            f"Change M_PER_ORBIT or L."
        )

    roles = np.array(["LLM"] * M_PER_ORBIT, dtype=object)
    fuser_offset = int(np.ceil(block_len / 2))

    for start in range(0, M_PER_ORBIT, block_len):
        roles[start] = "R"
        roles[start + fuser_offset] = "F"

    return roles


def constellation_snapshot(t):
    inc = np.deg2rad(INCLINATION_DEG)

    positions = []
    roles = []

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


# ============================================================
# REGION SELECTION
# ============================================================

def inside_centered_cone(user_pos, sat_positions, cone_half_angle_deg):
    axis = unit(user_pos)
    c = np.cos(np.deg2rad(cone_half_angle_deg))

    sat_hats = sat_positions / np.linalg.norm(
        sat_positions,
        axis=1,
        keepdims=True
    )

    return (sat_hats @ axis) >= c


# def inside_inscribed_diamond(user_pos, sat_positions, cone_half_angle_deg):
#     axis = unit(user_pos)

#     inside_cone = inside_centered_cone(
#         user_pos,
#         sat_positions,
#         cone_half_angle_deg
#     )

#     center, rho, v1, v2 = cap_circle_geometry(
#         axis,
#         cone_half_angle_deg,
#         ORBIT_R
#     )

#     d = sat_positions - center[None, :]
#     x = d @ v1
#     y = d @ v2

#     return inside_cone & ((np.abs(x) + np.abs(y)) <= rho + 1e-9)


def route_candidate_mask(inside_cone):
    return inside_cone


# ============================================================
# LATENCY OPTIMIZATION
# ============================================================

def optimize_llm_blender_route(user_pos, sat_positions, roles, inside_cone, g=1.0):
    """
    Objective:

        min over ranker R, fuser F, and selected L LLMs:

            max_i [
                d(user, LLM_i) / c
                + LLM_compute_time / g
                + d(LLM_i, ranker) / c
            ]
            + ranker_compute_time / g
            + d(ranker, fuser) / c
            + fuser_compute_time / g
            + d(fuser, user) / c
    """

    if g <= 0:
        raise ValueError("g must be positive.")

    llm_compute_sec = LLM_COMPUTE_SEC / g
    ranker_compute_sec = RANKER_COMPUTE_SEC / g
    fuser_compute_sec = FUSER_COMPUTE_SEC / g

    region = route_candidate_mask(inside_cone)

    llm_idx = np.where((roles == "LLM") & region)[0]
    ranker_idx = np.where((roles == "R") & region)[0]
    fuser_idx = np.where((roles == "F") & region)[0]

    if len(llm_idx) < L or len(ranker_idx) == 0 or len(fuser_idx) == 0:
        return None

    best = None

    user_to_llm_all = pairwise_distance(
        sat_positions[llm_idx],
        user_pos
    ) / C_LIGHT_KM_PER_SEC

    fuser_to_user_all = pairwise_distance(
        sat_positions[fuser_idx],
        user_pos
    ) / C_LIGHT_KM_PER_SEC

    for r_idx in ranker_idx:
        ranker_pos = sat_positions[r_idx]

        llm_to_ranker_all = pairwise_distance(
            sat_positions[llm_idx],
            ranker_pos
        ) / C_LIGHT_KM_PER_SEC

        llm_stage_times = (
            user_to_llm_all
            + llm_compute_sec
            + llm_to_ranker_all
        )

        order = np.argsort(llm_stage_times)
        selected_local = order[:L]
        selected_llm_idx = llm_idx[selected_local]

        bottleneck_sec = float(np.max(llm_stage_times[selected_local]))

        ranker_to_fuser_all = pairwise_distance(
            sat_positions[fuser_idx],
            ranker_pos
        ) / C_LIGHT_KM_PER_SEC

        total_for_fusers = (
            bottleneck_sec
            + ranker_compute_sec
            + ranker_to_fuser_all
            + fuser_compute_sec
            + fuser_to_user_all
        )

        best_f_local = int(np.argmin(total_for_fusers))
        f_idx = int(fuser_idx[best_f_local])
        total_sec = float(total_for_fusers[best_f_local])

        if best is None or total_sec < best["total_sec"]:
            best = {
                "total_sec": total_sec,
                "llm_parallel_stage_sec": bottleneck_sec,
                "ranker_compute_sec": ranker_compute_sec,
                "ranker_to_fuser_sec": float(ranker_to_fuser_all[best_f_local]),
                "fuser_compute_sec": fuser_compute_sec,
                "fuser_to_user_sec": float(fuser_to_user_all[best_f_local]),
                "ranker_idx": int(r_idx),
                "fuser_idx": int(f_idx),
                "llm_indices": selected_llm_idx.astype(int),
            }

    return best

def get_total_latency_ms(g=100, t_seconds=0.0, pool_llm_max_time=None):
    if pool_llm_max_time:
        LLM_COMPUTE_SEC = pool_llm_max_time
    """
    Return the total optimized latency in milliseconds.

    Parameters
    ----------
    t_seconds : float
        Simulation time in seconds. Default is 0.0, the initial position.

    g : float
        Compute speedup factor. Default is 1.0.
        All LLM, ranker, and fuser compute times are divided by g.

    Returns
    -------
    float
        Total latency in milliseconds.
    """
    # make random user position
    # import random
    latitude = random.uniform(-70.0, 70.0)
    longitude = random.uniform(-180.0, 180.0)
    user_pos = ground_user_position(latitude, longitude)

    pos, roles = constellation_snapshot(t_seconds)

    inside_cone = inside_centered_cone(user_pos, pos, CONE_HALF_ANGLE_DEG)

    route = optimize_llm_blender_route(
        user_pos,
        pos,
        roles,
        inside_cone,
        g=g
    )

    if route is None:
        raise ValueError("No valid route found.")

    return 1000 * route["total_sec"]


def random_llm_blender_route(user_pos, sat_positions, roles, inside_cone, g=1.0):
    """
    Randomly select L LLMs, 1 ranker, and 1 fuser inside ROUTE_REGION.

    This uses the same latency formula as the optimized route, but it does not
    search for the best satellites. It samples candidates uniformly at random
    from the satellites inside the selected region.

    System latency:
        max_i [d(user, LLM_i)/c + LLM_compute/g + d(LLM_i, ranker)/c]
        + ranker_compute/g
        + d(ranker, fuser)/c
        + fuser_compute/g
        + d(fuser, user)/c
    """

    if g <= 0:
        raise ValueError("g must be positive.")

    llm_compute_sec = LLM_COMPUTE_SEC / g
    ranker_compute_sec = RANKER_COMPUTE_SEC / g
    fuser_compute_sec = FUSER_COMPUTE_SEC / g

    region = route_candidate_mask(inside_cone)

    llm_idx = np.where((roles == "LLM") & region)[0]
    ranker_idx = np.where((roles == "R") & region)[0]
    fuser_idx = np.where((roles == "F") & region)[0]

    if len(llm_idx) < L or len(ranker_idx) == 0 or len(fuser_idx) == 0:
        return None

    # Random selection inside the region.
    selected_llm_idx = np.array(random.sample(list(llm_idx), L), dtype=int)
    r_idx = int(random.choice(list(ranker_idx)))
    f_idx = int(random.choice(list(fuser_idx)))

    ranker_pos = sat_positions[r_idx]
    fuser_pos = sat_positions[f_idx]
    selected_llm_pos = sat_positions[selected_llm_idx]

    user_to_llm_times = pairwise_distance(selected_llm_pos, user_pos) / C_LIGHT_KM_PER_SEC
    llm_to_ranker_times = pairwise_distance(selected_llm_pos, ranker_pos) / C_LIGHT_KM_PER_SEC

    llm_stage_times = user_to_llm_times + llm_compute_sec + llm_to_ranker_times
    llm_parallel_stage_sec = float(np.max(llm_stage_times))

    ranker_to_fuser_sec = float(pairwise_distance(ranker_pos, fuser_pos) / C_LIGHT_KM_PER_SEC)
    fuser_to_user_sec = float(pairwise_distance(fuser_pos, user_pos) / C_LIGHT_KM_PER_SEC)

    total_sec = (
        llm_parallel_stage_sec
        + ranker_compute_sec
        + ranker_to_fuser_sec
        + fuser_compute_sec
        + fuser_to_user_sec
    )

    return {
        "total_sec": float(total_sec),
        "llm_parallel_stage_sec": llm_parallel_stage_sec,
        "ranker_compute_sec": ranker_compute_sec,
        "ranker_to_fuser_sec": ranker_to_fuser_sec,
        "fuser_compute_sec": fuser_compute_sec,
        "fuser_to_user_sec": fuser_to_user_sec,
        "ranker_idx": int(r_idx),
        "fuser_idx": int(f_idx),
        "llm_indices": selected_llm_idx.astype(int),
    }

#Call this to get latency for random selection
def get_random_total_latency_ms(g=1.0, t_seconds=0.0):
    """
    Return the random-selection latency in milliseconds.

    Parameters
    ----------
    g : float
        Compute speedup factor. All LLM, ranker, and fuser compute times are
        divided by g.

    t_seconds : float
        Simulation time in seconds.

    Returns
    -------
    float
        Total latency in milliseconds for one random route.
    """

    if g <= 0:
        raise ValueError("g must be positive.")

    #random user position
    latitude = random.uniform(-70.0, 70.0)
    longitude = random.uniform(-180.0, 180.0)
    user_pos = ground_user_position(latitude, longitude)
    pos, roles = constellation_snapshot(t_seconds)

    inside_cone = inside_centered_cone(user_pos, pos, CONE_HALF_ANGLE_DEG)


    route = random_llm_blender_route(
        user_pos,
        pos,
        roles,
        inside_cone,
        g=g
    )

    if route is None:
        raise ValueError("No valid random route found.")

    return 1000 * route["total_sec"]
# ============================================================
# MAIN
# ============================================================

def main(g=1.0):
    if g <= 0:
        raise ValueError("g must be positive.")

    user_pos = ground_user_position(USER_LAT_DEG, USER_LON_DEG)

    sat_positions, roles = constellation_snapshot(T_SNAPSHOT_SEC)

    inside_cone = inside_centered_cone(
        user_pos,
        sat_positions,
        CONE_HALF_ANGLE_DEG
    )


    route = optimize_llm_blender_route(
        user_pos,
        sat_positions,
        roles,
        inside_cone,
        g=g
    )

    if route is None:
        print("No valid route found.")
        return
    
    print(f"g = {g}")
    # print(f"User pos = {user_pos}")
    print("Optimum Selection Latency Breakdown:")
    print(f"Total Latency = {1000 * route['total_sec']:.3f} ms")
    print(
        "Minimum Time from user to the pool LLMs to the ranker = "
        f"{1000 * route['llm_parallel_stage_sec']:.3f} ms"
    )
    print(f"Ranker compute / g = {1000 * route['ranker_compute_sec']:.3f} ms")
    print(f"Ranker to Fuser = {1000 * route['ranker_to_fuser_sec']:.3f} ms")
    print(f"Fuser compute / g = {1000 * route['fuser_compute_sec']:.3f} ms")
    print(f"Fuser to User = {1000 * route['fuser_to_user_sec']:.3f} ms")

    #Debug code
    random_route = random_llm_blender_route(
        user_pos,
        sat_positions,
        roles,
        inside_cone,
        g=g
    )

    if random_route is None:
        print("No valid random route found.")
        return

    print("\nRandom Selection Latency Breakdown:")
    print(f"Random Total Latency = {1000 * random_route['total_sec']:.3f} ms")
    print(
        "Random Time from user to the pool LLMs to the ranker = "
        f"{1000 * random_route['llm_parallel_stage_sec']:.3f} ms"
    )
    print(f"Random Ranker compute / g = {1000 * random_route['ranker_compute_sec']:.3f} ms")
    print(f"Random Ranker to Fuser = {1000 * random_route['ranker_to_fuser_sec']:.3f} ms")
    print(f"Random Fuser compute / g = {1000 * random_route['fuser_compute_sec']:.3f} ms")
    print(f"Random Fuser to User = {1000 * random_route['fuser_to_user_sec']:.3f} ms")
if __name__ == "__main__":
    main(g=100)

print(get_total_latency_ms(100))
print(get_random_total_latency_ms(100))

