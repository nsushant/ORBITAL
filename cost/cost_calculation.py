## Cost Oracle, functions that return costs of transfers 

from numba import njit, prange
import numpy as np 

# we identify state by (a,i,raan)
# oe_depot = np.empty((num_depots,3),dtype=np.float64)
# oe_clients = np.empty((num_clients,3), dtype=np.float64)
# units adopted = km, s, kg. so that 1 N = 10^-3 kg km /s^2    
# state = [a,i,raan,m,Isp,T (thrust force in kg km /s^2 )]

nodes, weights = np.polynomial.legendre.leggauss(16)


@njit 
def to_vspace(state,mu = 398600.44, g = 0.00981): 

    v = np.sqrt(mu/state[0]) 
    y = v * np.sin(np.pi/2 * state[1])
    x = v * np.cos(np.pi/2 * state[1])

    return np.array([x,y]) 

@njit
def from_vspace(point, mu = 398600.44, g = 0.00981): 

    v_squared = point[0]**2 + point[1]**2

    a = mu/v_squared

    theta_s = np.arctan2(point[1],point[0])

    i = 2/np.pi * theta_s

    return a,i
    


@njit
def dv_lt(state1,state2, g= 0.00981): 

    """
    calculates the edelbaum cost of an orbital 
    transfer from initial (a,i) to some target (a,i). 
    """

    vstate1 = to_vspace(state1)
    vstate2 = to_vspace(state2)

    dx = vstate2[0]- vstate1[0]
    dy = vstate2[1]- vstate1[1]

    dv = np.sqrt(dx * dx + dy * dy)
    ve = state1[4]*g

    dt = ve * (state1[3] / state1[5])*(1 - np.exp(-dv/ve))

    mnew = state1[3]*np.exp(-1*(dv/ve))

    v1 = np.linalg.norm(vstate1)
    v2 = np.linalg.norm(vstate2)

    di = np.abs(state2[1] - state1[1])

    num = np.sin(np.pi*di/2.0)
    den = v1/v2 - np.cos(np.pi*di/2.0)
    beta = np.arctan2(num, den)  

    return dv,beta,vstate1,vstate2,mnew,v1,v2,dt  



@njit
def edelbaum_trace_state(s, dv,state1, vstate1, vstate2, g = 0.00981,mu = 398600.44):

    dx = vstate2[0]- vstate1[0]
    dy = vstate2[1]- vstate1[1]

    if not dv > 0 : 
        print("total dv <= 0, invalid input arg provided")
        return np.nan,np.nan,np.nan  

    
    vs_squared = (vstate1[0] + (s/dv * dx))**2 + (vstate1[1] + (s/dv * dy))**2
    a_s = mu/(vs_squared) 

    xs = vstate1[0] + (s/dv * dx) 
    ys = vstate1[1] + (s/dv * dy)

    theta_s = np.arctan2(ys,xs)
    i_s = 2/np.pi * theta_s

    ve = state1[4]*g

    m_s = state1[3] * np.exp( -s / ve )
    f_s = state1[5]/ m_s

    return a_s,i_s,f_s 


@njit
def j2_raan_rate(a,i,e=0.0, j2 = 1.0826e-3, mu = 398600.44, Re = 6371): 

    n = np.sqrt(mu/a**3)
    p = a*(1-e**2)

    return -1.5 * j2 * (Re / p)**2 * n * np.cos(i)


@njit 
def calculate_raan_accumulated( state1, vstate1, vstate2, dv, 
                                nodes = nodes, weights = weights): 

    """
    Gauss legendre roots are between [-1,1] and the domain of s is [0,dv]
    so we have to shift the weights and nodes so that they occupy the
    domain of 's'. 
    """

    
    s_nodes =  dv * 0.5  * (nodes + 1.0) 
    scaled_weights = dv * 0.5 * weights

    accumulated_raan = 0.0 

    for k in range(len(s_nodes)): 

        s = s_nodes[k]
        w = scaled_weights[k]

        a_s,i_s,f_s  = edelbaum_trace_state(s, dv, state1, vstate1, vstate2)

        raan_rate = j2_raan_rate(a_s,i_s)

        raan_change = raan_rate/f_s 

        accumulated_raan += raan_change * w

    return accumulated_raan 


@njit
def grow_ellipse(growth, vstate1, vstate2, num_points):
    center = (vstate1 + vstate2) / 2
    d = vstate2 - vstate1
    c = np.linalg.norm(d) / 2

    unit_vec = d / np.linalg.norm(d)
    v = np.array([-unit_vec[1], unit_vec[0]])

    a_e = c + growth

    if growth == 0:
        c = np.linspace(0, 1, num_points)
        return (1 - c)[:, np.newaxis] * vstate1 + c[:, np.newaxis] * vstate2

    b = np.sqrt(a_e**2 - c**2)

    t = np.linspace(0, 2*np.pi, num_points)

    return  center + a_e * np.cos(t)[:, np.newaxis] * unit_vec + b * np.sin(t)[:, np.newaxis] * v 


@njit
def eval_point_in_vspace(point,draan,state1,state2,vstate1,vstate2,tof):

    # for all these points we need to see if any of them provides a feasible drift time
    a_drift,i_drift = from_vspace(np.array([point[0],point[1]]))

    # first find if coasting for full tof can exceed or close the raan gap
    max_draan = j2_raan_rate(a_drift,i_drift) * tof

    if np.sign(max_draan) != np.sign(draan):
        return np.nan,np.nan,np.nan , np.nan,np.nan

    if np.abs(max_draan) < np.abs(draan) :
        return np.nan,np.nan,np.nan, np.nan,np.nan
            
       
    # if it got here then raan gap can be closed, so next compute the actual cost 
    drift_state = np.copy(state1)
    drift_state[0] = a_drift
    drift_state[1] = i_drift 

    # first burn
    dv_1,beta_1,vstate1,vstate_drift,m_drift,v1,vdrift,dt1  = dv_lt(state1,drift_state) 
    drift_state[3] = m_drift 
    accumulated_raan = calculate_raan_accumulated(state1, vstate1, vstate_drift, dv_1)
    drift_state[2] += accumulated_raan

    if (tof - dt1) <= 0: 
        return np.nan,np.nan,np.nan ,np.nan,np.nan

    # second burn
    #dv,beta,vstate1,vstate2,mnew,v1,v2,dt 
    dv_2,beta_2,vstate_drift,vstate2_copy,m_final,vdrift,vfinal,dt2  = dv_lt(drift_state,state2) 
    accumulated_raan2 = calculate_raan_accumulated(drift_state, vstate_drift, vstate2, dv_2)
    t_drift = tof - dt1 - dt2 

    if t_drift < 0: 
        return np.nan, np.nan, np.nan ,np.nan,np.nan

    drift_rate = j2_raan_rate(drift_state[0],drift_state[1])
    raan_final = accumulated_raan + drift_rate*t_drift + accumulated_raan2

    if np.abs(raan_final) < np.abs(draan): 
        return np.nan,np.nan,np.nan,np.nan,np.nan

    return dv_1+dv_2, drift_state, vdrift, t_drift, raan_final



@njit
def find_by_bisection(draan,state1,state2,vstate1,vstate2,tof,num_points): 

    feasible_points = np.array([]) 
    dvs = np.array([])

    a_init = 1
    a_infeasible = 0 
    # grow ellipse until feasible points are found
    while (len(feasible_points) == 0):
        a_init *= 2 
        points = grow_ellipse(a_init,vstate1,vstate2,num_points)

        for point in points: 

            dv_tot, drift_state_can, vdrift_can, t_drift_can, raan_final = eval_point_in_vspace(point,draan,state1,
                                                                                                state2,vstate1,vstate2,
                                                                                                tof)

            if np.isnan(dv_tot):
                continue
            
            if np.abs(raan_final) >= np.abs(draan): 
                dvs = np.append(dvs,dv_tot)
                feasible_points = np.append(feasible_points,vdrift_can) 

        if (len(feasible_points) == 0): 
            a_infeasible = a_init

    # once we get here a feasible point has been found. 
    # we now need to find the feasible point with the lowest dcost 
    # for this we can bisect on the ellipse growth parameter

    best_dv = None 
    best_state= None 

    tol = 1e-2 

    a_feasible = a_init

    while a_feasible - a_infeasible > tol: 

        a_mid = 0.5 * (a_infeasible + a_feasible)

        points = grow_ellipse(a_mid,vstate1,vstate2,num_points)

        dvs_inner = np.array([]) 
        feasible_inner = np.array([])
        for point in points: 

            dv_tot, drift_state_can, vdrift_can, t_drift_can, raan_final = eval_point_in_vspace(point,draan,state1,state2,vstate1,vstate2,tof)

            if np.isnan(dv_tot):
                continue
            
            if np.abs(raan_final) >= np.abs(draan): 
                dvs_inner = np.append(dvs_inner,dv_tot)
                feasible_inner = np.append(feasible_inner,vdrift_can) 

        if (len(feasible_inner) > 0): 
            a_feasible = a_mid
            best_idx = np.argmin(dvs_inner)  
            best_dv = dvs_inner[best_idx]
            best_state = feasible_inner[best_idx]

        else: 
            # we hit an infeasible ellipse 
            a_infeasible = a_mid 

    
    return best_dv





@njit 
def ellipse_search(state1, state2, tof, mu = 398600.44, g=0.00981,num_points=30): 

    """
    To find the minimum transfer dv, we project the two (target and initial)
    orbits in the velocity plane. The shortest line joining them contains the lowest dv
    candiates for drift orbits. 

    It is possible however that given the tof, no point on the line (candidate drift orbit)
    closes the raan gap in time. This means we will have to expand the search. We do this 
    by growing an ellipse around the shortest path and looking for points that are feasible drift orbits. 
    Finally a bisection is performed to find the best dv drift orbit. 
    
    """

    # firstly, we know that around polar orbit the formulation fails because there is no 
    # effect of J2 at all, precession rate is 0. So lets set up a guard for this. 

    di = np.abs(state2[1] - state1[1])

    # break point of the edelbaum formulation
    assert di <= 2.0, "Edelbaum is no longer applicable"

    # raan gap 
    draan = state2[2] - state1[2]

    # convert to vstate 
    vstate1 = to_vspace(state1)
    vstate2 = to_vspace(state2)

    # fetch points that sit on the chord joining the two points 
    dv,beta,vstate1,vstate2,mnew,v1,v2,dt = dv_lt(state1,state2) 

    points = grow_ellipse(0, vstate1, vstate2, num_points)

    dvs_found = np.array([]) 
    candidates = np.array([]) 


    for point in points: 
        # for all these points we need to see if any of them provides a feasible drift time
        dv_tot, drift_state_can, vdrift_can, t_drift_can, raan_final = eval_point_in_vspace(point,draan,state1,state2,vstate1,vstate2,tof)

        if np.isnan(dv_tot): 
            continue 

        dvs_found = np.append(dvs_found,dv_tot)
        candidates = np.append(candidates,point)

    if (dvs_found.shape[0] == 0):

        # here we have found no feasible drift orbits on the chord 
        # grow ellipse and bisect to find the minimum cost drift orbit 

        best_dv = find_by_bisection(draan,state1,state2,vstate1,vstate2,tof,num_points)

    else:
        best_idx = np.argmin(dvs_found)
        best_dv = dvs_found[best_idx]

    return best_dv


@njit
def compute_bundle(state1, state2, point):
    a_drift, i_drift = from_vspace(np.array([point[0], point[1]]))
    drift_state = np.array([a_drift, i_drift, state1[2], state1[3], state1[4], state1[5]])

    dv_1, _, vstate1_vs, vstate_drift_vs, m_drift, _, _, dt1 = dv_lt(state1, drift_state)
    drift_state[3] = m_drift
    acc_raan1 = calculate_raan_accumulated(state1, vstate1_vs, vstate_drift_vs, dv_1)
    drift_state[2] += acc_raan1

    dv_2, _, _, vstate2_vs, _, _, _, dt2 = dv_lt(drift_state, state2)
    acc_raan2 = calculate_raan_accumulated(drift_state, vstate_drift_vs, vstate2_vs, dv_2)

    drift_rate = j2_raan_rate(a_drift, i_drift)
    dv_tot = dv_1 + dv_2

    return np.array([dv_tot, dt1, dt2, acc_raan1, acc_raan2, drift_rate])


@njit
def generate_candidates_for_growth_range(vstate1, vstate2, num_points, growths):
    n_growths = len(growths)
    candidates = np.empty((n_growths * num_points, 2))
    offset = 0
    for g in range(n_growths):
        ring = grow_ellipse(growths[g], vstate1, vstate2, num_points)
        candidates[offset:offset + num_points] = ring
        offset += num_points
    return candidates


@njit
def ellipse_search_multiple_tofs(state1, state2, tofs, mu=398600.44, g=0.00981,
                                  num_points=30, n_rings=20):
    n_tofs = len(tofs)
    dv_vals = np.full(n_tofs, np.nan)
    draan = state2[2] - state1[2]

    di = abs(state2[1] - state1[1])
    if di > 2.0:
        return dv_vals

    vstate1 = to_vspace(state1)
    vstate2 = to_vspace(state2)

    growth_max = 8.0
    MAX_GROWTH = 4096.0

    chord = grow_ellipse(0, vstate1, vstate2, num_points)
    growths = 10 ** np.linspace(-1, np.log10(growth_max), n_rings)
    rings = generate_candidates_for_growth_range(vstate1, vstate2, num_points, growths)
    n_chord = chord.shape[0]
    n_rings_pts = rings.shape[0]
    all_cands = np.empty((n_chord + n_rings_pts, 2))
    all_cands[:n_chord] = chord
    all_cands[n_chord:] = rings

    n_cand = all_cands.shape[0]
    all_bundles = np.empty((n_cand, 6))
    for c in range(n_cand):
        all_bundles[c] = compute_bundle(state1, state2, all_cands[c])

    missing = np.ones(n_tofs, dtype=np.bool_)

    while growth_max <= MAX_GROWTH:
        n_cand = all_bundles.shape[0]
        still_missing = False
        for k in range(n_tofs):
            if not missing[k]:
                continue
            best = np.inf
            tof = tofs[k]
            for c in range(n_cand):
                r = all_bundles[c, 5]
                dv_tot = all_bundles[c, 0]
                dt1 = all_bundles[c, 1]
                dt2 = all_bundles[c, 2]

                if np.sign(r * tof) != np.sign(draan):
                    continue
                if abs(r * tof) < abs(draan):
                    continue
                if tof <= dt1:
                    continue
                t_avail = tof - dt1 - dt2
                if t_avail < 0:
                    continue
                acc1 = all_bundles[c, 3]
                acc2 = all_bundles[c, 4]
                raan_final = acc1 + r * t_avail + acc2
                if abs(raan_final) < abs(draan):
                    continue
                if dv_tot < best:
                    best = dv_tot

            if best < np.inf:
                dv_vals[k] = best
                missing[k] = False
            else:
                still_missing = True

        if not still_missing:
            break

        growth_max *= 2.0
        new_growths = 10 ** np.linspace(np.log10(growth_max / 2.0) + 0.01,
                                         np.log10(growth_max), n_rings // 2)
        new_rings = generate_candidates_for_growth_range(vstate1, vstate2, num_points, new_growths)
        n_new = new_rings.shape[0]
        old_n = all_cands.shape[0]
        temp_cands = np.empty((old_n + n_new, 2))
        temp_bundles = np.empty((old_n + n_new, 6))
        temp_cands[:old_n] = all_cands
        temp_cands[old_n:] = new_rings
        temp_bundles[:old_n] = all_bundles
        for c in range(n_new):
            temp_bundles[old_n + c] = compute_bundle(state1, state2, new_rings[c])
        all_cands = temp_cands
        all_bundles = temp_bundles

    return dv_vals


@njit(parallel=True)
def build_cost_table(a_i_raan1, a_i_raan2, tofs, m, Isp, T, num_points=30, n_rings=20):
    n_pairs = a_i_raan1.shape[0]
    n_tofs = len(tofs)
    cost_table = np.empty((n_pairs, n_tofs))

    for p in prange(n_pairs):
        state1 = np.array([a_i_raan1[p, 0], a_i_raan1[p, 1], a_i_raan1[p, 2], m, Isp, T])
        state2 = np.array([a_i_raan2[p, 0], a_i_raan2[p, 1], a_i_raan2[p, 2], m, Isp, T])

        di = abs(state2[1] - state1[1])
        if di > 2.0:
            for k in range(n_tofs):
                cost_table[p, k] = np.nan
            continue

        draan = state2[2] - state1[2]
        vstate1 = to_vspace(state1)
        vstate2 = to_vspace(state2)

        growth_max = 8.0
        MAX_GROWTH = 4096.0

        chord = grow_ellipse(0, vstate1, vstate2, num_points)
        growths = 10 ** np.linspace(-1, np.log10(growth_max), n_rings)
        rings = generate_candidates_for_growth_range(vstate1, vstate2, num_points, growths)
        n_chord = chord.shape[0]
        n_rings_pts = rings.shape[0]
        all_cands = np.empty((n_chord + n_rings_pts, 2))
        all_cands[:n_chord] = chord
        all_cands[n_chord:] = rings

        n_cand = all_cands.shape[0]
        all_bundles = np.empty((n_cand, 6))
        for c in range(n_cand):
            all_bundles[c] = compute_bundle(state1, state2, all_cands[c])

        missing = np.ones(n_tofs, dtype=np.bool_)

        while growth_max <= MAX_GROWTH:
            n_cand = all_bundles.shape[0]
            still_missing = False
            for k in range(n_tofs):
                if not missing[k]:
                    continue
                best = np.inf
                tof = tofs[k]
                for c in range(n_cand):
                    r = all_bundles[c, 5]
                    dv_tot = all_bundles[c, 0]
                    dt1 = all_bundles[c, 1]
                    dt2 = all_bundles[c, 2]

                    if np.sign(r * tof) != np.sign(draan):
                        continue
                    if abs(r * tof) < abs(draan):
                        continue
                    if tof <= dt1:
                        continue
                    t_avail = tof - dt1 - dt2
                    if t_avail < 0:
                        continue
                    acc1 = all_bundles[c, 3]
                    acc2 = all_bundles[c, 4]
                    raan_final = acc1 + r * t_avail + acc2
                    if abs(raan_final) < abs(draan):
                        continue
                    if dv_tot < best:
                        best = dv_tot

                if best < np.inf:
                    cost_table[p, k] = best
                    missing[k] = False
                else:
                    still_missing = True

            if not still_missing:
                break

            growth_max *= 2.0
            new_growths = 10 ** np.linspace(np.log10(growth_max / 2.0) + 0.01,
                                             np.log10(growth_max), n_rings // 2)
            new_rings = generate_candidates_for_growth_range(vstate1, vstate2, num_points, new_growths)
            n_new = new_rings.shape[0]
            old_n = all_cands.shape[0]
            temp_cands = np.empty((old_n + n_new, 2))
            temp_bundles = np.empty((old_n + n_new, 6))
            temp_cands[:old_n] = all_cands
            temp_cands[old_n:] = new_rings
            temp_bundles[:old_n] = all_bundles
            for c in range(n_new):
                temp_bundles[old_n + c] = compute_bundle(state1, state2, new_rings[c])
            all_cands = temp_cands
            all_bundles = temp_bundles

        for k in range(n_tofs):
            if missing[k]:
                cost_table[p, k] = np.nan

    return cost_table






