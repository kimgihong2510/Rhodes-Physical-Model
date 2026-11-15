"""
Based on:
Tobias Sønderbo, "Real-time physical model of the Rhodes Electric Piano,"
master's thesis, Aalborg University, 2024.

Note: 
The reference implementation approximates the clamped boundary by fixing
u[0] and u[1] to zero, which produced pitches above the target frequencies
in this model. 
To address this upward pitch shift, this implementation
fixes u[0] = 0 and enforces zero slope using the ghost-node relation
u[-1] = u[1], with a separate update equation for index 1.
"""

import math
from dataclasses import dataclass

import librosa
import numpy as np
from numba import njit

@dataclass 
class TINE: 
    r: float = 0.0005 
    rho: float = 7850
    E: float = 2e11
    sigma_0: float = 0.001 # 0.0001
    sigma_1: float = 0.0005 # 0.005
    I: float = math.pi * r ** 4 / 4
    A: float = math.pi * r**2
    kappa: float = math.sqrt(E*I / (rho*A)) # Eq. 3.4

@dataclass
class HAMMER:
    M: float = 0.011 # 0.011
    alpha: float = 2.8
    k: float = 1.5e11
    max_v_in: float = 4.0 # 4.0
    lam: float = 5e9 # 9e10
    contact_pos: float = 0.2
    contact_weights = [1.0]
    pos_at_rest: float = -3e-3

@dataclass
class DAMPER:
    K1: float = 1e2
    K3: float = 1e5
    R: float = 0.5
    contact_pos: float = 0.3

@dataclass
class PICKUP:
    gamma: float = 7e-6

@njit(cache=True)
def _render(
    sample_rate, h, N, pending_strike_velocity, 
    u, u_prev, u_next,
    hammer_contact_indices, hammer_weight, hammer_pos_at_rest,
    hammer_u, hammer_u_prev, hammer_u_next, 
    hammer_lam, hammer_k, hammer_m, hammer_alpha, 
    tine_kappa, tine_sigma_0, tine_sigma_1, tine_rho,
    tine_A, tine_length,
    is_key_released, damper_contact_index, damper_K1, damper_K3, damper_R,
    use_clamped_ghost_node
):
    """
    Function outside RhodesSingleKey class to enable Numba 
    JIT compilation for faster computation.
    """
    k = 1 / sample_rate

    # 1. Calculate the contact force; Eq. 3.20
    tine_contact_u = np.dot(hammer_weight, u[hammer_contact_indices])
    x = hammer_u - tine_contact_u

    F = 0.0
    if x > 0.0:
        tine_contact_u_prev = np.dot(hammer_weight, u_prev[hammer_contact_indices])
        prev_x = hammer_u_prev - tine_contact_u_prev

        mu_h = hammer_lam/hammer_k
        F = hammer_k * x**hammer_alpha * (1 + mu_h * (x-prev_x)/k)
        if F < 0.0:
            print("Warning: negative contact force detected.")
            F = 0.0

    # 2. Update the hammer motion.
    hammer_u_next = 2*hammer_u - hammer_u_prev - F*k**2 / hammer_m

    assert pending_strike_velocity >= 0.0
    if hammer_u <= hammer_pos_at_rest: 
        if pending_strike_velocity > 0.0 or hammer_u_next < hammer_u: 
            # Reset the hammer position if it is below rest and
            # a MIDI strike is pending or the hammer is moving downward.
            hammer_u_next = hammer_pos_at_rest + k * pending_strike_velocity
            hammer_u = hammer_pos_at_rest
            pending_strike_velocity = 0.0

    hammer_u_prev = hammer_u
    hammer_u = hammer_u_next

    # 3. Update the tine motion.
    # 3-1. Update interior grid points; Eq. 3.5
    mu = tine_kappa * k / h**2
    s = tine_sigma_1 * k / h**2

    u_next[2:-2] = (
        (2 - 6*mu**2 - 4*s) * u[2:-2]
        + (4*mu**2 + 2*s) * (u[3:-1] + u[1:-3])
        - mu**2 * (u[4:] + u[:-4])
        + (-1 + tine_sigma_0 * k + 4*s) * u_prev[2:-2]
        - 2*s * (u_prev[3:-1] + u_prev[1:-3])
    ) / (1 + tine_sigma_0 * k)

    # 3-2. clamped boundary condtion
    u_next[0] = 0.0

    if use_clamped_ghost_node:
        u_next[1] = (
            (2 - 7*mu**2 - 4*s) * u[1]
            + (4*mu**2 + 2*s) * u[2]
            - mu**2 * u[3]
            + (-1 + tine_sigma_0*k + 4*s) * u_prev[1]
            - 2*s * u_prev[2]
        ) / (1 + tine_sigma_0*k)
    else:
        u_next[1] = 0.0
    

    # 3-3. free-end boundary condition
    # Eq 3.9
    u_next[N-1] = (
        (2 - 5*mu**2 - 4*s) * u[N-1]
        + (2*s + 2*mu**2) * u[N]
        + (4*mu**2 + 2*s) * u[N-2]
        - mu**2 * u[N-3]
        + (-1 + tine_sigma_0*k + 4*s) * u_prev[N-1]
        - (2*s) * (u_prev[N-2] + u_prev[N]) 
    ) / (1 + tine_sigma_0 * k) 

    # Eq 3.10
    u_next[N] = (
        (2 - 2*mu**2 - 4*s) * u[N]
        + (4*mu**2 + 4*s) * u[N-1]
        - (2*mu**2) * u[N-2]
        + (-1 + tine_sigma_0*k + 4*s) * u_prev[N]
        - (4*s) * u_prev[N-1]
    ) / (1 + tine_sigma_0 * k)

    # 3-4. Apply the hammer contact force to the tine. 
    u_next[hammer_contact_indices] += hammer_weight * F * k**2 / (tine_rho*tine_A*h*(1+tine_sigma_0*k))

    # 4. Apply the damper force.
    if is_key_released:
        eta = u[damper_contact_index]
        eta_prev = u_prev[damper_contact_index]
        u_star = u_next[damper_contact_index]
        r_plus = damper_K1/4 + damper_K3*eta**2/2 + damper_R/(2*k)
        r_minus = damper_K1/4 + damper_K3*eta**2/2 - damper_R/(2*k)

        # Eq. 3.23
        numerator = (u_star + damper_K1/(2*r_plus)*eta + r_minus/r_plus*eta_prev)
        denominator = (1/r_plus + k**2/(h*tine_rho*tine_A*(1+tine_sigma_0*k)))
        f = 1/2 * (numerator) / (denominator)
        u_next[damper_contact_index] = u_star - k**2 / (h*tine_rho*tine_A*(1+tine_sigma_0*k)) * f

    u_prev, u, u_next= (
        u,
        u_next,
        u_prev
    )

    return (
        u[N], u_prev[N],
        u_prev, u, u_next,
        hammer_u_prev, hammer_u,
        pending_strike_velocity,
    )

class RhodesSingleKey():
    def __init__(self, 
                 sample_rate, 
                 midi_note, 
                 use_clamped_ghost_node=True):
        self.sample_rate = sample_rate

        # Estimate the tine length from the target fundamental frequency.
        f0 = librosa.midi_to_hz(midi_note) 
        self.tine_length = math.sqrt(
            1.426 * math.pi * TINE.r / 16 / f0
            * math.sqrt(TINE.E / TINE.rho)
        ) # Eq. 3.8

        # Choose the grid spacing to satisfy the linear tine scheme's stability condition.
        # Eq 3.3
        k = 1 / self.sample_rate 
        a = 4 * TINE.sigma_1 * k
        b = 16 * TINE.kappa**2 * k**2
        h_min = math.sqrt((a + math.sqrt(a**2 + b)) / 2) 

        self.N = math.floor(self.tine_length / h_min) # Eq. 3.6
        self.h = self.tine_length / self.N # Eq. 3.7
        assert self.h > h_min
        assert abs(self.tine_length - self.h * self.N) <= 1e-16

        contact_center = round(self.N * HAMMER.contact_pos)
        self.hammer_weight = np.array(HAMMER.contact_weights)
        num_contact_pts = len(self.hammer_weight)
        assert 0 <= contact_center-num_contact_pts//2 and contact_center+num_contact_pts//2 <= self.N
        self.hammer_contact_indices = contact_center + np.arange(num_contact_pts) - num_contact_pts // 2

        self.damper_contact_index = round(self.N * DAMPER.contact_pos)

        with np.load("flux_table.npz") as data:
            self.u_grid = data["displacement_grid"]
            self.flux_table = data["flux_table"] * PICKUP.gamma
            self.max_tine_tip_u = data["max_displacement"] 
            self.magnetic_flux_prev = float(
                np.interp(0.0, self.u_grid, self.flux_table)
            )

        # tine tip displacement
        self.u_prev = np.zeros(self.N + 1)
        self.u = np.zeros(self.N + 1)
        self.u_next = np.zeros(self.N + 1)

        # hammer displacement
        self.hammer_u_prev = HAMMER.pos_at_rest
        self.hammer_u = HAMMER.pos_at_rest
        self.hammer_u_next = HAMMER.pos_at_rest

        self.is_key_released = True
        self.pending_strike_velocity = 0.0

        # Skip computation until the first note-on event.
        self.has_started = False

        self.init_state()

        # test related parameters
        self.use_clamped_ghost_node = use_clamped_ghost_node 
    
    def note_on(self, velocity):
        self.is_key_released = False
        self.pending_strike_velocity = HAMMER.max_v_in * (velocity / 127)**3 # midi velocity range: 0~127

        if not self.has_started:
            self.init_state()
            self.has_started = True

    def note_off(self):
        self.is_key_released = True

    def init_state(self):
        self.magnetic_flux_prev = float(
            np.interp(0.0, self.u_grid, self.flux_table)
        )
    
        self.u_prev.fill(0)
        self.u.fill(0)
        self.u_next.fill(0)

        self.hammer_u_prev = HAMMER.pos_at_rest
        self.hammer_u = HAMMER.pos_at_rest
        self.hammer_u_next = HAMMER.pos_at_rest

    def render_sample(self):
        if not self.has_started:
            return 0.0, 0.0
        
        (
            tine_tip_u, tine_tip_prev,
            self.u_prev, self.u, self.u_next,
            self.hammer_u_prev, self.hammer_u,
            self.pending_strike_velocity,
        ) = _render(
            self.sample_rate, self.h, self.N, self.pending_strike_velocity,
            self.u, self.u_prev, self.u_next,
            self.hammer_contact_indices, self.hammer_weight, HAMMER.pos_at_rest,
            self.hammer_u, self.hammer_u_prev, self.hammer_u_next,
            HAMMER.lam, HAMMER.k, HAMMER.M, HAMMER.alpha,
            TINE.kappa, TINE.sigma_0, TINE.sigma_1, TINE.rho, 
            TINE.A, self.tine_length,
            self.is_key_released, self.damper_contact_index, DAMPER.K1, DAMPER.K3, DAMPER.R,
            self.use_clamped_ghost_node
        )

        if abs(tine_tip_u) > self.max_tine_tip_u:
            print(f"Warning: tine-tip displacement exceeds the lookup-table range.")

        # The lookup table takes displacement from tine tip's rest position,not the absolute tip position.
        flux = np.interp(tine_tip_u, self.u_grid, self.flux_table)
        # Compute the induced EMF.
        output = -(flux - self.magnetic_flux_prev) * self.sample_rate
        self.magnetic_flux_prev = flux
    
        return tine_tip_u, output



        
