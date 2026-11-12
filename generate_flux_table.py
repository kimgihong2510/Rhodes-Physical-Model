"""
Based on Section VI, "Simplified Method," of:
I. Nishiguchi et al., "Modeling of Magnetic Pickups for Electric
Musical Instruments," COMPUMAG, 2019.

Note:
1. The constants from the paper B0 and gamma are combined into
a single scaling parameter. This factor is omitted from the
lookup-table calculation and applied later in rhodes_synth.py.

2. Approximate the coil as a single circular loop in the z=0 plane,
rather than multiple turns wrapped around the length of the magnet.
"""

from dataclasses import dataclass

import numpy as np
import scipy as sp
from tqdm import tqdm

@dataclass
class PICKUP:
    radius_magnet: float = 4 * 1e-3
    radius_coil: float = 4.625 * 1e-3
    magnet_length: float = 21.5 * 1e-3

    # Tine position at rest.
    tine_tip_x: float = 3.0 * 1e-3
    tine_tip_y: float = 0.0
    tine_tip_z: float = 2.3 * 1e-3

    max_displacement: float = 35 * 1e-3
    flux_table_size: int = 16384

def calculate_plate_field(tip_pos, plate_z, radius):
    tip_pos = np.asarray(tip_pos, dtype=float)

    def integrand(phi, rho, component):
        source_pos = np.array([rho * np.cos(phi), rho * np.sin(phi), plate_z])
        displacement = tip_pos - source_pos
        distance = np.linalg.norm(displacement)

        return rho * displacement[component] / distance**3

    field = np.zeros(3)
    for component in range(3):
        field[component], _ = sp.integrate.dblquad(
            lambda phi, rho: integrand(phi, rho, component),
            0, radius,
            0, 2 * np.pi,
        )

    return field

def generate_flux_table():
    displacement_grid = np.linspace(
        -PICKUP.max_displacement, 
        PICKUP.max_displacement, 
        PICKUP.flux_table_size)

    # u is the x-displacement from the tine tip's rest position.
    tip_pos_at_rest = np.asarray([PICKUP.tine_tip_x, PICKUP.tine_tip_y, PICKUP.tine_tip_z], dtype=float)
    flux_table = []
    
    for u in tqdm(displacement_grid, desc="generating flux table..."):
        tip_pos = tip_pos_at_rest.copy()
        tip_pos[0] += u

        positive_field = calculate_plate_field(
            tip_pos,
            plate_z=0.0,
            radius=PICKUP.radius_magnet,
        )
        negative_field = calculate_plate_field(
            tip_pos,
            plate_z=-PICKUP.magnet_length,
            radius=PICKUP.radius_magnet,
        )
        B_tine_tip = positive_field - negative_field
        
        def integrand_B_coil(phi, rho):
            coil_pos = np.asarray([rho*np.cos(phi), rho*np.sin(phi), 0], dtype=float)
            r = coil_pos - tip_pos
            distance = np.linalg.norm(r)
            return rho * r[2] / distance**3 # Project onto the coil normal
        
        magnetic_flux = sp.integrate.dblquad(
            integrand_B_coil,
            0, PICKUP.radius_coil,
            0, 2 * np.pi,
            epsabs=1e-10,
            epsrel=1e-7
        )[0] * (-np.linalg.norm(B_tine_tip))

        flux_table.append(magnetic_flux)

    return displacement_grid, np.asarray(flux_table)

if __name__ == "__main__":
    displacement_grid, flux_table = generate_flux_table()

    np.savez(
        "flux_table.npz",
        displacement_grid=displacement_grid,
        flux_table=flux_table,
        max_displacement = PICKUP.max_displacement,
        tine_tip_x = PICKUP.tine_tip_x,
        tine_tip_y = PICKUP.tine_tip_y,
        tine_tip_z = PICKUP.tine_tip_z
    )