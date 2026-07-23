from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class TissueConfig:
    width_um: float = 1200.0
    height_um: float = 800.0
    nx: int = 121
    ny: int = 81
    diffusion_um2_s: float = 120.0
    degradation_rate_s: float = 0.003
    healthy_uptake_rate_s: float = 0.004
    tumor_uptake_rate_s: float = 0.012
    porosity: float = 0.75
    duration_s: float = 900.0
    snapshots: int = 90
    source_concentration: float = 1.0
    source_radius_um: float = 55.0
    tumor_center_x_um: float = 780.0
    tumor_center_y_um: float = 400.0
    tumor_radius_um: float = 145.0

    @property
    def dx_um(self): return self.width_um / (self.nx - 1)

    @property
    def dy_um(self): return self.height_um / (self.ny - 1)

def make_geometry(c):
    x = np.linspace(0, c.width_um, c.nx)
    y = np.linspace(0, c.height_um, c.ny)
    X, Y = np.meshgrid(x, y)
    sx, sy = 0.08*c.width_um, 0.5*c.height_um
    source = (X-sx)**2 + (Y-sy)**2 <= c.source_radius_um**2
    tumor = (X-c.tumor_center_x_um)**2 + (Y-c.tumor_center_y_um)**2 <= c.tumor_radius_um**2
    uptake = np.full_like(X, c.healthy_uptake_rate_s, dtype=float)
    uptake[tumor] = c.tumor_uptake_rate_s
    return x, y, source, tumor, uptake

def stable_dt(c):
    deff = c.diffusion_um2_s * c.porosity
    inv = 1/c.dx_um**2 + 1/c.dy_um**2
    diff_limit = 1/(2*deff*inv)
    react_limit = 0.25/max(c.degradation_rate_s+c.tumor_uptake_rate_s, 1e-12)
    return 0.8*min(diff_limit, react_limit)

def laplacian_neumann(a, dx, dy):
    p = np.pad(a, 1, mode="edge")
    return ((p[1:-1,2:]-2*p[1:-1,1:-1]+p[1:-1,:-2])/dx**2 +
            (p[2:,1:-1]-2*p[1:-1,1:-1]+p[:-2,1:-1])/dy**2)

def simulate_tissue_diffusion(c):
    if not 0 < c.porosity <= 1:
        raise ValueError("Porosity must be between 0 and 1.")
    x, y, source, tumor, uptake = make_geometry(c)
    C = np.zeros((c.ny, c.nx), dtype=float)
    C[source] = c.source_concentration
    times = np.linspace(0, c.duration_s, c.snapshots)
    fields = np.empty((c.snapshots, c.ny, c.nx), dtype=np.float32)
    fields[0] = C
    dt = stable_dt(c)
    deff = c.diffusion_um2_s*c.porosity
    area = c.dx_um*c.dy_um
    current = 0.0
    nxt = 1
    tumor_abs = healthy_abs = 0.0
    tumor_series = np.zeros(c.snapshots)
    healthy_series = np.zeros(c.snapshots)

    while current < c.duration_s-1e-12:
        step = min(dt, c.duration_s-current)
        uptake_loss = uptake*C
        C = C + step*(deff*laplacian_neumann(C,c.dx_um,c.dy_um)
                      - uptake_loss - c.degradation_rate_s*C)
        C = np.clip(C, 0, None)
        C[source] = c.source_concentration
        absorbed = uptake_loss*step*area
        tumor_abs += float(absorbed[tumor].sum())
        healthy_abs += float(absorbed[~tumor].sum())
        current += step
        while nxt < c.snapshots and current+1e-9 >= times[nxt]:
            fields[nxt] = C
            tumor_series[nxt] = tumor_abs
            healthy_series[nxt] = healthy_abs
            nxt += 1

    tumor_mean = fields[:,tumor].mean(axis=1)
    healthy_mean = fields[:,~tumor].mean(axis=1)
    penetration = np.zeros(c.snapshots)
    source_x = 0.08*c.width_um
    for i, f in enumerate(fields):
        reached = np.where(f.mean(axis=0) >= 0.1*c.source_concentration)[0]
        penetration[i] = max(0, x[reached[-1]]-source_x) if len(reached) else 0

    metrics = pd.DataFrame({
        "time_s": times,
        "tumor_mean_concentration": tumor_mean,
        "healthy_mean_concentration": healthy_mean,
        "penetration_depth_um": penetration,
        "tumor_absorbed_integral": tumor_series,
        "healthy_absorbed_integral": healthy_series,
    })
    return {"times_s":times, "concentration":fields, "tumor_mask":tumor,
            "source_mask":source, "x_um":x, "y_um":y, "metrics":metrics,
            "dt_s":dt}

def concentration_snapshot_to_dataframe(result, index):
    X, Y = np.meshgrid(result["x_um"], result["y_um"])
    return pd.DataFrame({
        "x_um": X.ravel(),
        "y_um": Y.ravel(),
        "concentration": result["concentration"][index].ravel(),
        "is_tumor": result["tumor_mask"].ravel(),
        "is_source": result["source_mask"].ravel(),
    })
