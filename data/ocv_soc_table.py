# =============================================================================
# data/ocv_soc_table.py — NMC OCV-SOC Lookup Table
# =============================================================================
# Open Circuit Voltage (OCV) vs State of Charge (SOC) characterization data
# for NMC (Lithium Nickel Manganese Cobalt Oxide) 18650 cells.
#
# Data sourced from: CALCE NMC cell characterization + standard NMC OCV curves.
# OCV measured after 1-hour rest at each SOC setpoint (removes polarization).
#
# Usage:
#   from data.ocv_soc_table import get_ocv, get_soc_from_ocv
#   voltage = get_ocv(0.80)        # OCV at 80% SOC → ~4.02 V
#   soc     = get_soc_from_ocv(3.7) # SOC at 3.7 V OCV → ~0.42
# =============================================================================
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

# -----------------------------------------------------------------------------
# OCV-SOC TABLE (NMC, 25°C)
# SOC points: 0.0 (fully discharged) → 1.0 (fully charged)
# OCV points: corresponding open-circuit voltages [V]
# -----------------------------------------------------------------------------

# SOC breakpoints [0.0 → 1.0] in 5% increments
SOC_POINTS = np.array([
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00
])

# Corresponding OCV values [V] — NMC characteristic S-curve
OCV_POINTS = np.array([
    2.500, 2.820, 3.000, 3.130, 3.230, 3.310, 3.380, 3.430, 3.480, 3.520,
    3.560, 3.600, 3.650, 3.710, 3.760, 3.820, 3.890, 3.960, 4.040, 4.130, 4.200
])

# -----------------------------------------------------------------------------
# Temperature correction coefficients (dOCV/dT) [V/°C]
# NMC OCV slightly decreases with increasing temperature
# Reference temperature: 25°C
# -----------------------------------------------------------------------------
OCV_TEMP_COEFFICIENT = -0.0004      # V/°C — small but relevant for EKF accuracy


# -----------------------------------------------------------------------------
# CORE LOOKUP FUNCTIONS
# -----------------------------------------------------------------------------

def get_ocv(soc: float, temp_c: float = 25.0) -> float:
    """
    Returns Open Circuit Voltage [V] for a given SOC and temperature.

    Args:
        soc    : State of Charge [0.0 → 1.0]
        temp_c : Cell temperature [°C], default 25°C

    Returns:
        ocv [V]
    """
    soc = float(np.clip(soc, 0.0, 1.0))
    ocv_25 = float(np.interp(soc, SOC_POINTS, OCV_POINTS))
    temp_correction = OCV_TEMP_COEFFICIENT * (temp_c - 25.0)
    return ocv_25 + temp_correction


def get_soc_from_ocv(ocv: float, temp_c: float = 25.0) -> float:
    """
    Returns SOC [0.0 → 1.0] for a given Open Circuit Voltage.
    Inverse lookup via linear interpolation on the OCV-SOC curve.

    Args:
        ocv    : Measured open-circuit voltage [V]
        temp_c : Cell temperature [°C], default 25°C

    Returns:
        soc [0.0 → 1.0]
    """
    # Correct OCV for temperature before inverse lookup
    temp_correction = OCV_TEMP_COEFFICIENT * (temp_c - 25.0)
    ocv_corrected = ocv - temp_correction

    # Clamp to valid OCV range
    ocv_corrected = float(np.clip(ocv_corrected, OCV_POINTS[0], OCV_POINTS[-1]))

    # Inverse interpolation: OCV → SOC (note reversed arrays for interp)
    return float(np.interp(ocv_corrected, OCV_POINTS, SOC_POINTS))


def get_docv_dsoc(soc: float) -> float:
    """
    Returns dOCV/dSOC [V per unit SOC] at a given SOC point.
    Used by the Extended Kalman Filter as the observation Jacobian (H matrix).

    Args:
        soc : State of Charge [0.0 → 1.0]

    Returns:
        dOCV/dSOC [V]
    """
    soc = float(np.clip(soc, 0.0, 1.0))

    # Numerical derivative using central difference
    delta = 0.01
    soc_hi = min(soc + delta, 1.0)
    soc_lo = max(soc - delta, 0.0)
    return (get_ocv(soc_hi) - get_ocv(soc_lo)) / (soc_hi - soc_lo)


# -----------------------------------------------------------------------------
# DIAGNOSTICS — run this file directly to inspect the OCV curve
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("NMC OCV-SOC Table — Diagnostics")
    print("=" * 50)
    print(f"{'SOC':>6}  {'OCV@25C':>10}  {'OCV@45C':>10}  {'dOCV/dSOC':>12}")
    print("-" * 50)
    for soc in SOC_POINTS:
        ocv_25 = get_ocv(soc, temp_c=25.0)
        ocv_45 = get_ocv(soc, temp_c=45.0)
        dv     = get_docv_dsoc(soc)
        print(f"{soc:>6.2f}  {ocv_25:>10.4f}  {ocv_45:>10.4f}  {dv:>12.6f}")

    print("\nInverse lookup test:")
    for v in [2.50, 3.00, 3.50, 3.70, 4.00, 4.20]:
        soc = get_soc_from_ocv(v)
        print(f"  OCV = {v:.2f} V  →  SOC = {soc:.4f} ({soc*100:.1f}%)")