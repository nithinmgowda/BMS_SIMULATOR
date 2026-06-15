# =============================================================================
# core/cell_model.py — Single Cell Equivalent Circuit Model (ECM)
# =============================================================================
# Models a single Li-ion NMC cell using a 1RC Thevenin Equivalent Circuit:
#
#         R0          R1
#   +---/\/\/---+---/\/\/---+
#   |           |           |
#  OCV         C1          Vt
#   |           |           |
#   +-----------+-----------+
#
#   OCV : Open Circuit Voltage (SOC-dependent, from lookup table)
#   R0  : Internal ohmic resistance (immediate voltage drop)
#   R1  : Polarization resistance  (delayed response)
#   C1  : Polarization capacitance (time constant τ = R1*C1)
#   Vt  : Terminal voltage (measured output)
#
# Terminal voltage equation:
#   Vt = OCV(SOC) - I*R0 - V_RC
#   dV_RC/dt = -V_RC/(R1*C1) + I/C1
#
# Thermal model:
#   m*Cp * dT/dt = I²*R0 - (T - T_amb)/R_th
#
# Sign convention: I > 0 = discharge, I < 0 = charge
# =============================================================================
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from dataclasses import dataclass, field
from data.ocv_soc_table import get_ocv, get_soc_from_ocv, get_docv_dsoc
from config import (
    CELL_NOMINAL_CAPACITY_AH,
    CELL_NOMINAL_VOLTAGE_V,
    CELL_MAX_VOLTAGE_V,
    CELL_MIN_VOLTAGE_V,
    CELL_R0_OHMS,
    CELL_R1_OHMS,
    CELL_C1_FARADS,
    CELL_THERMAL_MASS_J_K,
    CELL_THERMAL_RESISTANCE,
    AMBIENT_TEMP_C,
    CELL_CAPACITY_VARIATION,
    CELL_RESISTANCE_VARIATION,
    SOC_INITIAL,
    SIM_TIMESTEP_S,
)


# =============================================================================
# CELL STATE — snapshot of all cell variables at a given tick
# =============================================================================

@dataclass
class CellState:
    """
    Complete state of one cell at a single simulation tick.
    Returned by CellModel.step() every tick.
    """
    cell_id      : int   = 0
    soc          : float = SOC_INITIAL   # State of Charge [0.0 → 1.0]
    ocv          : float = 0.0           # Open Circuit Voltage [V]
    v_rc         : float = 0.0           # RC pair voltage [V]
    voltage      : float = 0.0           # Terminal voltage [V]
    current      : float = 0.0           # Applied current [A] (+ = discharge)
    temperature  : float = AMBIENT_TEMP_C  # Cell temperature [°C]
    r0           : float = CELL_R0_OHMS  # Ohmic resistance [Ω] (grows with SOH)
    capacity_ah  : float = CELL_NOMINAL_CAPACITY_AH  # Current capacity [Ah]
    heat_gen_w   : float = 0.0           # Heat generated this tick [W]
    is_balancing : bool  = False         # Whether cell is being balanced


# =============================================================================
# CELL MODEL
# =============================================================================

class CellModel:
    """
    Physics-based single Li-ion NMC cell using 1RC Thevenin ECM.

    Each cell maintains its own internal state and is stepped
    independently by the pack model every simulation tick.

    Args:
        cell_id  : Unique cell index in the pack (0 → NUM_CELLS-1)
        soc_init : Initial SOC [0.0 → 1.0]
        seed     : Random seed for cell-to-cell variation
    """

    def __init__(self, cell_id: int, soc_init: float = SOC_INITIAL, seed: int = None):
        self.cell_id  = cell_id
        self.dt       = SIM_TIMESTEP_S

        # Apply cell-to-cell manufacturing variation
        rng = np.random.default_rng(seed if seed is not None else cell_id * 137)
        cap_factor = 1.0 + rng.uniform(-CELL_CAPACITY_VARIATION, CELL_CAPACITY_VARIATION)
        res_factor = 1.0 + rng.uniform(-CELL_RESISTANCE_VARIATION, CELL_RESISTANCE_VARIATION)

        # Cell parameters (slightly varied from nominal)
        self.capacity_ah  = CELL_NOMINAL_CAPACITY_AH * cap_factor   # [Ah]
        self.capacity_as  = self.capacity_ah * 3600                  # [As = Coulombs]
        self.r0           = CELL_R0_OHMS  * res_factor               # [Ω]
        self.r1           = CELL_R1_OHMS  * res_factor               # [Ω]
        self.c1           = CELL_C1_FARADS                           # [F]
        self.tau          = self.r1 * self.c1                        # RC time constant [s]

        # Thermal parameters
        self.thermal_mass = CELL_THERMAL_MASS_J_K                    # [J/K]
        self.r_thermal    = CELL_THERMAL_RESISTANCE                  # [K/W]

        # Initial state
        self.soc         = float(np.clip(soc_init, 0.0, 1.0))
        self.v_rc        = 0.0                                       # RC voltage [V]
        self.temperature = AMBIENT_TEMP_C                            # [°C]
        self.current     = 0.0                                       # [A]
        self.is_balancing = False

        # Cycle accumulator for SOH model
        self._charge_accumulator_as = 0.0   # Tracks Ah throughput

    # -------------------------------------------------------------------------
    # MAIN STEP — call every simulation tick
    # -------------------------------------------------------------------------

    def step(self, current: float, ambient_temp: float = AMBIENT_TEMP_C) -> CellState:
        """
        Advance cell state by one time step (dt seconds).

        Args:
            current      : Applied current [A]. Positive = discharge, negative = charge.
            ambient_temp : Ambient temperature [°C]

        Returns:
            CellState snapshot after this tick
        """
        self.current = current

        # --- 1. Update SOC via Coulomb Counting ----------------------------
        # ΔSOC = -I * dt / Q_nominal   (negative because discharge reduces SOC)
        delta_soc = -(current * self.dt) / self.capacity_as
        self.soc  = float(np.clip(self.soc + delta_soc, 0.0, 1.0))

        # Accumulate absolute charge throughput (for SOH model)
        self._charge_accumulator_as += abs(current) * self.dt

        # --- 2. OCV from SOC lookup ----------------------------------------
        ocv = get_ocv(self.soc, temp_c=self.temperature)

        # --- 3. RC pair dynamics (discretized) --------------------------------
        # Analytical solution to: dV_RC/dt = -V_RC/τ + I/C1
        # V_RC[k+1] = V_RC[k]*exp(-dt/τ) + I*R1*(1 - exp(-dt/τ))
        exp_factor = np.exp(-self.dt / self.tau)
        self.v_rc  = self.v_rc * exp_factor + current * self.r1 * (1.0 - exp_factor)

        # --- 4. Terminal voltage -------------------------------------------
        # Vt = OCV - I*R0 - V_RC
        voltage = ocv - current * self.r0 - self.v_rc

        # Clamp terminal voltage to physical limits
        voltage = float(np.clip(voltage, CELL_MIN_VOLTAGE_V - 0.1, CELL_MAX_VOLTAGE_V + 0.1))

        # --- 5. Thermal model -----------------------------------------------
        # Heat generated: Q_gen = I²*R0 + I²*R1 (both resistors dissipate heat)
        heat_gen = (current ** 2) * (self.r0 + self.r1)   # [W]

        # Heat dissipated to ambient: Q_diss = (T - T_amb) / R_th
        heat_diss = (self.temperature - ambient_temp) / self.r_thermal   # [W]

        # Temperature update: dT/dt = (Q_gen - Q_diss) / (m*Cp)
        dT = (heat_gen - heat_diss) * self.dt / self.thermal_mass
        self.temperature = float(self.temperature + dT)

        # Return full state snapshot
        return CellState(
            cell_id      = self.cell_id,
            soc          = self.soc,
            ocv          = ocv,
            v_rc         = self.v_rc,
            voltage      = voltage,
            current      = current,
            temperature  = self.temperature,
            r0           = self.r0,
            capacity_ah  = self.capacity_ah,
            heat_gen_w   = heat_gen,
            is_balancing = self.is_balancing,
        )

    # -------------------------------------------------------------------------
    # SOH INTERFACE — called by soh_estimator.py
    # -------------------------------------------------------------------------

    def get_charge_throughput_ah(self) -> float:
        """Returns total cumulative Ah throughput since init."""
        return self._charge_accumulator_as / 3600.0

    def apply_soh_update(self, new_capacity_ah: float, new_r0: float):
        """
        Called by SOH estimator to degrade cell parameters over time.

        Args:
            new_capacity_ah : Updated capacity after degradation [Ah]
            new_r0          : Updated ohmic resistance after degradation [Ω]
        """
        self.capacity_ah = new_capacity_ah
        self.capacity_as = new_capacity_ah * 3600.0
        self.r0          = new_r0

    # -------------------------------------------------------------------------
    # BALANCING INTERFACE — called by balancer.py
    # -------------------------------------------------------------------------

    def apply_balance_current(self, balance_current: float):
        """
        Injects a balancing current into the cell for one tick.
        Passive: balance_current > 0 (bleed discharge)
        Active:  balance_current < 0 (charge from neighbor)

        This is called by the balancer BEFORE step() each tick,
        so the net current passed to step() = load_current + balance_current.
        """
        self._balance_current = balance_current
        self.is_balancing = abs(balance_current) > 1e-6

    def get_balance_current(self) -> float:
        return getattr(self, "_balance_current", 0.0)

    # -------------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------------

    def get_ocv(self) -> float:
        """Current OCV based on SOC and temperature."""
        return get_ocv(self.soc, self.temperature)

    def get_docv_dsoc(self) -> float:
        """dOCV/dSOC at current SOC — used by EKF Jacobian."""
        return get_docv_dsoc(self.soc)

    def reset(self, soc_init: float = SOC_INITIAL):
        """Reset cell to initial conditions (new cycle)."""
        self.soc          = float(np.clip(soc_init, 0.0, 1.0))
        self.v_rc         = 0.0
        self.temperature  = AMBIENT_TEMP_C
        self.current      = 0.0
        self.is_balancing = False
        self._charge_accumulator_as = 0.0

    def __repr__(self):
        return (
            f"CellModel(id={self.cell_id}, "
            f"SOC={self.soc:.3f}, "
            f"V={self.get_ocv():.3f}V, "
            f"T={self.temperature:.1f}°C, "
            f"Q={self.capacity_ah:.3f}Ah)"
        )


# =============================================================================
# DIAGNOSTICS — run directly to simulate a single cell discharge
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from config import PACK_NOMINAL_CAPACITY_AH

    print("=" * 55)
    print("Cell Model — 1C Constant Current Discharge Diagnostics")
    print("=" * 55)

    cell    = CellModel(cell_id=0, soc_init=1.0, seed=0)
    I_1C    = PACK_NOMINAL_CAPACITY_AH / 8   # Per-cell current (pack / parallel)

    times, voltages, socs, temps, v_rcs = [], [], [], [], []

    t = 0
    while cell.soc > 0.02 and t < 7200:
        state = cell.step(current=I_1C)
        times.append(t)
        voltages.append(state.voltage)
        socs.append(state.soc * 100)
        temps.append(state.temperature)
        v_rcs.append(state.v_rc)
        t += SIM_TIMESTEP_S

    print(f"Discharge complete at t={t}s")
    print(f"  Final SOC      : {cell.soc*100:.1f}%")
    print(f"  Final Voltage  : {voltages[-1]:.3f} V")
    print(f"  Final Temp     : {cell.temperature:.2f} °C")
    print(f"  Ah throughput  : {cell.get_charge_throughput_ah():.3f} Ah")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Single Cell 1C Discharge — ECM Response", fontweight="bold")

    axes[0].plot(times, voltages, color="steelblue", linewidth=1.5)
    axes[0].set_ylabel("Terminal Voltage [V]")
    axes[0].set_ylim(2.4, 4.3)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, socs, color="forestgreen", linewidth=1.5)
    axes[1].set_ylabel("SOC [%]")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, temps, color="tomato", linewidth=1.5)
    axes[2].set_ylabel("Temperature [°C]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("logs/cell_model_discharge.png", dpi=120)
    plt.show()