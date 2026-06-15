# =============================================================================
# core/soh_estimator.py — SOH Estimation
# =============================================================================
# Tracks battery State of Health (SOH) for all 64 cells using two methods:
#
#   1. Capacity Fade Model
#      SOH_Q(n) = exp(-α * n^β)
#      Capacity shrinks sub-linearly with cycle count.
#      α = degradation rate, β = shape exponent (< 1 = accelerating fade)
#
#   2. Internal Resistance Growth Model
#      R(n) = R0_fresh * (1 + γ * n)
#      Resistance grows linearly with cycle count.
#      γ = resistance growth rate per cycle
#
#   3. Composite SOH
#      SOH = 0.6 * SOH_Q + 0.4 * SOH_R
#      Weighted combination — capacity fade weighted more (industry standard)
#
#   4. Stress Factors (modifiers on degradation rate)
#      - Temperature stress : accelerates fade at high/low temps (Arrhenius)
#      - DoD stress         : deeper discharges degrade faster
#      - C-rate stress      : high current increases degradation
#
#   5. Remaining Useful Life (RUL)
#      Estimated cycles remaining before SOH < EOL threshold (70%)
#
# SOH updates happen at end-of-cycle (not every tick) to match
# real BMS behaviour where degradation is tracked per charge cycle.
# =============================================================================
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from config import (
    NUM_CELLS,
    CELL_NOMINAL_CAPACITY_AH,
    CELL_R0_OHMS,
    SOH_INITIAL,
    EOL_SOH_THRESHOLD,
    SOH_ALPHA,
    SOH_BETA,
    SOH_GAMMA,
    AMBIENT_TEMP_C,
    SIM_TIMESTEP_S,
)


# =============================================================================
# SOH STATE — per-cell SOH snapshot
# =============================================================================

@dataclass
class SOHState:
    """SOH metrics for a single cell."""
    cell_id         : int   = 0
    soh             : float = 1.0    # Composite SOH [0→1]
    soh_capacity    : float = 1.0    # Capacity-based SOH [0→1]
    soh_resistance  : float = 1.0    # Resistance-based SOH [0→1]
    capacity_ah     : float = CELL_NOMINAL_CAPACITY_AH   # Current capacity [Ah]
    resistance_ohm  : float = CELL_R0_OHMS               # Current R0 [Ω]
    cycle_count     : float = 0.0    # Equivalent full cycles
    rul_cycles      : float = 0.0    # Remaining Useful Life [cycles]
    is_eol          : bool  = False  # End-of-life flag


@dataclass
class PackSOHState:
    """Pack-level SOH aggregated from all cells."""
    pack_soh        : float = 1.0
    min_cell_soh    : float = 1.0    # Weakest cell limits pack
    max_cell_soh    : float = 1.0
    mean_capacity_ah: float = CELL_NOMINAL_CAPACITY_AH
    mean_resistance : float = CELL_R0_OHMS
    total_cycles    : float = 0.0    # Mean equivalent cycles across pack
    cells_at_eol    : int   = 0
    cell_soh_states : List[SOHState] = field(default_factory=list)


# =============================================================================
# STRESS FACTOR MODELS
# =============================================================================

def _temperature_stress(temp_c: float, ref_temp_c: float = 25.0) -> float:
    """
    Arrhenius-based temperature stress multiplier.
    Degradation accelerates exponentially above/below reference temp.

    Returns multiplier > 1 when temp deviates from reference.
    """
    activation_energy = 0.5    # Normalised activation energy [eV-like]
    delta_temp = abs(temp_c - ref_temp_c)
    return 1.0 + activation_energy * (delta_temp / 30.0) ** 2


def _dod_stress(depth_of_discharge: float) -> float:
    """
    Depth of Discharge stress multiplier.
    Shallow cycles (DoD < 50%) degrade much slower than deep cycles.

    Returns multiplier ≥ 1.0 (deeper = more stress).
    """
    # Power law: stress = DoD^0.5 normalised to 1.0 at DoD=1.0
    return max(1.0, (depth_of_discharge ** 0.5) / 1.0)


def _c_rate_stress(c_rate: float) -> float:
    """
    C-rate stress multiplier.
    High current during charge/discharge increases lithium plating risk.

    Returns multiplier ≥ 1.0 (higher C = more stress).
    """
    return max(1.0, 1.0 + 0.2 * (c_rate - 1.0))


# =============================================================================
# SINGLE CELL SOH ESTIMATOR
# =============================================================================

class CellSOHEstimator:
    """
    Tracks SOH for a single cell across charge/discharge cycles.

    Degradation is applied at end-of-cycle, not every tick.
    Mid-cycle, it accumulates charge throughput and operating conditions.
    """

    def __init__(self, cell_id: int, capacity_ah: float = CELL_NOMINAL_CAPACITY_AH,
                 r0_fresh: float = CELL_R0_OHMS):
        self.cell_id       = cell_id
        self.capacity_fresh = capacity_ah      # Original rated capacity [Ah]
        self.r0_fresh       = r0_fresh         # Original resistance [Ω]

        # Current degraded values
        self.capacity_ah   = capacity_ah
        self.r0            = r0_fresh

        # Cycle tracking
        self.cycle_count   = 0.0              # Equivalent full cycles
        self.soh           = SOH_INITIAL
        self.soh_capacity  = SOH_INITIAL
        self.soh_resistance= SOH_INITIAL

        # Within-cycle accumulators
        self._charge_in_as   = 0.0           # Charge accumulated this half-cycle
        self._discharge_as   = 0.0
        self._temp_samples   = []            # Temperature samples this cycle
        self._soc_max        = 0.0           # Peak SOC this cycle
        self._soc_min        = 1.0           # Trough SOC this cycle
        self._prev_current   = 0.0
        self._in_discharge   = False

    # -------------------------------------------------------------------------
    # TICK UPDATE — called every simulation tick
    # -------------------------------------------------------------------------

    def tick(self, current: float, soc: float, temp_c: float):
        """
        Accumulate within-cycle data every simulation tick.

        Args:
            current : Cell current [A] (+ discharge, - charge)
            soc     : Current SOC [0→1]
            temp_c  : Cell temperature [°C]
        """
        dt_h = SIM_TIMESTEP_S / 3600.0   # dt in hours

        if current > 0:
            self._discharge_as += current * SIM_TIMESTEP_S
        else:
            self._charge_in_as += abs(current) * SIM_TIMESTEP_S

        self._temp_samples.append(temp_c)
        self._soc_max = max(self._soc_max, soc)
        self._soc_min = min(self._soc_min, soc)

        # Detect cycle completion: discharge → charge transition
        if self._prev_current > 0.1 and current < -0.1:
            self._complete_cycle()

        self._prev_current = current

    # -------------------------------------------------------------------------
    # CYCLE COMPLETION — apply degradation
    # -------------------------------------------------------------------------

    def _complete_cycle(self):
        """
        Called at end of each discharge→charge transition.
        Applies stress-weighted degradation to capacity and resistance.
        """
        if self._discharge_as < 100:    # Ignore micro-cycles
            self._reset_accumulators()
            return

        # Equivalent full cycle contribution (based on Ah throughput)
        discharge_ah   = self._discharge_as / 3600.0
        cycle_fraction = discharge_ah / self.capacity_fresh
        self.cycle_count += cycle_fraction

        # Compute stress factors from this cycle's conditions
        mean_temp = float(np.mean(self._temp_samples)) if self._temp_samples else AMBIENT_TEMP_C
        dod       = self._soc_max - self._soc_min
        c_rate    = (self._discharge_as / SIM_TIMESTEP_S) / max(self.capacity_fresh, 1e-6) \
                    if self._discharge_as > 0 else 1.0

        stress_t  = _temperature_stress(mean_temp)
        stress_d  = _dod_stress(dod)
        stress_c  = _c_rate_stress(c_rate)

# Average stress — additive not multiplicative
        stress = (stress_t + stress_d + stress_c) / 3.0

# Effective cycle count
        n_eff = self.cycle_count * stress

# Capacity fade
        soh_q = np.exp(-SOH_ALPHA * (n_eff ** SOH_BETA))
        soh_q = float(np.clip(soh_q, 0.0, 1.0))

# Resistance growth — use base cycle count, not stress-weighted
        r_ratio = 1.0 + SOH_GAMMA * self.cycle_count
        soh_r   = float(np.clip(1.0 / r_ratio, 0.0, 1.0))

        # --- Composite SOH (capacity-weighted) ---
        self.soh_capacity   = soh_q
        self.soh_resistance = soh_r
        self.soh            = 0.6 * soh_q + 0.4 * soh_r

        # Update physical parameters
        self.capacity_ah = self.capacity_fresh * soh_q
        self.r0          = self.r0_fresh * r_ratio

        self._reset_accumulators()

    def _reset_accumulators(self):
        self._charge_in_as  = 0.0
        self._discharge_as  = 0.0
        self._temp_samples  = []
        self._soc_max       = 0.0
        self._soc_min       = 1.0

    # -------------------------------------------------------------------------
    # RUL ESTIMATION
    # -------------------------------------------------------------------------

    def estimate_rul(self) -> float:
        """
        Estimate Remaining Useful Life in equivalent full cycles.
        Solves: EOL_SOH = exp(-α * n_eol^β) for n_eol, then RUL = n_eol - n_current
        """
        if self.soh <= EOL_SOH_THRESHOLD:
            return 0.0
        try:
            # Invert capacity fade: n_eol = (-ln(EOL_threshold) / α)^(1/β)
            n_eol = (-np.log(EOL_SOH_THRESHOLD) / SOH_ALPHA) ** (1.0 / SOH_BETA)
            rul   = max(0.0, n_eol - self.cycle_count)
        except (ValueError, ZeroDivisionError):
            rul = 0.0
        return float(rul)

    # -------------------------------------------------------------------------
    # STATE SNAPSHOT
    # -------------------------------------------------------------------------

    def get_state(self) -> SOHState:
        return SOHState(
            cell_id        = self.cell_id,
            soh            = self.soh,
            soh_capacity   = self.soh_capacity,
            soh_resistance = self.soh_resistance,
            capacity_ah    = self.capacity_ah,
            resistance_ohm = self.r0,
            cycle_count    = self.cycle_count,
            rul_cycles     = self.estimate_rul(),
            is_eol         = self.soh <= EOL_SOH_THRESHOLD,
        )


# =============================================================================
# PACK-LEVEL SOH ESTIMATOR
# =============================================================================

class PackSOHEstimator:
    """
    Manages SOH estimation for all 64 cells.
    Feeds degraded parameters back into PackModel and PackSOCEstimator.

    Used by sim_loop.py — call tick() every simulation tick.
    """

    def __init__(self, pack_model, soc_estimator=None):
        """
        Args:
            pack_model    : PackModel instance
            soc_estimator : PackSOCEstimator (optional — for EKF parameter sync)
        """
        self.soc_estimator = soc_estimator
        self.pack_model    = pack_model

        # One SOH estimator per cell
        self.cell_estimators: List[CellSOHEstimator] = [
            CellSOHEstimator(
                cell_id     = cell.cell_id,
                capacity_ah = cell.capacity_ah,
                r0_fresh    = cell.r0,
            )
            for cell in pack_model.all_cells
        ]

    def tick(self, cell_states) -> PackSOHState:
        """
        Update all cell SOH estimators and push degraded params to pack.

        Args:
            cell_states : List[CellState] from PackModel.step()

        Returns:
            PackSOHState snapshot
        """
        soh_states: List[SOHState] = []

        for i, state in enumerate(cell_states):
            est = self.cell_estimators[i]

            # Accumulate cycle data
            est.tick(
                current = state.current,
                soc     = state.soc,
                temp_c  = state.temperature,
            )

            soh_state = est.get_state()
            soh_states.append(soh_state)

            # Push degraded parameters back to pack cell
            self.pack_model.update_soh(
                cell_id        = state.cell_id,
                soh_value      = soh_state.soh,
                new_capacity_ah= soh_state.capacity_ah,
                new_r0         = soh_state.resistance_ohm,
            )

            # Sync EKF with new capacity and resistance
            if self.soc_estimator is not None:
                self.soc_estimator.notify_soh_update(
                    cell_id        = state.cell_id,
                    new_capacity_ah= soh_state.capacity_ah,
                    new_r0         = soh_state.resistance_ohm,
                )

        # Pack-level aggregation
        soh_values   = np.array([s.soh          for s in soh_states])
        cap_values   = np.array([s.capacity_ah  for s in soh_states])
        res_values   = np.array([s.resistance_ohm for s in soh_states])
        cycle_values = np.array([s.cycle_count  for s in soh_states])

        return PackSOHState(
            pack_soh         = float(np.mean(soh_values)),
            min_cell_soh     = float(np.min(soh_values)),
            max_cell_soh     = float(np.max(soh_values)),
            mean_capacity_ah = float(np.mean(cap_values)),
            mean_resistance  = float(np.mean(res_values)),
            total_cycles     = float(np.mean(cycle_values)),
            cells_at_eol     = int(np.sum([s.is_eol for s in soh_states])),
            cell_soh_states  = soh_states,
        )

    def get_pack_soh(self) -> float:
        """Quick pack SOH — mean of all cell SOHs."""
        return float(np.mean([e.soh for e in self.cell_estimators]))

    def force_cycle_update(self):
        """Force end-of-cycle degradation on all cells (for testing)."""
        for est in self.cell_estimators:
            est._complete_cycle()


# =============================================================================
# DIAGNOSTICS — simulate 500 cycles of degradation
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 55)
    print("SOH Estimator — 500 Cycle Degradation Simulation")
    print("=" * 55)

    est = CellSOHEstimator(cell_id=0)

    cycles, soh_vals, cap_vals, res_vals, rul_vals = [], [], [], [], []

    for cycle in range(500):
        cap_ah      = est.capacity_ah
        i_cell      = cap_ah / 1.0
        soc         = 1.0
        capacity_as = cap_ah * 3600

        # Discharge phase
        for t in range(3600):
            soc -= (i_cell * SIM_TIMESTEP_S) / capacity_as
            soc  = max(soc, 0.0)
            est.tick(current=i_cell, soc=soc, temp_c=30.0)

        # Charge phase transition — triggers _complete_cycle()
        for t in range(10):
            est.tick(current=-i_cell, soc=soc, temp_c=30.0)

        state = est.get_state()
        cycles.append(cycle)
        soh_vals.append(state.soh * 100)
        cap_vals.append(state.capacity_ah)
        res_vals.append(state.resistance_ohm * 1000)
        rul_vals.append(state.rul_cycles)

    print(f"  After 500 cycles:")
    print(f"  SOH          : {soh_vals[-1]:.2f}%")
    print(f"  Capacity     : {cap_vals[-1]:.3f} Ah  (was {CELL_NOMINAL_CAPACITY_AH:.1f} Ah)")
    print(f"  Resistance   : {res_vals[-1]:.2f} mΩ  (was {CELL_R0_OHMS*1000:.1f} mΩ)")
    print(f"  RUL          : {rul_vals[-1]:.0f} cycles")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle("Cell SOH Degradation — 500 Cycles", fontweight="bold")

    axes[0].plot(cycles, soh_vals, color="steelblue", linewidth=1.8)
    axes[0].axhline(EOL_SOH_THRESHOLD * 100, color="red",
                    linestyle="--", linewidth=1.0, label=f"EOL ({EOL_SOH_THRESHOLD*100:.0f}%)")
    axes[0].set_ylabel("SOH [%]")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cycles, cap_vals, color="forestgreen", linewidth=1.8)
    axes[1].set_ylabel("Capacity [Ah]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(cycles, res_vals, color="tomato", linewidth=1.8)
    axes[2].set_ylabel("Internal Resistance [mΩ]")
    axes[2].set_xlabel("Cycle Number")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("logs/soh_degradation.png", dpi=120)
    plt.show()