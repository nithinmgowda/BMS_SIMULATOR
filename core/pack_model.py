# =============================================================================
# core/pack_model.py — 8S8P Battery Pack Model
# =============================================================================
# Wires 64 Li-ion NMC cells into an 8S8P topology:
#
#   8 parallel groups × 8 in series = 64 total cells
#
#   Pack topology:
#   [P-Group 0] --- [P-Group 1] --- ... --- [P-Group 7]
#    (8 cells)       (8 cells)               (8 cells)
#      series →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→
#
#   Within each parallel group:
#   Cell_0 ║ Cell_1 ║ Cell_2 ║ ... ║ Cell_7  (parallel)
#
#   Pack voltage     = sum of series group voltages         (~28.8 V nominal)
#   Pack current     = total load current                   (split across parallel cells)
#   Cell current     = pack_current / NUM_CELLS_PARALLEL    (2 A per cell at 1C)
#   Pack capacity    = cell_capacity × NUM_CELLS_PARALLEL   (16 Ah)
#
# Naming convention:
#   cell[s][p] → series group s (0-7), parallel index p (0-7)
#   cell_id    = s * NUM_CELLS_PARALLEL + p  → 0 to 63
#
# =============================================================================
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from core.cell_model import CellModel, CellState
from config import (
    NUM_CELLS_SERIES,
    NUM_CELLS_PARALLEL,
    NUM_CELLS,
    PACK_NOMINAL_VOLTAGE_V,
    PACK_NOMINAL_CAPACITY_AH,
    CELL_MAX_VOLTAGE_V,
    CELL_MIN_VOLTAGE_V,
    AMBIENT_TEMP_C,
    SOC_INITIAL,
    SIM_TIMESTEP_S,
)


# =============================================================================
# PACK STATE — full snapshot every tick
# =============================================================================

@dataclass
class PackState:
    """
    Complete pack-level state at a single simulation tick.
    Aggregated from all 64 cell states.
    """
    # Pack-level aggregates
    pack_voltage        : float = 0.0    # Sum of series group voltages [V]
    pack_current        : float = 0.0    # Total load current [A]
    pack_soc            : float = 0.0    # Capacity-weighted mean SOC [0→1]
    pack_soh            : float = 1.0    # Mean SOH across all cells [0→1]
    pack_power_w        : float = 0.0    # Instantaneous power [W]
    pack_energy_wh      : float = 0.0    # Cumulative energy [Wh]

    # Temperature
    max_temp_c          : float = 0.0    # Hottest cell [°C]
    min_temp_c          : float = 0.0    # Coolest cell [°C]
    mean_temp_c         : float = 0.0    # Average temperature [°C]

    # Cell voltage stats
    max_cell_voltage    : float = 0.0    # Highest cell voltage [V]
    min_cell_voltage    : float = 0.0    # Lowest cell voltage [V]
    mean_cell_voltage   : float = 0.0    # Average cell voltage [V]
    delta_voltage       : float = 0.0    # Max - Min cell voltage [V]

    # SOC stats
    max_cell_soc        : float = 0.0
    min_cell_soc        : float = 0.0
    delta_soc           : float = 0.0    # SOC imbalance across cells

    # Per-cell states (list of 64 CellState objects)
    cell_states         : List[CellState] = field(default_factory=list)

    # Simulation time
    time_s              : float = 0.0


# =============================================================================
# PACK MODEL
# =============================================================================

class PackModel:
    """
    8S8P Li-ion NMC battery pack.
    Manages 64 CellModel instances in series-parallel topology.

    Args:
        soc_init     : Initial SOC for all cells [0.0 → 1.0]
        soc_spread   : Optional SOC spread across cells (simulates imbalance)
                       e.g. 0.05 = ±5% SOC variation at init
        ambient_temp : Starting ambient temperature [°C]
    """

    def __init__(
        self,
        soc_init     : float = SOC_INITIAL,
        soc_spread   : float = 0.03,
        ambient_temp : float = AMBIENT_TEMP_C,
    ):
        self.n_series   = NUM_CELLS_SERIES
        self.n_parallel = NUM_CELLS_PARALLEL
        self.n_cells    = NUM_CELLS
        self.ambient    = ambient_temp
        self.time_s     = 0.0
        self._energy_wh = 0.0

        # Build cell grid: cells[s][p] where s=series group, p=parallel index
        rng = np.random.default_rng(seed=42)
        self.cells: List[List[CellModel]] = []

        for s in range(self.n_series):
            group = []
            for p in range(self.n_parallel):
                cell_id  = s * self.n_parallel + p
                soc_var  = rng.uniform(-soc_spread, soc_spread)
                soc_cell = float(np.clip(soc_init + soc_var, 0.05, 0.99))
                cell     = CellModel(cell_id=cell_id, soc_init=soc_cell, seed=cell_id)
                group.append(cell)
            self.cells.append(group)

        # Flat list of all 64 cells for convenience
        self.all_cells: List[CellModel] = [
            self.cells[s][p]
            for s in range(self.n_series)
            for p in range(self.n_parallel)
        ]

        # SOH per cell (initialised to 1.0, updated by soh_estimator)
        self.soh = np.ones(self.n_cells, dtype=float)

    # -------------------------------------------------------------------------
    # MAIN STEP
    # -------------------------------------------------------------------------

    def step(self, pack_current: float) -> PackState:
        """
        Advance all 64 cells by one time step.

        Args:
            pack_current : Total pack load current [A]
                           Positive = discharge, Negative = charge

        Returns:
            PackState — full pack snapshot after this tick
        """
        # Current splits equally across parallel cells in each group
        cell_current = pack_current / self.n_parallel

        cell_states: List[CellState] = []
        group_voltages = np.zeros(self.n_series)

        for s in range(self.n_series):
            group_cell_voltages = []

            for p in range(self.n_parallel):
                cell = self.cells[s][p]

                # Net current = load share + any balancing current
                net_current = cell_current + cell.get_balance_current()

                # Step the cell physics
                state = cell.step(current=net_current, ambient_temp=self.ambient)
                cell_states.append(state)
                group_cell_voltages.append(state.voltage)

            # Series group voltage = mean of parallel cell voltages
            # (In ideal parallel: all cells share same terminal voltage)
            group_voltages[s] = float(np.mean(group_cell_voltages))

        # Pack voltage = sum of all series group voltages
        pack_voltage = float(np.sum(group_voltages))

        # Power and energy
        pack_power   = pack_voltage * pack_current
        self._energy_wh += abs(pack_power) * SIM_TIMESTEP_S / 3600.0

        # Aggregate cell stats
        voltages  = np.array([cs.voltage     for cs in cell_states])
        socs      = np.array([cs.soc         for cs in cell_states])
        temps     = np.array([cs.temperature for cs in cell_states])

        # Pack SOC = capacity-weighted mean across all cells
        capacities = np.array([cs.capacity_ah for cs in cell_states])
        pack_soc   = float(np.average(socs, weights=capacities))

        # Pack SOH = mean SOH (updated externally by soh_estimator)
        pack_soh = float(np.mean(self.soh))

        self.time_s += SIM_TIMESTEP_S

        return PackState(
            pack_voltage      = pack_voltage,
            pack_current      = pack_current,
            pack_soc          = pack_soc,
            pack_soh          = pack_soh,
            pack_power_w      = pack_power,
            pack_energy_wh    = self._energy_wh,
            max_temp_c        = float(np.max(temps)),
            min_temp_c        = float(np.min(temps)),
            mean_temp_c       = float(np.mean(temps)),
            max_cell_voltage  = float(np.max(voltages)),
            min_cell_voltage  = float(np.min(voltages)),
            mean_cell_voltage = float(np.mean(voltages)),
            delta_voltage     = float(np.max(voltages) - np.min(voltages)),
            max_cell_soc      = float(np.max(socs)),
            min_cell_soc      = float(np.min(socs)),
            delta_soc         = float(np.max(socs) - np.min(socs)),
            cell_states       = cell_states,
            time_s            = self.time_s,
        )

    # -------------------------------------------------------------------------
    # ACCESSORS
    # -------------------------------------------------------------------------

    def get_cell(self, series: int, parallel: int) -> CellModel:
        """Get cell at grid position [series][parallel]."""
        return self.cells[series][parallel]

    def get_cell_by_id(self, cell_id: int) -> CellModel:
        """Get cell by flat cell_id (0 → 63)."""
        return self.all_cells[cell_id]

    def get_series_group(self, s: int) -> List[CellModel]:
        """Get all 8 parallel cells in series group s."""
        return self.cells[s]

    def get_cell_voltages(self) -> np.ndarray:
        """Current terminal voltages of all 64 cells [V]."""
        return np.array([c.get_ocv() - c.current * c.r0 for c in self.all_cells])

    def get_cell_socs(self) -> np.ndarray:
        """Current SOC of all 64 cells [0→1]."""
        return np.array([c.soc for c in self.all_cells])

    def get_cell_temperatures(self) -> np.ndarray:
        """Current temperature of all 64 cells [°C]."""
        return np.array([c.temperature for c in self.all_cells])

    def get_series_group_voltages(self) -> np.ndarray:
        """Mean voltage of each of the 8 series groups [V]."""
        return np.array([
            np.mean([self.cells[s][p].get_ocv() for p in range(self.n_parallel)])
            for s in range(self.n_series)
        ])

    def update_soh(self, cell_id: int, soh_value: float,
                   new_capacity_ah: float, new_r0: float):
        """
        Called by soh_estimator.py to push degraded parameters into a cell.

        Args:
            cell_id        : Target cell (0→63)
            soh_value      : Updated SOH [0→1]
            new_capacity_ah: Degraded capacity [Ah]
            new_r0         : Increased resistance [Ω]
        """
        self.soh[cell_id] = soh_value
        self.all_cells[cell_id].apply_soh_update(new_capacity_ah, new_r0)

    def set_ambient_temperature(self, temp_c: float):
        """Update ambient temperature for all cells."""
        self.ambient = temp_c

    def reset(self, soc_init: float = SOC_INITIAL):
        """Reset entire pack to initial conditions."""
        for cell in self.all_cells:
            cell.reset(soc_init=soc_init)
        self._energy_wh = 0.0
        self.time_s     = 0.0
        self.soh        = np.ones(self.n_cells, dtype=float)

    def __repr__(self):
        socs = self.get_cell_socs()
        temps = self.get_cell_temperatures()
        return (
            f"PackModel(8S8P | 64 cells | "
            f"SOC={np.mean(socs)*100:.1f}% | "
            f"ΔV={np.ptp(self.get_cell_voltages())*1000:.1f}mV | "
            f"T_max={np.max(temps):.1f}°C)"
        )


# =============================================================================
# DIAGNOSTICS — run directly to simulate full pack discharge
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data.load_profiles import get_profile

    print("=" * 55)
    print("Pack Model — 8S8P 1C Discharge Diagnostics")
    print("=" * 55)

    pack    = PackModel(soc_init=1.0, soc_spread=0.03)
    profile = get_profile("constant_1c")

    times, pack_voltages, pack_socs, delta_vs, max_temps = [], [], [], [], []

    t = 0
    while pack.get_cell_socs().mean() > 0.05 and t < 7200:
        current = profile.get_current(t)
        state   = pack.step(pack_current=current)

        times.append(t)
        pack_voltages.append(state.pack_voltage)
        pack_socs.append(state.pack_soc * 100)
        delta_vs.append(state.delta_voltage * 1000)   # mV
        max_temps.append(state.max_temp_c)
        t += SIM_TIMESTEP_S

    print(f"  Cells          : {pack.n_cells} ({pack.n_series}S{pack.n_parallel}P)")
    print(f"  Final Pack SOC : {pack_socs[-1]:.1f}%")
    print(f"  Final Pack V   : {pack_voltages[-1]:.2f} V")
    print(f"  Max ΔV         : {max(delta_vs):.1f} mV")
    print(f"  Max Temp       : {max(max_temps):.2f} °C")
    print(f"  Energy out     : {pack.step(0).pack_energy_wh:.2f} Wh")
    print(pack)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    fig.suptitle("8S8P Pack — 1C Discharge", fontweight="bold")

    axes[0].plot(times, pack_voltages, color="steelblue", linewidth=1.5)
    axes[0].set_ylabel("Pack Voltage [V]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, pack_socs, color="forestgreen", linewidth=1.5)
    axes[1].set_ylabel("Pack SOC [%]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, delta_vs, color="darkorange", linewidth=1.5)
    axes[2].set_ylabel("Cell ΔV [mV]")
    axes[2].axhline(20, color="red", linestyle="--", linewidth=0.8, label="Balance threshold")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(times, max_temps, color="tomato", linewidth=1.5)
    axes[3].set_ylabel("Max Temp [°C]")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("logs/pack_model_discharge.png", dpi=120)
    plt.show()