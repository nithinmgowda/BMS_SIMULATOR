# =============================================================================
# core/balancer.py — Cell Balancing Logic
# =============================================================================
# Implements two balancing strategies for the 8S8P pack:
#
#   1. PASSIVE BALANCING
#      Bleeds excess charge from high-SOC cells through a resistor.
#      Simple, reliable, cheap — wastes energy as heat.
#
#      Logic per series group:
#        - Find max SOC cell in group
#        - Any cell with SOC > (max_SOC - threshold) is bypassed (bled)
#        - Bleed current = V_cell / R_bleed
#
#   2. ACTIVE BALANCING
#      Transfers charge from high-SOC to low-SOC cells via DC-DC converter.
#      Efficient, complex — preserves energy.
#
#      Logic per series group:
#        - Find highest and lowest SOC cells
#        - If ΔV > threshold, transfer current from high → low
#        - Transfer efficiency accounts for converter losses
#
#   Balancing operates per series group (not across groups).
#   Only activates when pack is NOT in fault state.
#   Balancing currents are written to cells via apply_balance_current()
#   and consumed by PackModel.step() each tick.
#
# =============================================================================
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from config import (
    NUM_CELLS_SERIES,
    NUM_CELLS_PARALLEL,
    BALANCING_MODE,
    PASSIVE_BALANCE_RESISTOR,
    PASSIVE_BALANCE_THRESHOLD_V,
    ACTIVE_BALANCE_CURRENT_A,
    ACTIVE_BALANCE_THRESHOLD_V,
    ACTIVE_BALANCE_EFFICIENCY,
    FAULT_IMBALANCE_V,
)


# =============================================================================
# BALANCER STATE — snapshot each tick
# =============================================================================

@dataclass
class BalancerState:
    """Balancing activity snapshot for one simulation tick."""
    mode              : str   = "passive"
    is_balancing      : bool  = False         # Any cell being balanced
    cells_balancing   : List[int] = field(default_factory=list)  # cell_ids active
    total_bleed_power : float = 0.0           # Total power dissipated [W] (passive)
    total_transferred : float = 0.0           # Total charge transferred [Ah] (active)
    max_delta_v       : float = 0.0           # Worst-case cell voltage spread [V]
    max_delta_soc     : float = 0.0           # Worst-case SOC spread [0→1]
    balance_currents  : Dict[int, float] = field(default_factory=dict)  # cell_id → A


# =============================================================================
# PASSIVE BALANCER
# =============================================================================

class PassiveBalancer:
    """
    Resistive (passive) cell balancing.

    Within each series group of 8 parallel cells:
      - Identify the cell(s) with highest voltage
      - Apply a bleed current to those cells only
      - Bleed current: I_bleed = V_cell / R_bleed
      - Bleed resistance: from config (PASSIVE_BALANCE_RESISTOR)

    Energy is dissipated as heat — factor this into thermal model.
    Balancing only runs when:
      1. Pack is at rest or charging (not discharging — no point)
      2. ΔV across any series group > PASSIVE_BALANCE_THRESHOLD_V
    """

    def __init__(self):
        self.r_bleed    = PASSIVE_BALANCE_RESISTOR       # [Ω]
        self.threshold  = PASSIVE_BALANCE_THRESHOLD_V    # [V]

    def compute(self, cell_states, pack_current: float) -> Dict[int, float]:
        """
        Compute passive balance currents for all cells this tick.

        Args:
            cell_states  : List[CellState] from PackModel
            pack_current : Total pack current [A] (+ discharge, - charge)

        Returns:
            Dict mapping cell_id → balance current [A]
            (positive = bleed discharge added on top of load current)
        """
        balance_currents: Dict[int, float] = {}

        # Only balance during charging or rest (not during discharge)
        if pack_current > 0.5:
            return balance_currents

        n_s = NUM_CELLS_SERIES
        n_p = NUM_CELLS_PARALLEL

        for s in range(n_s):
            # Get all cells in this series group
            group_indices = [s * n_p + p for p in range(n_p)]
            group_states  = [cell_states[i] for i in group_indices]

            voltages = np.array([cs.voltage for cs in group_states])
            max_v    = float(np.max(voltages))

            for local_p, cs in enumerate(group_states):
                cell_id = group_indices[local_p]
                delta_v = max_v - cs.voltage

                # Cells NOT at the peak get bled if delta is within threshold
                # i.e., bleed the HIGH cells (those close to max)
                if (max_v - cs.voltage) <= self.threshold and \
                   cs.voltage >= (max_v - self.threshold):
                    # Bleed current through bypass resistor
                    i_bleed = cs.voltage / self.r_bleed
                    balance_currents[cell_id] = i_bleed
                else:
                    balance_currents[cell_id] = 0.0

        return balance_currents

    def compute_bleed_power(self, balance_currents: Dict[int, float],
                            cell_states) -> float:
        """Total power dissipated in bleed resistors [W]."""
        total = 0.0
        for cs in cell_states:
            i_bleed = balance_currents.get(cs.cell_id, 0.0)
            total  += (i_bleed ** 2) * self.r_bleed
        return total


# =============================================================================
# ACTIVE BALANCER
# =============================================================================

class ActiveBalancer:
    """
    Charge-shuttling (active) cell balancing.

    Within each series group:
      - Find highest SOC cell (donor) and lowest SOC cell (recipient)
      - If ΔV > ACTIVE_BALANCE_THRESHOLD_V, transfer charge
      - Donor gets extra discharge current: +I_transfer
      - Recipient gets charge current:      -I_transfer × efficiency
      - Efficiency < 1.0 accounts for DC-DC converter losses

    Active balancing works during both charge and discharge.
    Multiple donor-recipient pairs can operate simultaneously.
    """

    def __init__(self):
        self.i_transfer  = ACTIVE_BALANCE_CURRENT_A    # Transfer current [A]
        self.efficiency  = ACTIVE_BALANCE_EFFICIENCY   # DC-DC efficiency
        self.threshold   = ACTIVE_BALANCE_THRESHOLD_V  # Trigger threshold [V]

    def compute(self, cell_states) -> Dict[int, float]:
        """
        Compute active balance currents for all cells this tick.

        Args:
            cell_states : List[CellState] from PackModel

        Returns:
            Dict mapping cell_id → balance current [A]
            Positive = extra discharge (donor)
            Negative = extra charge   (recipient)
        """
        balance_currents: Dict[int, float] = {
            cs.cell_id: 0.0 for cs in cell_states
        }

        n_s = NUM_CELLS_SERIES
        n_p = NUM_CELLS_PARALLEL

        for s in range(n_s):
            group_indices = [s * n_p + p for p in range(n_p)]
            group_states  = [cell_states[i] for i in group_indices]

            voltages = np.array([cs.voltage for cs in group_states])
            delta_v  = float(np.max(voltages) - np.min(voltages))

            if delta_v < self.threshold:
                continue    # Within tolerance — no balancing needed

            # Identify donors (high voltage) and recipients (low voltage)
            mean_v = float(np.mean(voltages))

            donors     = [(group_states[p], voltages[p])
                          for p in range(n_p) if voltages[p] > mean_v + self.threshold / 2]
            recipients = [(group_states[p], voltages[p])
                          for p in range(n_p) if voltages[p] < mean_v - self.threshold / 2]

            if not donors or not recipients:
                continue

            # Sort: highest donor first, lowest recipient first
            donors.sort(key=lambda x: -x[1])
            recipients.sort(key=lambda x: x[1])

            # Pair donors and recipients (round-robin for multiple pairs)
            n_pairs = min(len(donors), len(recipients))
            for k in range(n_pairs):
                donor_state    = donors[k][0]
                recipient_state= recipients[k][0]

                # Apply transfer currents
                balance_currents[donor_state.cell_id]     += self.i_transfer
                balance_currents[recipient_state.cell_id] -= (
                    self.i_transfer * self.efficiency
                )

        return balance_currents

    def compute_transferred_ah(self, balance_currents: Dict[int, float]) -> float:
        """Total charge transferred this tick [Ah]."""
        positive = sum(v for v in balance_currents.values() if v > 0)
        return positive / 3600.0     # [Ah per tick]


# =============================================================================
# PACK BALANCER — top-level, mode-selectable
# =============================================================================

class PackBalancer:
    """
    Top-level balancer for the full 8S8P pack.
    Selects between passive and active mode from config.
    Used by sim_loop.py — call compute() each tick before PackModel.step().

    Args:
        mode : "passive" or "active" (overrides config if provided)
    """

    def __init__(self, mode: str = BALANCING_MODE):
        self.mode    = mode.lower()
        self.passive = PassiveBalancer()
        self.active  = ActiveBalancer()

        if self.mode not in ("passive", "active"):
            raise ValueError(f"Unknown balancing mode '{mode}'. Use 'passive' or 'active'.")

    def compute(self, cell_states, pack_current: float) -> BalancerState:
        """
        Compute and apply balance currents to all cells this tick.

        Balance currents are written directly to each cell via
        apply_balance_current() so PackModel.step() picks them up.

        Args:
            cell_states  : List[CellState] from last PackModel.step()
            pack_current : Total pack load current [A]

        Returns:
            BalancerState snapshot
        """
        # Compute voltage and SOC spreads (for diagnostics/GUI)
        voltages = np.array([cs.voltage for cs in cell_states])
        socs     = np.array([cs.soc     for cs in cell_states])
        max_dv   = float(np.max(voltages) - np.min(voltages))
        max_dsoc = float(np.max(socs)     - np.min(socs))

        # Select algorithm
        if self.mode == "passive":
            balance_currents = self.passive.compute(cell_states, pack_current)
            bleed_power      = self.passive.compute_bleed_power(
                balance_currents, cell_states
            )
            transferred_ah = 0.0

        else:   # active
            balance_currents = self.active.compute(cell_states)
            bleed_power      = 0.0
            transferred_ah   = self.active.compute_transferred_ah(balance_currents)

        # Write balance currents to each cell model
        cells_active: List[int] = []
        for cs in cell_states:
            i_bal = balance_currents.get(cs.cell_id, 0.0)
            # Access cell via pack — sim_loop passes pack_model reference
            # Balance current is stored on the cell object, read by pack.step()
            if abs(i_bal) > 1e-6:
                cells_active.append(cs.cell_id)

        return BalancerState(
            mode              = self.mode,
            is_balancing      = len(cells_active) > 0,
            cells_balancing   = cells_active,
            total_bleed_power = bleed_power,
            total_transferred = transferred_ah,
            max_delta_v       = max_dv,
            max_delta_soc     = max_dsoc,
            balance_currents  = balance_currents,
        )

    def apply_to_pack(self, pack_model, balance_state: BalancerState):
        """
        Push computed balance currents into cell objects in the pack.
        Must be called BEFORE pack_model.step() each tick.

        Args:
            pack_model    : PackModel instance
            balance_state : BalancerState from compute()
        """
        for cell in pack_model.all_cells:
            i_bal = balance_state.balance_currents.get(cell.cell_id, 0.0)
            cell.apply_balance_current(i_bal)

    def switch_mode(self, mode: str):
        """Dynamically switch balancing mode (for GUI control)."""
        if mode.lower() not in ("passive", "active"):
            raise ValueError(f"Invalid mode '{mode}'")
        self.mode = mode.lower()

    def __repr__(self):
        return f"PackBalancer(mode={self.mode})"


# =============================================================================
# DIAGNOSTICS — simulate balancing on an imbalanced pack
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from core.pack_model import PackModel
    from data.load_profiles import get_profile

    print("=" * 55)
    print("Balancer Diagnostics — Passive vs Active")
    print("=" * 55)

    def run_sim(mode: str, steps: int = 1800):
        pack     = PackModel(soc_init=0.8, soc_spread=0.08)  # High imbalance
        balancer = PackBalancer(mode=mode)
        profile  = get_profile("constant_half_c")

        delta_vs, bleed_powers = [], []

        for t in range(steps):
            current   = profile.get_current(t)
            # Get last state for balancer input
            last_state = pack.step(pack_current=0)   # Peek at state
            bal_state  = balancer.compute(last_state.cell_states, current)
            balancer.apply_to_pack(pack, bal_state)
            state      = pack.step(pack_current=current)

            delta_vs.append(bal_state.max_delta_v * 1000)      # mV
            bleed_powers.append(bal_state.total_bleed_power)

        return delta_vs, bleed_powers

    print("  Running passive balancing sim...")
    dv_passive, pw_passive = run_sim("passive")
    print("  Running active balancing sim...")
    dv_active,  pw_active  = run_sim("active")

    t_axis = list(range(len(dv_passive)))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle("Cell Balancing — Passive vs Active (High Initial Imbalance)", fontweight="bold")

    axes[0].plot(t_axis, dv_passive, "b-",  linewidth=1.4, label="Passive")
    axes[0].plot(t_axis, dv_active,  "r--", linewidth=1.4, label="Active")
    axes[0].axhline(PASSIVE_BALANCE_THRESHOLD_V * 1000, color="gray",
                    linestyle=":", linewidth=1.0, label="Passive threshold")
    axes[0].axhline(ACTIVE_BALANCE_THRESHOLD_V * 1000, color="orange",
                    linestyle=":", linewidth=1.0, label="Active threshold")
    axes[0].set_ylabel("Max Cell ΔV [mV]")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_axis, pw_passive, "b-", linewidth=1.4, label="Passive bleed power [W]")
    axes[1].set_ylabel("Bleed Power [W]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("logs/balancer_comparison.png", dpi=120)
    plt.show()

    print(f"\n  Passive: Final ΔV = {dv_passive[-1]:.1f} mV")
    print(f"  Active:  Final ΔV = {dv_active[-1]:.1f} mV")