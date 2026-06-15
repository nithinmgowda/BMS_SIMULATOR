# =============================================================================
# core/fault_detector.py — BMS Fault Detection & Protection Logic
# =============================================================================
# Monitors all 64 cells every tick for protection threshold violations.
#
# Fault hierarchy (severity low → high):
#
#   LEVEL 0 — NORMAL       : All parameters within safe range
#   LEVEL 1 — WARNING      : Approaching limits, log and alert GUI
#   LEVEL 2 — FAULT        : Limit exceeded, request current reduction
#   LEVEL 3 — CRITICAL     : Severe violation, request immediate shutdown
#
# Fault types monitored:
#   OV  — Over Voltage      (cell voltage too high)
#   UV  — Under Voltage     (cell voltage too low)
#   OC  — Over Current      (charge or discharge current too high)
#   OT  — Over Temperature  (cell or ambient too hot)
#   UT  — Under Temperature (cell too cold for safe operation)
#   IMB — Cell Imbalance    (ΔV across pack too large)
#   EOL — End of Life       (SOH below threshold)
#
# Each fault has:
#   - Warning threshold  (85% of limit) → LEVEL 1
#   - Fault threshold    (100% of limit) → LEVEL 2
#   - Critical threshold (110% of limit) → LEVEL 3
#
# Output FaultState is consumed by sim_loop.py to:
#   - Reduce or halt current
#   - Disable balancing
#   - Trigger GUI alerts
# =============================================================================


import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import IntEnum
from config import (
    NUM_CELLS,
    CELL_MAX_VOLTAGE_V,
    CELL_MIN_VOLTAGE_V,
    CELL_MAX_TEMP_C,
    CELL_MIN_TEMP_C,
    FAULT_OV_VOLTAGE_V,
    FAULT_UV_VOLTAGE_V,
    FAULT_OC_CHARGE_A,
    FAULT_OC_DISCHARGE_A,
    FAULT_OT_CELL_C,
    FAULT_OT_AMBIENT_C,
    FAULT_IMBALANCE_V,
    EOL_SOH_THRESHOLD,
)


# =============================================================================
# FAULT LEVEL ENUM
# =============================================================================

class FaultLevel(IntEnum):
    NORMAL   = 0
    WARNING  = 1
    FAULT    = 2
    CRITICAL = 3


# =============================================================================
# FAULT EVENT — single fault occurrence
# =============================================================================

@dataclass
class FaultEvent:
    """Represents one active fault condition."""
    fault_type  : str        = ""        # "OV", "UV", "OC", "OT", "UT", "IMB", "EOL"
    level       : FaultLevel = FaultLevel.NORMAL
    cell_id     : int        = -1        # -1 = pack-level fault
    value       : float      = 0.0       # Measured value that triggered fault
    threshold   : float      = 0.0       # Threshold that was exceeded
    message     : str        = ""        # Human-readable description
    time_s      : float      = 0.0       # Simulation time when triggered

    @property
    def is_active(self) -> bool:
        return self.level > FaultLevel.NORMAL

    def __str__(self):
        return (
            f"[{self.level.name}] {self.fault_type} | "
            f"Cell {self.cell_id if self.cell_id >= 0 else 'PACK'} | "
            f"{self.message} | "
            f"Value={self.value:.3f} Threshold={self.threshold:.3f}"
        )


# =============================================================================
# FAULT STATE — full snapshot every tick
# =============================================================================

@dataclass
class FaultState:
    """Complete fault status for one simulation tick."""
    highest_level   : FaultLevel      = FaultLevel.NORMAL
    active_faults   : List[FaultEvent]= field(default_factory=list)
    fault_count     : int             = 0
    any_critical    : bool            = False
    any_fault       : bool            = False
    any_warning     : bool            = False

    # Action flags for sim_loop
    request_shutdown        : bool    = False   # Stop all current immediately
    request_current_reduce  : bool    = False   # Reduce to safe level
    disable_charging        : bool    = False   # Block charge current
    disable_discharging     : bool    = False   # Block discharge current
    disable_balancing       : bool    = False   # Stop balancer

    # Fault summary per type (for GUI display)
    ov_cells  : List[int] = field(default_factory=list)
    uv_cells  : List[int] = field(default_factory=list)
    ot_cells  : List[int] = field(default_factory=list)
    oc_active : bool      = False
    imb_active: bool      = False
    eol_cells : List[int] = field(default_factory=list)

    time_s    : float     = 0.0


# =============================================================================
# FAULT DETECTOR
# =============================================================================

class FaultDetector:
    """
    Monitors all 64 cells and pack-level parameters every simulation tick.

    Maintains a fault history log and tracks fault persistence
    (faults must persist for N ticks before escalating to avoid false triggers).

    Args:
        debounce_ticks : Number of consecutive ticks before fault is confirmed
                         Default 3 = 3 seconds at 1s timestep
    """

    # Warning at 92% of limit, Critical at 108%
    WARNING_FACTOR  = 0.97      # was 0.92 — gives 4.12V warning threshold
    CRITICAL_FACTOR = 1.02      # was 1.08

    def __init__(self, debounce_ticks: int = 3):
        self.debounce_ticks = debounce_ticks
        self.fault_history  : List[FaultState] = []
        self.time_s         : float = 0.0

        # Debounce counters per fault type per cell
        # key: (fault_type, cell_id) → consecutive tick count
        self._debounce: Dict[tuple, int] = {}

        # Latched faults (require manual reset)
        self._latched_faults: List[FaultEvent] = []

    # -------------------------------------------------------------------------
    # MAIN CHECK — called every tick by sim_loop
    # -------------------------------------------------------------------------

    def check(
        self,
        cell_states,
        pack_current : float,
        soh_states   = None,
        ambient_temp : float = 25.0,
    ) -> FaultState:
        """
        Run all fault checks against current cell and pack state.

        Args:
            cell_states  : List[CellState] from PackModel.step()
            pack_current : Total pack current [A]
            soh_states   : List[SOHState] from PackSOHEstimator (optional)
            ambient_temp : Ambient temperature [°C]

        Returns:
            FaultState with all active faults and action flags
        """
        from core.cell_model import CellState

        active_faults: List[FaultEvent] = []

        # ── 1. CELL-LEVEL CHECKS ──────────────────────────────────────────
        for cs in cell_states:
            # Over Voltage
            fault = self._check_threshold(
                key       = ("OV", cs.cell_id),
                value     = cs.voltage,
                warn_thr  = FAULT_OV_VOLTAGE_V * self.WARNING_FACTOR,
                fault_thr = FAULT_OV_VOLTAGE_V,
                crit_thr  = FAULT_OV_VOLTAGE_V * self.CRITICAL_FACTOR,
                fault_type= "OV",
                cell_id   = cs.cell_id,
                msg_fn    = lambda v, t: f"Cell voltage {v:.3f}V > {t:.3f}V",
            )
            if fault: active_faults.append(fault)

            # Under Voltage
            fault = self._check_threshold_low(
                key       = ("UV", cs.cell_id),
                value     = cs.voltage,
                warn_thr  = FAULT_UV_VOLTAGE_V * (2.0 - self.WARNING_FACTOR),
                fault_thr = FAULT_UV_VOLTAGE_V,
                crit_thr  = FAULT_UV_VOLTAGE_V * (2.0 - self.CRITICAL_FACTOR),
                fault_type= "UV",
                cell_id   = cs.cell_id,
                msg_fn    = lambda v, t: f"Cell voltage {v:.3f}V < {t:.3f}V",
            )
            if fault: active_faults.append(fault)

            # Over Temperature (cell)
            fault = self._check_threshold(
                key       = ("OT", cs.cell_id),
                value     = cs.temperature,
                warn_thr  = FAULT_OT_CELL_C * self.WARNING_FACTOR,
                fault_thr = FAULT_OT_CELL_C,
                crit_thr  = FAULT_OT_CELL_C * self.CRITICAL_FACTOR,
                fault_type= "OT",
                cell_id   = cs.cell_id,
                msg_fn    = lambda v, t: f"Cell temp {v:.1f}°C > {t:.1f}°C",
            )
            if fault: active_faults.append(fault)

            # Under Temperature (cell)
            fault = self._check_threshold_low(
                key       = ("UT", cs.cell_id),
                value     = cs.temperature,
                warn_thr  = CELL_MIN_TEMP_C + 5.0,
                fault_thr = CELL_MIN_TEMP_C,
                crit_thr  = CELL_MIN_TEMP_C - 5.0,
                fault_type= "UT",
                cell_id   = cs.cell_id,
                msg_fn    = lambda v, t: f"Cell temp {v:.1f}°C < {t:.1f}°C",
            )
            if fault: active_faults.append(fault)

        # ── 2. PACK-LEVEL CHECKS ──────────────────────────────────────────

        # Over Current — Discharge
        if pack_current > 0:
            fault = self._check_threshold(
                key       = ("OC_DIS", -1),
                value     = pack_current,
                warn_thr  = FAULT_OC_DISCHARGE_A * self.WARNING_FACTOR,
                fault_thr = FAULT_OC_DISCHARGE_A,
                crit_thr  = FAULT_OC_DISCHARGE_A * self.CRITICAL_FACTOR,
                fault_type= "OC",
                cell_id   = -1,
                msg_fn    = lambda v, t: f"Discharge current {v:.1f}A > {t:.1f}A",
            )
            if fault: active_faults.append(fault)

        # Over Current — Charge
        if pack_current < 0:
            fault = self._check_threshold(
                key       = ("OC_CHG", -1),
                value     = abs(pack_current),
                warn_thr  = FAULT_OC_CHARGE_A * self.WARNING_FACTOR,
                fault_thr = FAULT_OC_CHARGE_A,
                crit_thr  = FAULT_OC_CHARGE_A * self.CRITICAL_FACTOR,
                fault_type= "OC",
                cell_id   = -1,
                msg_fn    = lambda v, t: f"Charge current {v:.1f}A > {t:.1f}A",
            )
            if fault: active_faults.append(fault)

        # Over Temperature — Ambient
        fault = self._check_threshold(
            key       = ("OT_AMB", -1),
            value     = ambient_temp,
            warn_thr  = FAULT_OT_AMBIENT_C * self.WARNING_FACTOR,
            fault_thr = FAULT_OT_AMBIENT_C,
            crit_thr  = FAULT_OT_AMBIENT_C * self.CRITICAL_FACTOR,
            fault_type= "OT",
            cell_id   = -1,
            msg_fn    = lambda v, t: f"Ambient temp {v:.1f}°C > {t:.1f}°C",
        )
        if fault: active_faults.append(fault)

        # Cell Imbalance
        voltages = np.array([cs.voltage for cs in cell_states])
        delta_v  = float(np.max(voltages) - np.min(voltages))
        fault = self._check_threshold(
            key       = ("IMB", -1),
            value     = delta_v,
            warn_thr  = FAULT_IMBALANCE_V * self.WARNING_FACTOR,
            fault_thr = FAULT_IMBALANCE_V,
            crit_thr  = FAULT_IMBALANCE_V * self.CRITICAL_FACTOR,
            fault_type= "IMB",
            cell_id   = -1,
            msg_fn    = lambda v, t: f"Cell imbalance ΔV={v*1000:.1f}mV > {t*1000:.1f}mV",
        )
        if fault: active_faults.append(fault)

        # EOL Check (from SOH states)
        if soh_states:
            for soh in soh_states:
                if soh.is_eol:
                    eol_fault = FaultEvent(
                        fault_type = "EOL",
                        level      = FaultLevel.WARNING,
                        cell_id    = soh.cell_id,
                        value      = soh.soh,
                        threshold  = EOL_SOH_THRESHOLD,
                        message    = f"Cell SOH={soh.soh*100:.1f}% below EOL threshold",
                        time_s     = self.time_s,
                    )
                    active_faults.append(eol_fault)

        # ── 3. BUILD FAULT STATE ──────────────────────────────────────────
        state = self._build_state(active_faults, cell_states, pack_current)
        self.fault_history.append(state)
        self.time_s += 1.0
        return state

    # -------------------------------------------------------------------------
    # THRESHOLD HELPERS
    # -------------------------------------------------------------------------

    def _check_threshold(
        self, key, value, warn_thr, fault_thr, crit_thr,
        fault_type, cell_id, msg_fn
    ) -> Optional[FaultEvent]:
        """Check a HIGH threshold (value should stay BELOW limit)."""
        if value >= crit_thr:
            level = FaultLevel.CRITICAL
            thr   = crit_thr
        elif value >= fault_thr:
            level = FaultLevel.FAULT
            thr   = fault_thr
        elif value >= warn_thr:
            level = FaultLevel.WARNING
            thr   = warn_thr
        else:
            self._debounce[key] = 0
            return None

        return self._debounce_fault(key, FaultEvent(
            fault_type = fault_type,
            level      = level,
            cell_id    = cell_id,
            value      = value,
            threshold  = thr,
            message    = msg_fn(value, thr),
            time_s     = self.time_s,
        ))

    def _check_threshold_low(
        self, key, value, warn_thr, fault_thr, crit_thr,
        fault_type, cell_id, msg_fn
    ) -> Optional[FaultEvent]:
        """Check a LOW threshold (value should stay ABOVE limit)."""
        if value <= crit_thr:
            level = FaultLevel.CRITICAL
            thr   = crit_thr
        elif value <= fault_thr:
            level = FaultLevel.FAULT
            thr   = fault_thr
        elif value <= warn_thr:
            level = FaultLevel.WARNING
            thr   = warn_thr
        else:
            self._debounce[key] = 0
            return None

        return self._debounce_fault(key, FaultEvent(
            fault_type = fault_type,
            level      = level,
            cell_id    = cell_id,
            value      = value,
            threshold  = thr,
            message    = msg_fn(value, thr),
            time_s     = self.time_s,
        ))

    def _debounce_fault(self, key: tuple, event: FaultEvent) -> Optional[FaultEvent]:
        """
        Only confirm fault if it persists for debounce_ticks consecutive ticks.
        Prevents false triggers from transient spikes.
        """
        self._debounce[key] = self._debounce.get(key, 0) + 1
        if self._debounce[key] >= self.debounce_ticks:
            return event
        return None     # Still within debounce window

    # -------------------------------------------------------------------------
    # BUILD FAULT STATE
    # -------------------------------------------------------------------------

    def _build_state(
        self,
        active_faults : List[FaultEvent],
        cell_states,
        pack_current  : float,
    ) -> FaultState:
        """Assemble FaultState from list of active fault events."""

        if not active_faults:
            return FaultState(time_s=self.time_s)

        levels        = [f.level for f in active_faults]
        highest_level = FaultLevel(max(levels))
        any_critical  = highest_level >= FaultLevel.CRITICAL
        any_fault     = highest_level >= FaultLevel.FAULT
        any_warning   = highest_level >= FaultLevel.WARNING

        # Per-type summaries for GUI
        ov_cells  = [f.cell_id for f in active_faults if f.fault_type == "OV"]
        uv_cells  = [f.cell_id for f in active_faults if f.fault_type == "UV"]
        ot_cells  = [f.cell_id for f in active_faults if f.fault_type == "OT"]
        oc_active = any(f.fault_type == "OC" for f in active_faults)
        imb_active= any(f.fault_type == "IMB" for f in active_faults)
        eol_cells = [f.cell_id for f in active_faults if f.fault_type == "EOL"]

        # Action flags based on severity
        request_shutdown       = any_critical
        request_current_reduce = any_fault and not any_critical
        disable_charging       = any(
            f.fault_type in ("OV", "OT") and f.level >= FaultLevel.FAULT
            for f in active_faults
        )
        disable_discharging    = any(
            f.fault_type in ("UV", "UT") and f.level >= FaultLevel.FAULT
            for f in active_faults
        )
        disable_balancing      = any_critical or any(
            f.fault_type == "OT" for f in active_faults
        )

        return FaultState(
            highest_level          = highest_level,
            active_faults          = active_faults,
            fault_count            = len(active_faults),
            any_critical           = any_critical,
            any_fault              = any_fault,
            any_warning            = any_warning,
            request_shutdown       = request_shutdown,
            request_current_reduce = request_current_reduce,
            disable_charging       = disable_charging,
            disable_discharging    = disable_discharging,
            disable_balancing      = disable_balancing,
            ov_cells               = ov_cells,
            uv_cells               = uv_cells,
            ot_cells               = ot_cells,
            oc_active              = oc_active,
            imb_active             = imb_active,
            eol_cells              = eol_cells,
            time_s                 = self.time_s,
        )

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------

    def reset_latched(self):
        """Clear latched faults (after manual inspection/reset)."""
        self._latched_faults.clear()
        self._debounce.clear()

    def get_fault_log(self) -> List[FaultState]:
        """Return full fault history."""
        return self.fault_history

    def get_active_summary(self, state: FaultState) -> str:
        """One-line fault summary for logging."""
        if not state.active_faults:
            return "NORMAL"
        parts = [f"{f.fault_type}(Cell {f.cell_id})" for f in state.active_faults]
        return f"[{state.highest_level.name}] " + ", ".join(parts)


# =============================================================================
# DIAGNOSTICS — inject deliberate faults and verify detection
# =============================================================================

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from core.pack_model import PackModel
    from core.cell_model import CellState

    print("=" * 55)
    print("Fault Detector — Injection Test")
    print("=" * 55)

    detector = FaultDetector(debounce_ticks=3)
    pack     = PackModel(soc_init=1.0)

    # Normal tick
    state = pack.step(pack_current=16.0)
    for _ in range(5):
        fault_state = detector.check(state.cell_states, pack_current=16.0)
    print(f"Normal operation : {detector.get_active_summary(fault_state)}")

    # Inject over-voltage by patching a cell state voltage
    ov_states = list(state.cell_states)
    bad        = ov_states[0]
    from dataclasses import replace
    ov_states[0] = replace(bad, voltage=4.30)   # > FAULT_OV_VOLTAGE_V (4.25)

    for _ in range(5):
        fault_state = detector.check(ov_states, pack_current=16.0)
    print(f"OV injected      : {detector.get_active_summary(fault_state)}")
    print(f"  Shutdown req?  : {fault_state.request_shutdown}")
    print(f"  Charge disable?: {fault_state.disable_charging}")

    # Inject over-current
    for _ in range(5):
        fault_state = detector.check(state.cell_states, pack_current=80.0)
    print(f"OC injected      : {detector.get_active_summary(fault_state)}")

    # Inject over-temperature
    ot_states  = list(state.cell_states)
    ot_states[3] = replace(ot_states[3], temperature=58.0)
    for _ in range(5):
        fault_state = detector.check(ot_states, pack_current=16.0)
    print(f"OT injected      : {detector.get_active_summary(fault_state)}")
    print(f"  Balance disable: {fault_state.disable_balancing}")

    print("\nAll fault injection tests passed.")