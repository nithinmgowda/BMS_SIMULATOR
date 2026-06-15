# =============================================================================
# sim_loop.py — Simulation Engine & Module Orchestrator
# =============================================================================
# The central tick engine. Every simulation second:
#
#   1. Get demanded current from load profile
#   2. Apply fault-based current limits
#   3. Compute & apply balancing currents
#   4. Step pack physics (ECM, thermal)
#   5. Update SOC estimates (CC, OCV, EKF)
#   6. Update SOH estimates (capacity fade, resistance growth)
#   7. Check faults (OV, UV, OC, OT, IMB, EOL)
#   8. Log state
#   9. Push state to GUI callback (if registered)
#
# SimLoop is designed to run in a background thread so the GUI
# stays responsive. State is shared via thread-safe callbacks.
#
# Usage:
#   sim = SimLoop(profile="udds", balancing_mode="passive")
#   sim.register_callback(my_gui_update_fn)
#   sim.start()        # non-blocking, runs in thread
#   sim.pause()
#   sim.resume()
#   sim.stop()
# =============================================================================
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict

from core.pack_model      import PackModel, PackState
from core.soc_estimator   import PackSOCEstimator
from core.soh_estimator   import PackSOHEstimator, PackSOHState
from core.balancer        import PackBalancer, BalancerState
from core.fault_detector  import FaultDetector, FaultState, FaultLevel
from data.load_profiles   import get_profile, LoadProfile
from config import (
    SIM_TIMESTEP_S,
    SIM_DEFAULT_DURATION_S,
    SIM_SPEED_MULTIPLIER,
    AMBIENT_TEMP_C,
    BALANCING_MODE,
    LOG_ENABLED,
    LOG_INTERVAL_S,
    LOG_OUTPUT_PATH,
)


# =============================================================================
# FULL SIMULATION STATE — one snapshot per tick
# =============================================================================

@dataclass
class SimState:
    """
    Complete simulator state snapshot pushed to GUI every tick.
    Contains pack, SOC, SOH, balancer, and fault states.
    """
    tick            : int          = 0
    time_s          : float        = 0.0
    pack            : Optional[PackState]    = None
    soh             : Optional[PackSOHState] = None
    balancer        : Optional[BalancerState]= None
    fault           : Optional[FaultState]   = None

    # Per-cell arrays (length 64) for GUI panels
    cell_voltages   : List[float]  = field(default_factory=list)
    cell_socs_ekf   : List[float]  = field(default_factory=list)
    cell_socs_cc    : List[float]  = field(default_factory=list)
    cell_temps      : List[float]  = field(default_factory=list)
    cell_sohs       : List[float]  = field(default_factory=list)
    cell_balancing  : List[bool]   = field(default_factory=list)

    # Scalar summary for gauges
    pack_soc_ekf    : float        = 0.0
    pack_soh        : float        = 1.0
    pack_voltage    : float        = 0.0
    pack_current    : float        = 0.0
    pack_power_w    : float        = 0.0
    pack_energy_wh  : float        = 0.0
    max_temp_c      : float        = 0.0
    delta_v_mv      : float        = 0.0

    # Status
    is_running      : bool         = False
    fault_summary   : str          = "NORMAL"
    profile_name    : str          = ""


# =============================================================================
# SIMULATION LOOP
# =============================================================================

class SimLoop:
    """
    Main BMS simulation engine.

    Orchestrates all modules in the correct tick order and runs
    in a background thread to keep GUI responsive.

    Args:
        profile        : Load profile name (see data/load_profiles.py)
        balancing_mode : "passive" or "active"
        soc_init       : Initial pack SOC [0.0 → 1.0]
        soc_spread     : Cell-to-cell SOC imbalance at init [0.0 → 0.1]
        ambient_temp   : Ambient temperature [°C]
        duration_s     : Simulation duration [seconds] (0 = run forever)
        speed          : Simulation speed multiplier (1 = real-time, 10 = 10x)
    """

    def __init__(
        self,
        profile        : str   = "constant_1c",
        balancing_mode : str   = BALANCING_MODE,
        soc_init       : float = 1.0,
        soc_spread     : float = 0.03,
        ambient_temp   : float = AMBIENT_TEMP_C,
        duration_s     : int   = SIM_DEFAULT_DURATION_S,
        speed          : float = SIM_SPEED_MULTIPLIER,
    ):
        self.profile_name    = profile
        self.balancing_mode  = balancing_mode
        self.soc_init        = soc_init
        self.soc_spread      = soc_spread
        self.ambient_temp    = ambient_temp
        self.duration_s      = duration_s
        self.speed           = max(speed, 0.1)

        # Internal state
        self._running        = False
        self._paused         = False
        self._thread         = None
        self._tick           = 0
        self._time_s         = 0.0
        self._callbacks      : List[Callable] = []
        self._logger         = None

        # Modules (initialised on start)
        self.pack            = None
        self.soc_estimator   = None
        self.soh_estimator   = None
        self.balancer        = None
        self.fault_detector  = None
        self.load_profile    = None

        # Last known state (thread-safe read by GUI)
        self._state_lock     = threading.Lock()
        self._last_state     : Optional[SimState] = None

    # -------------------------------------------------------------------------
    # PUBLIC CONTROL API
    # -------------------------------------------------------------------------

    def register_callback(self, fn: Callable[[SimState], None]):
        """
        Register a function called every tick with the latest SimState.
        Used by GUI dashboard to update displays.

        Args:
            fn : Callable accepting SimState
        """
        self._callbacks.append(fn)

    def start(self):
        """Initialise all modules and start simulation in background thread."""
        if self._running:
            return
        self._init_modules()
        self._running = True
        self._paused  = False
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[SimLoop] Started | Profile: {self.profile_name} | "
              f"Mode: {self.balancing_mode} | Speed: {self.speed}x")

    def pause(self):
        """Pause simulation (resumable)."""
        self._paused = True
        print("[SimLoop] Paused")

    def resume(self):
        """Resume paused simulation."""
        self._paused = False
        print("[SimLoop] Resumed")

    def stop(self):
        """Stop simulation and save logs."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._logger:
            self._logger.close()
        print(f"[SimLoop] Stopped at t={self._time_s:.0f}s | Tick={self._tick}")

    def get_state(self) -> Optional[SimState]:
        """Thread-safe read of latest simulation state."""
        with self._state_lock:
            return self._last_state

    def set_profile(self, profile_name: str):
        """Hot-swap load profile mid-simulation."""
        self.load_profile = get_profile(profile_name)
        self.profile_name = profile_name
        print(f"[SimLoop] Profile switched to: {profile_name}")

    def set_balancing_mode(self, mode: str):
        """Hot-swap balancing mode mid-simulation."""
        if self.balancer:
            self.balancer.switch_mode(mode)
            self.balancing_mode = mode
            print(f"[SimLoop] Balancing mode switched to: {mode}")

    def set_ambient_temp(self, temp_c: float):
        """Update ambient temperature mid-simulation."""
        self.ambient_temp = temp_c
        if self.pack:
            self.pack.set_ambient_temperature(temp_c)

    def reset(self):
        """Full reset — restart from initial conditions."""
        was_running = self._running
        self.stop()
        self._tick   = 0
        self._time_s = 0.0
        if was_running:
            self.start()

    # -------------------------------------------------------------------------
    # MODULE INITIALISATION
    # -------------------------------------------------------------------------

    def _init_modules(self):
        """Build all simulation modules fresh."""
        print("[SimLoop] Initialising modules...")

        self.pack = PackModel(
            soc_init     = self.soc_init,
            soc_spread   = self.soc_spread,
            ambient_temp = self.ambient_temp,
        )

        self.soc_estimator = PackSOCEstimator(pack_model=self.pack)

        self.soh_estimator = PackSOHEstimator(
            pack_model    = self.pack,
            soc_estimator = self.soc_estimator,
        )

        self.balancer = PackBalancer(mode=self.balancing_mode)

        self.fault_detector = FaultDetector(debounce_ticks=3)

        self.load_profile = get_profile(self.profile_name)

        if LOG_ENABLED:
            self._init_logger()

        print(f"[SimLoop] Pack: {self.pack.n_series}S{self.pack.n_parallel}P "
              f"| {self.pack.n_cells} cells | "
              f"Capacity: {self.pack.n_parallel * 2.0:.0f} Ah")

    # -------------------------------------------------------------------------
    # MAIN TICK LOOP
    # -------------------------------------------------------------------------

    def _run(self):
        """Background simulation thread."""
        dt_real = SIM_TIMESTEP_S / self.speed   # Real-time sleep per tick

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            # Check duration limit
            if self.duration_s > 0 and self._time_s >= self.duration_s:
                print(f"[SimLoop] Duration reached ({self.duration_s}s). Stopping.")
                self._running = False
                break

            # ── TICK ────────────────────────────────────────────────────
            tick_start = time.perf_counter()
            state = self._tick_once()
            tick_elapsed = time.perf_counter() - tick_start

            # Thread-safe state update
            with self._state_lock:
                self._last_state = state

            # Fire GUI callbacks
            for cb in self._callbacks:
                try:
                    cb(state)
                except Exception as e:
                    print(f"[SimLoop] Callback error: {e}")

            # Log
            if LOG_ENABLED and self._tick % max(1, int(LOG_INTERVAL_S)) == 0:
                self._log(state)

            # Enforce real-time pacing
            sleep_time = dt_real - tick_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            self._tick   += 1
            self._time_s += SIM_TIMESTEP_S

    # -------------------------------------------------------------------------
    # SINGLE TICK — ordered module calls
    # -------------------------------------------------------------------------

    def _tick_once(self) -> SimState:
        """
        Execute one complete simulation tick.
        Order is critical — do not reorder steps.
        """

        # ── STEP 1: Get demanded current from profile ────────────────────
        demanded_current = self.load_profile.get_current(self._time_s)

        # ── STEP 2: Apply fault-based current limits ─────────────────────
        last_fault = self.fault_detector.get_fault_log()[-1] \
                     if self.fault_detector.get_fault_log() else None

        effective_current = self._apply_fault_limits(
            demanded_current, last_fault
        )

        # ── STEP 3: Compute balancing currents ───────────────────────────
        #    Need a peek at last cell states for balancer input
        #    Use pack's current cell SOCs/voltages directly
        if self._tick > 0 and self._last_state and self._last_state.pack:
            last_cell_states = self._last_state.pack.cell_states
            bal_state = self.balancer.compute(last_cell_states, effective_current)
            self.balancer.apply_to_pack(self.pack, bal_state)
        else:
            from core.balancer import BalancerState
            bal_state = BalancerState()

        # ── STEP 4: Step pack physics ────────────────────────────────────
        pack_state = self.pack.step(pack_current=effective_current)

        # ── STEP 5: Update SOC estimates ─────────────────────────────────
        soc_estimates = self.soc_estimator.update(pack_state.cell_states)

        # ── STEP 6: Update SOH estimates ─────────────────────────────────
        soh_state = self.soh_estimator.tick(pack_state.cell_states)

        # ── STEP 7: Check faults ─────────────────────────────────────────
        fault_state = self.fault_detector.check(
            cell_states  = pack_state.cell_states,
            pack_current = effective_current,
            soh_states   = soh_state.cell_soh_states,
            ambient_temp = self.ambient_temp,
        )

        # ── STEP 8: Assemble SimState ────────────────────────────────────
        cell_voltages  = [cs.voltage     for cs in pack_state.cell_states]
        cell_temps     = [cs.temperature for cs in pack_state.cell_states]
        cell_balancing = [cs.is_balancing for cs in pack_state.cell_states]
        cell_socs_ekf  = [e.soc_ekf for e in soc_estimates]
        cell_socs_cc   = [e.soc_cc  for e in soc_estimates]
        cell_sohs      = [s.soh     for s in soh_state.cell_soh_states]

        return SimState(
            tick           = self._tick,
            time_s         = self._time_s,
            pack           = pack_state,
            soh            = soh_state,
            balancer       = bal_state,
            fault          = fault_state,
            cell_voltages  = cell_voltages,
            cell_socs_ekf  = cell_socs_ekf,
            cell_socs_cc   = cell_socs_cc,
            cell_temps     = cell_temps,
            cell_sohs      = cell_sohs,
            cell_balancing = cell_balancing,
            pack_soc_ekf   = float(np.mean(cell_socs_ekf)),
            pack_soh       = soh_state.pack_soh,
            pack_voltage   = pack_state.pack_voltage,
            pack_current   = effective_current,
            pack_power_w   = pack_state.pack_power_w,
            pack_energy_wh = pack_state.pack_energy_wh,
            max_temp_c     = pack_state.max_temp_c,
            delta_v_mv     = pack_state.delta_voltage * 1000,
            is_running     = self._running,
            fault_summary  = self.fault_detector.get_active_summary(fault_state),
            profile_name   = self.profile_name,
        )

    # -------------------------------------------------------------------------
    # FAULT-BASED CURRENT LIMITING
    # -------------------------------------------------------------------------

    def _apply_fault_limits(
        self,
        demanded: float,
        fault   : Optional[FaultState],
    ) -> float:
        """
        Clamp demanded current based on active fault state.

        CRITICAL → 0A (shutdown)
        FAULT    → 50% current reduction
        WARNING  → no change (monitor only)
        """
        if fault is None:
            return demanded

        if fault.request_shutdown:
            return 0.0

        if fault.disable_charging and demanded < 0:
            return 0.0

        if fault.disable_discharging and demanded > 0:
            return 0.0

        if fault.request_current_reduce:
            return demanded * 0.5

        return demanded

    # -------------------------------------------------------------------------
    # LOGGER
    # -------------------------------------------------------------------------

    def _init_logger(self):
        """Initialise CSV log file."""
        import os, csv
        os.makedirs(os.path.dirname(LOG_OUTPUT_PATH), exist_ok=True)
        self._log_file = open(LOG_OUTPUT_PATH, "w", newline="")
        self._csv_writer = csv.writer(self._log_file)
        self._csv_writer.writerow([
            "tick", "time_s", "pack_voltage", "pack_current",
            "pack_soc_ekf", "pack_soh", "max_temp_c",
            "delta_v_mv", "fault_summary", "pack_energy_wh"
        ])

    def _log(self, state: SimState):
        """Write one row to CSV log."""
        try:
            self._csv_writer.writerow([
                state.tick,
                f"{state.time_s:.1f}",
                f"{state.pack_voltage:.4f}",
                f"{state.pack_current:.4f}",
                f"{state.pack_soc_ekf:.6f}",
                f"{state.pack_soh:.6f}",
                f"{state.max_temp_c:.3f}",
                f"{state.delta_v_mv:.3f}",
                state.fault_summary,
                f"{state.pack_energy_wh:.4f}",
            ])
        except Exception:
            pass

    def _close_logger(self):
        if hasattr(self, "_log_file"):
            self._log_file.close()

    # -------------------------------------------------------------------------
    # REPR
    # -------------------------------------------------------------------------

    def __repr__(self):
        return (
            f"SimLoop(profile={self.profile_name}, "
            f"mode={self.balancing_mode}, "
            f"t={self._time_s:.0f}s, "
            f"tick={self._tick}, "
            f"running={self._running})"
        )


# =============================================================================
# DIAGNOSTICS — run a headless simulation and print live state
# =============================================================================

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("SimLoop — Headless Integration Test (30s real-time)")
    print("=" * 60)

    results = []

    def capture(state: SimState):
        results.append(state)
        if state.tick % 60 == 0:
            print(
                f"  t={state.time_s:>6.0f}s | "
                f"V={state.pack_voltage:>6.2f}V | "
                f"SOC={state.pack_soc_ekf*100:>5.1f}% | "
                f"SOH={state.pack_soh*100:>5.1f}% | "
                f"T={state.max_temp_c:>5.1f}°C | "
                f"ΔV={state.delta_v_mv:>5.1f}mV | "
                f"{state.fault_summary}"
            )

    sim = SimLoop(
        profile        = "constant_1c",
        balancing_mode = "passive",
        soc_init       = 1.0,
        soc_spread     = 0.05,
        duration_s     = 600,     # 10 min sim
        speed          = 20.0,    # 20x real-time → 30s wall clock
    )
    sim.register_callback(capture)
    sim.start()

    # Wait for completion
    while sim._running:
        time.sleep(0.5)

    print(f"\n  Simulation complete.")
    print(f"  Total ticks    : {len(results)}")
    print(f"  Final SOC      : {results[-1].pack_soc_ekf*100:.2f}%")
    print(f"  Final Voltage  : {results[-1].pack_voltage:.3f} V")
    print(f"  Energy out     : {results[-1].pack_energy_wh:.2f} Wh")
    print(f"  Max Temp       : {results[-1].max_temp_c:.2f} °C")
    print(f"  Max ΔV         : {max(s.delta_v_mv for s in results):.2f} mV")