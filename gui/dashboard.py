# =============================================================================
# gui/dashboard.py — BMS Simulator Dashboard (v4 — micro-interactions, compact panels)
# =============================================================================
import sys
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QSlider,
    QTextEdit, QSizePolicy, QFrame, QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

from sim_loop import SimLoop, SimState
from config import (
    NUM_CELLS_SERIES, NUM_CELLS_PARALLEL, NUM_CELLS,
    CELL_MAX_VOLTAGE_V, CELL_MIN_VOLTAGE_V,
    PACK_NOMINAL_VOLTAGE_V,
)

# =============================================================================
# DESIGN TOKENS
# =============================================================================
BG_DARK     = "#080B10"
BG_PANEL    = "#0E1318"
BG_CARD     = "#151C26"
BG_CARD2    = "#1A2235"
ACCENT_CYAN = "#00D4FF"
ACCENT_GRN  = "#00FF9C"
ACCENT_YLW  = "#FFD60A"
ACCENT_RED  = "#FF4757"
ACCENT_PRP  = "#7B61FF"
TEXT_PRI    = "#E8EAF0"
TEXT_SEC    = "#6B7280"
TEXT_MID    = "#9CA3AF"
GRID_LINE   = "#1E2A3A"
BORDER      = "#252F40"

FONT_MONO = "Courier New"
FONT_UI   = "Segoe UI"
HISTORY_LEN = 300

CTRL_LABEL = f"color:{TEXT_SEC}; font-size:10px; font-family:{FONT_UI}; font-weight:700; letter-spacing:1.2px;"

COMBO_STYLE = f"""
    QComboBox {{
        background: {BG_CARD2};
        color: {TEXT_PRI};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 0px 14px;
        font-family: {FONT_UI};
        font-size: 13px;
        font-weight: 500;
        min-height: 44px;
        min-width: 160px;
    }}
    QComboBox:hover {{ border: 1.5px solid #3A4860; background: {BG_CARD}; }}
    QComboBox:focus {{ border: 1.5px solid {ACCENT_CYAN}; }}
    QComboBox::drop-down {{ border: none; width: 28px; }}
    QComboBox QAbstractItemView {{
        background: {BG_CARD};
        color: {TEXT_PRI};
        selection-background-color: {ACCENT_CYAN};
        selection-color: {BG_DARK};
        border: 1px solid {BORDER};
        font-size: 13px;
        padding: 4px;
        outline: none;
    }}
"""

# =============================================================================
# MICRO-INTERACTION BUTTON  (glow on hover via QPropertyAnimation)
# =============================================================================
class MicroButton(QPushButton):
    def __init__(self, text, variant="default", parent=None):
        super().__init__(text, parent)
        self._variant = variant
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(0)
        self._shadow.setOffset(0, 0)
        self.setGraphicsEffect(self._shadow)
        self._apply_style()

    def _apply_style(self):
        if self._variant == "primary":
            self._glow_color = QColor(ACCENT_CYAN)
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 #1A3A52, stop:1 #0D2035);
                    color: {ACCENT_CYAN};
                    border: 1.5px solid {ACCENT_CYAN};
                    border-radius: 8px;
                    padding: 0 28px;
                    font-family: {FONT_UI};
                    font-size: 13px;
                    font-weight: 700;
                    min-height: 44px;
                    min-width: 120px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 {ACCENT_CYAN}, stop:1 #0099BB);
                    color: {BG_DARK};
                }}
                QPushButton:pressed {{
                    background: #007A99;
                    color: {BG_DARK};
                    padding-top: 2px;
                }}
            """)
        elif self._variant == "danger":
            self._glow_color = QColor(ACCENT_RED)
            self.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                        stop:0 #2A1A1A, stop:1 #1A0D0D);
                    color: {ACCENT_RED};
                    border: 1.5px solid #3A1818;
                    border-radius: 8px;
                    padding: 0 28px;
                    font-family: {FONT_UI};
                    font-size: 13px;
                    font-weight: 600;
                    min-height: 44px;
                    min-width: 110px;
                }}
                QPushButton:hover {{
                    background: {ACCENT_RED};
                    color: {BG_DARK};
                    border-color: {ACCENT_RED};
                }}
                QPushButton:pressed {{
                    background: #CC3344;
                    color: {BG_DARK};
                    padding-top: 2px;
                }}
            """)
        else:
            self._glow_color = QColor(ACCENT_CYAN)
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {BG_CARD2};
                    color: {TEXT_MID};
                    border: 1px solid {BORDER};
                    border-radius: 8px;
                    padding: 0 28px;
                    font-family: {FONT_UI};
                    font-size: 13px;
                    font-weight: 600;
                    min-height: 44px;
                    min-width: 110px;
                }}
                QPushButton:hover {{
                    background: {BG_CARD};
                    color: {TEXT_PRI};
                    border-color: #3A4860;
                }}
                QPushButton:pressed {{
                    background: #0E1318;
                    color: {TEXT_PRI};
                    padding-top: 2px;
                }}
            """)

    def enterEvent(self, event):
        self._shadow.setColor(self._glow_color)
        anim = QPropertyAnimation(self._shadow, b"blurRadius", self)
        anim.setDuration(180)
        anim.setStartValue(self._shadow.blurRadius())
        anim.setEndValue(20)
        anim.setEasingCurve(QEasingCurve.OutQuad)
        anim.start(QPropertyAnimation.DeleteWhenStopped)
        super().enterEvent(event)

    def leaveEvent(self, event):
        anim = QPropertyAnimation(self._shadow, b"blurRadius", self)
        anim.setDuration(260)
        anim.setStartValue(self._shadow.blurRadius())
        anim.setEndValue(0)
        anim.setEasingCurve(QEasingCurve.OutQuad)
        anim.start(QPropertyAnimation.DeleteWhenStopped)
        super().leaveEvent(event)


# =============================================================================
# MATPLOTLIB CANVAS
# =============================================================================
class MplCanvas(FigureCanvas):
    def __init__(self, fig):
        super().__init__(fig)
        self.setStyleSheet(f"background-color:{BG_PANEL};")
        fig.patch.set_facecolor(BG_PANEL)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


def styled_ax(ax, bg=BG_CARD):
    ax.set_facecolor(bg)
    ax.tick_params(colors=TEXT_SEC, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(GRID_LINE)
    ax.grid(color=GRID_LINE, linewidth=0.4, alpha=0.5)


# =============================================================================
# PANEL: CELL VOLTAGE BAR CHART
# =============================================================================
class CellVoltagePanel(QWidget):
    def __init__(self):
        super().__init__()
        self.fig = Figure(figsize=(5, 2), dpi=90)
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.22)
        self.ax = self.fig.add_subplot(111)
        self.canvas = MplCanvas(self.fig)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self._init_chart()

    def _init_chart(self):
        ax = self.ax
        styled_ax(ax)
        ax.set_xlabel("Series Group", color=TEXT_SEC, fontsize=7)
        ax.set_ylabel("V", color=TEXT_SEC, fontsize=7)
        ax.set_ylim(CELL_MIN_VOLTAGE_V - 0.05, CELL_MAX_VOLTAGE_V + 0.05)
        ax.axhline(CELL_MAX_VOLTAGE_V, color=ACCENT_RED, lw=0.7, ls="--", alpha=0.5)
        ax.axhline(CELL_MIN_VOLTAGE_V, color=ACCENT_YLW, lw=0.7, ls="--", alpha=0.5)
        ax.grid(axis="y", color=GRID_LINE, linewidth=0.4)
        ax.set_axisbelow(True)
        self.bars = ax.bar(range(NUM_CELLS_SERIES),
                           [CELL_MAX_VOLTAGE_V] * NUM_CELLS_SERIES,
                           color=ACCENT_CYAN, alpha=0.85, width=0.68, zorder=2)
        ax.set_xticks(range(NUM_CELLS_SERIES))
        ax.set_xticklabels([f"S{i+1}" for i in range(NUM_CELLS_SERIES)], fontsize=7)
        ax.set_title("Cell Voltages", color=TEXT_PRI, fontsize=8, fontweight="bold", pad=3)
        self.canvas.draw()

    def update(self, state: SimState):
        if not state.cell_voltages:
            return
        voltages = np.array(state.cell_voltages)
        n_p = NUM_CELLS_PARALLEL
        for s, bar in enumerate(self.bars):
            grp = voltages[s * n_p:(s + 1) * n_p]
            mean_v = float(np.mean(grp))
            bar.set_height(mean_v)
            if mean_v >= CELL_MAX_VOLTAGE_V * 0.98:
                bar.set_color(ACCENT_RED)
            elif mean_v <= CELL_MIN_VOLTAGE_V * 1.02:
                bar.set_color(ACCENT_YLW)
            else:
                norm = (mean_v - CELL_MIN_VOLTAGE_V) / (CELL_MAX_VOLTAGE_V - CELL_MIN_VOLTAGE_V)
                bar.set_color(ACCENT_CYAN if norm > 0.4 else ACCENT_PRP)
        self.ax.set_title(
            f"Cell Voltages  ·  \u0394V = {state.delta_v_mv:.1f} mV",
            color=TEXT_PRI, fontsize=8, fontweight="bold", pad=3)
        self.canvas.draw_idle()


# =============================================================================
# PANEL: SOC HALF-DONUT + SOH TREND  (side by side, compact)
# =============================================================================
class SOCSOHPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.fig = Figure(figsize=(4, 2), dpi=90)
        self.fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.18, wspace=0.42)
        gs = gridspec.GridSpec(1, 2, figure=self.fig)
        self.ax_gauge = self.fig.add_subplot(gs[0])
        self.ax_soh   = self.fig.add_subplot(gs[1])
        self.canvas   = MplCanvas(self.fig)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self._soh_history = deque(maxlen=HISTORY_LEN)
        self._t_history   = deque(maxlen=HISTORY_LEN)
        self._init_charts()

    def _init_charts(self):
        # ── SOC half-donut ──
        ax = self.ax_gauge
        ax.set_facecolor(BG_PANEL)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-0.25, 1.25)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        # Track
        theta_bg = np.linspace(np.pi, 0, 200)
        ax.plot(np.cos(theta_bg), np.sin(theta_bg), lw=14,
                color=GRID_LINE, solid_capstyle="round", zorder=1)
        # Active arc (starts full)
        theta_act = np.linspace(np.pi, 0, 200)
        self._arc_line, = ax.plot(np.cos(theta_act), np.sin(theta_act),
                                  lw=14, color=ACCENT_GRN,
                                  solid_capstyle="round", zorder=2)
        self._soc_pct = ax.text(0, 0.28, "100%", ha="center", va="center",
                                fontsize=11, fontweight="bold", color=ACCENT_GRN,
                                fontfamily=FONT_MONO, zorder=5)
        ax.text(0, -0.12, "SOC", ha="center", va="center",
                fontsize=6, color=TEXT_SEC, fontfamily=FONT_UI,
                fontweight="700", zorder=5)
        ax.set_title("SOC (EKF)", color=TEXT_PRI, fontsize=8, fontweight="bold", pad=2)

        # ── SOH trend ──
        ax2 = self.ax_soh
        styled_ax(ax2)
        ax2.set_ylim(0.6, 1.05)
        ax2.set_ylabel("SOH", color=TEXT_SEC, fontsize=7)
        ax2.set_xlabel("Time [s]", color=TEXT_SEC, fontsize=7)
        ax2.axhline(0.70, color=ACCENT_RED, lw=0.7, ls="--", alpha=0.6)
        self._soh_line, = ax2.plot([], [], color=ACCENT_PRP, lw=1.5)
        ax2.set_title("SOH Trend", color=TEXT_PRI, fontsize=8, fontweight="bold", pad=3)
        self.canvas.draw()

    def update(self, state: SimState):
        soc = state.pack_soc_ekf  # 0-1
        theta_end = np.pi - soc * np.pi
        pts = max(2, int(soc * 200))
        theta = np.linspace(np.pi, theta_end, pts)
        self._arc_line.set_data(np.cos(theta), np.sin(theta))
        color = ACCENT_GRN if soc > 0.5 else (ACCENT_YLW if soc > 0.2 else ACCENT_RED)
        self._arc_line.set_color(color)
        self._soc_pct.set_text(f"{soc * 100:.1f}%")
        self._soc_pct.set_color(color)
        self._soh_history.append(state.pack_soh)
        self._t_history.append(state.time_s)
        self._soh_line.set_data(list(self._t_history), list(self._soh_history))
        self.ax_soh.set_xlim(max(0, state.time_s - HISTORY_LEN), state.time_s + 10)
        self.canvas.draw_idle()


# =============================================================================
# PANEL: TEMPERATURE HEATMAP
# =============================================================================
class TempHeatmapPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.fig = Figure(figsize=(4, 2), dpi=90)
        self.fig.subplots_adjust(left=0.10, right=0.88, top=0.85, bottom=0.18)
        self.ax = self.fig.add_subplot(111)
        self.canvas = MplCanvas(self.fig)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self._init_chart()

    def _init_chart(self):
        ax = self.ax
        ax.set_facecolor(BG_CARD)
        data = np.full((NUM_CELLS_SERIES, NUM_CELLS_PARALLEL), 25.0)
        self._im = ax.imshow(data, cmap="RdYlGn_r", vmin=20, vmax=60, aspect="auto")
        cb = self.fig.colorbar(self._im, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label("°C", color=TEXT_SEC, fontsize=7)
        cb.ax.tick_params(labelsize=6, colors=TEXT_SEC)
        ax.set_xlabel("Parallel", color=TEXT_SEC, fontsize=7)
        ax.set_ylabel("Series", color=TEXT_SEC, fontsize=7)
        ax.tick_params(colors=TEXT_SEC, labelsize=6)
        ax.set_xticks(range(NUM_CELLS_PARALLEL))
        ax.set_yticks(range(NUM_CELLS_SERIES))
        ax.set_xticklabels([f"P{i+1}" for i in range(NUM_CELLS_PARALLEL)], fontsize=5)
        ax.set_yticklabels([f"S{i+1}" for i in range(NUM_CELLS_SERIES)], fontsize=5)
        self._txts = []
        for s in range(NUM_CELLS_SERIES):
            row = []
            for p in range(NUM_CELLS_PARALLEL):
                t = ax.text(p, s, "25", ha="center", va="center",
                            fontsize=4.5, color="white", fontweight="bold")
                row.append(t)
            self._txts.append(row)
        ax.set_title("Temperature Map [°C]", color=TEXT_PRI, fontsize=8, fontweight="bold", pad=3)
        self.canvas.draw()

    def update(self, state: SimState):
        if not state.cell_temps:
            return
        temps = np.array(state.cell_temps).reshape(NUM_CELLS_SERIES, NUM_CELLS_PARALLEL)
        self._im.set_data(temps)
        for s in range(NUM_CELLS_SERIES):
            for p in range(NUM_CELLS_PARALLEL):
                self._txts[s][p].set_text(f"{temps[s, p]:.0f}")
        self.ax.set_title(f"Temperature Map [°C]  ·  Max {state.max_temp_c:.1f}°C",
                          color=TEXT_PRI, fontsize=8, fontweight="bold", pad=3)
        self.canvas.draw_idle()


# =============================================================================
# PANEL: BALANCING ACTIVITY
# =============================================================================
class BalancingPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.fig = Figure(figsize=(4, 2), dpi=90)
        self.fig.subplots_adjust(left=0.10, right=0.97, top=0.85, bottom=0.22)
        self.ax = self.fig.add_subplot(111)
        self.canvas = MplCanvas(self.fig)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self._dv_history = deque(maxlen=HISTORY_LEN)
        self._t_history  = deque(maxlen=HISTORY_LEN)
        self._init_chart()

    def _init_chart(self):
        ax = self.ax
        styled_ax(ax)
        ax.set_ylabel("\u0394V [mV]", color=TEXT_SEC, fontsize=7)
        ax.set_xlabel("Time [s]", color=TEXT_SEC, fontsize=7)
        ax.axhline(20, color=ACCENT_YLW, lw=0.7, ls="--", alpha=0.7, label="Passive 20mV")
        ax.axhline(10, color=ACCENT_GRN,  lw=0.7, ls="--", alpha=0.7, label="Active 10mV")
        self._dv_line,    = ax.plot([], [], color=ACCENT_CYAN, lw=1.4)
        self._bal_scatter = ax.scatter([], [], color=ACCENT_YLW, s=8, zorder=5, alpha=0.7)
        ax.legend(fontsize=6, facecolor=BG_CARD, labelcolor=TEXT_SEC, edgecolor=GRID_LINE)
        ax.set_title("Balancing Activity", color=TEXT_PRI, fontsize=8, fontweight="bold", pad=3)
        self.canvas.draw()

    def update(self, state: SimState):
        self._dv_history.append(state.delta_v_mv)
        self._t_history.append(state.time_s)
        self._dv_line.set_data(list(self._t_history), list(self._dv_history))
        self.ax.set_xlim(max(0, state.time_s - HISTORY_LEN), state.time_s + 10)
        self.ax.set_ylim(0, max(50, max(self._dv_history) * 1.2))
        if state.balancer and state.balancer.is_balancing:
            tl = list(self._t_history)
            dl = list(self._dv_history)
            self._bal_scatter.set_offsets(np.column_stack([tl[-10:], dl[-10:]]))
        n_bal = len(state.balancer.cells_balancing) if state.balancer else 0
        mode  = state.balancer.mode.upper() if state.balancer else "-"
        self.ax.set_title(
            f"Balancing — {mode}  ·  {n_bal} cells active",
            color=ACCENT_YLW if n_bal > 0 else TEXT_PRI,
            fontsize=8, fontweight="bold", pad=3)
        self.canvas.draw_idle()


# =============================================================================
# PANEL: FAULT LOG
# =============================================================================
class FaultPanel(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(5)

        hdr = QLabel("FAULT LOG")
        hdr.setStyleSheet(
            f"color:{TEXT_SEC}; font-size:8px; font-family:{FONT_UI}; font-weight:700;"
            f" letter-spacing:1.5px; background:transparent; border:none; padding:2px 0;"
        )
        lay.addWidget(hdr)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_DARK};
                color: {TEXT_PRI};
                font-family: {FONT_MONO};
                font-size: 8px;
                border: 1px solid {GRID_LINE};
                border-radius: 5px;
                padding: 5px;
            }}
        """)
        lay.addWidget(self.log, stretch=1)

        self.status = QLabel("\u25cf NORMAL")
        self.status.setStyleSheet(
            f"color:{ACCENT_GRN}; font-size:11px; font-family:{FONT_MONO}; font-weight:bold;"
            f" background:{BG_DARK}; border:1.5px solid {ACCENT_GRN}33;"
            f" border-radius:6px; padding:6px 8px;"
        )
        self.status.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.status)

    def update(self, state: SimState):
        if not state.fault or not state.fault.active_faults:
            self.status.setText("\u25cf NORMAL")
            self.status.setStyleSheet(
                f"color:{ACCENT_GRN}; font-size:11px; font-family:{FONT_MONO}; font-weight:bold;"
                f" background:{BG_DARK}; border:1.5px solid {ACCENT_GRN}33;"
                f" border-radius:6px; padding:6px 8px;"
            )
            return
        level = state.fault.highest_level
        clr = {0: ACCENT_GRN, 1: ACCENT_YLW, 2: ACCENT_RED, 3: ACCENT_RED}.get(int(level), ACCENT_RED)
        self.status.setText(f"\u25b2 {level.name}")
        self.status.setStyleSheet(
            f"color:{clr}; font-size:11px; font-family:{FONT_MONO}; font-weight:bold;"
            f" background:{BG_DARK}; border:1.5px solid {clr}44;"
            f" border-radius:6px; padding:6px 8px;"
        )
        for fault in state.fault.active_faults[-5:]:
            cell_lbl = f"Cell {fault.cell_id}" if fault.cell_id >= 0 else "PACK"
            self.log.append(
                f"[{state.time_s:.0f}s][{fault.level.name}] {fault.fault_type} {cell_lbl}: {fault.message}"
            )
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())


# =============================================================================
# HEADER  — compact instrument-cluster metric strip
# =============================================================================
class HeaderWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(56)
        self.setStyleSheet(f"background-color:{BG_PANEL}; border-bottom:1px solid {GRID_LINE};")

        root = QHBoxLayout(self)
        root.setContentsMargins(14, 5, 14, 5)
        root.setSpacing(0)

        brand = QLabel("BMS")
        brand.setStyleSheet(
            f"color:{ACCENT_CYAN}; font-size:17px; font-family:{FONT_MONO}; font-weight:bold;"
            f" letter-spacing:3px; background:transparent;"
        )
        root.addWidget(brand)
        sub = QLabel("  SIMULATOR")
        sub.setStyleSheet(
            f"color:{TEXT_SEC}; font-size:9px; font-family:{FONT_MONO};"
            f" font-weight:500; background:transparent;"
        )
        sub.setAlignment(Qt.AlignBottom)
        root.addWidget(sub)
        root.addSpacing(24)

        def vsep():
            s = QFrame(); s.setFrameShape(QFrame.VLine)
            s.setStyleSheet(f"color:{GRID_LINE};"); s.setFixedWidth(1)
            return s

        self._widgets = {}
        metrics = [
            ("PACK",   "pack_voltage",   "V",  ACCENT_CYAN),
            ("CURR",   "pack_current",   "A",  TEXT_PRI),
            ("SOC",    "pack_soc_ekf",   "%",  ACCENT_GRN),
            ("SOH",    "pack_soh",       "%",  ACCENT_PRP),
            ("TEMP",   "max_temp_c",     "°C", ACCENT_YLW),
            ("\u0394V","delta_v_mv",     "mV", ACCENT_CYAN),
            ("ENERGY", "pack_energy_wh", "Wh", TEXT_MID),
        ]
        for lbl_txt, attr, unit, color in metrics:
            root.addWidget(vsep())
            root.addSpacing(14)
            cell = QWidget(); cell.setStyleSheet("background:transparent;")
            cl = QVBoxLayout(cell); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)
            lbl = QLabel(lbl_txt)
            lbl.setStyleSheet(
                f"color:{TEXT_SEC}; font-size:8px; font-family:{FONT_UI};"
                f" font-weight:700; letter-spacing:1px; background:transparent;"
            )
            val = QLabel("---")
            val.setStyleSheet(
                f"color:{color}; font-size:16px; font-family:{FONT_MONO};"
                f" font-weight:bold; background:transparent;"
            )
            cl.addWidget(lbl); cl.addWidget(val)
            root.addWidget(cell)
            root.addSpacing(14)
            self._widgets[attr] = (val, unit)
        root.addWidget(vsep())
        root.addStretch()

    def update(self, state: SimState):
        for attr, (w, unit) in self._widgets.items():
            v = getattr(state, attr, None)
            if v is None: continue
            if "soc" in attr or "soh" in attr:
                w.setText(f"{v*100:.1f}{unit}")
            elif attr == "pack_voltage":
                w.setText(f"{v:.2f}{unit}")
            else:
                w.setText(f"{v:.1f}{unit}")


# =============================================================================
# CONTROLS PANEL — bigger, glow-on-hover buttons
# =============================================================================
class ControlsPanel(QWidget):
    def __init__(self, sim: SimLoop):
        super().__init__()
        self.sim = sim
        self.setFixedHeight(80)
        self.setStyleSheet(
            f"background-color:{BG_PANEL}; border-top:1px solid {GRID_LINE};"
        )
        outer = QHBoxLayout(self)
        outer.setContentsMargins(18, 10, 18, 10)
        outer.setSpacing(14)

        def labeled(label_txt, widget):
            w = QWidget(); w.setStyleSheet("background:transparent;")
            vl = QVBoxLayout(w); vl.setContentsMargins(0, 0, 0, 0); vl.setSpacing(3)
            lbl = QLabel(label_txt)
            lbl.setStyleSheet(CTRL_LABEL + " background:transparent;")
            vl.addWidget(lbl); vl.addWidget(widget)
            return w

        def vsep():
            s = QFrame(); s.setFrameShape(QFrame.VLine)
            s.setStyleSheet(f"color:{GRID_LINE};"); s.setFixedWidth(1)
            return s

        # Profile
        self.profile_combo = QComboBox()
        self.profile_combo.addItems([
            "constant_1c", "constant_2c", "constant_half_c",
            "step_hppc", "udds", "cc_cv_charge", "random_walk",
        ])
        self.profile_combo.setStyleSheet(COMBO_STYLE)
        self.profile_combo.currentTextChanged.connect(self.sim.set_profile)
        outer.addWidget(labeled("PROFILE", self.profile_combo))

        # Balancing
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["passive", "active"])
        self.mode_combo.setStyleSheet(COMBO_STYLE)
        self.mode_combo.currentTextChanged.connect(self.sim.set_balancing_mode)
        outer.addWidget(labeled("BALANCING", self.mode_combo))

        outer.addWidget(vsep())

        # Pause
        self._paused = False
        self.pause_btn = MicroButton("\u23f8  Pause", variant="primary")
        self.pause_btn.clicked.connect(self._toggle_pause)
        outer.addWidget(labeled("SIMULATION", self.pause_btn))

        # Reset
        self.reset_btn = MicroButton("\u21ba  Reset", variant="danger")
        self.reset_btn.clicked.connect(self.sim.reset)
        outer.addWidget(labeled("", self.reset_btn))

        outer.addWidget(vsep())

        # Speed
        speed_w = QWidget(); speed_w.setStyleSheet("background:transparent;"); speed_w.setMinimumWidth(200)
        sv = QVBoxLayout(speed_w); sv.setContentsMargins(0, 0, 0, 0); sv.setSpacing(3)
        top_row = QHBoxLayout()
        spd_lbl = QLabel("SPEED"); spd_lbl.setStyleSheet(CTRL_LABEL + " background:transparent;")
        self.speed_val = QLabel("1\u00d7")
        self.speed_val.setStyleSheet(
            f"color:{ACCENT_CYAN}; font-family:{FONT_MONO}; font-size:13px;"
            f" font-weight:bold; background:transparent;"
        )
        top_row.addWidget(spd_lbl); top_row.addStretch(); top_row.addWidget(self.speed_val)
        sv.addLayout(top_row)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1); self.speed_slider.setMaximum(50); self.speed_slider.setValue(1)
        self.speed_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background:{GRID_LINE}; height:6px; border-radius:3px;
            }}
            QSlider::sub-page:horizontal {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 {ACCENT_PRP}, stop:1 {ACCENT_CYAN});
                height:6px; border-radius:3px;
            }}
            QSlider::handle:horizontal {{
                background:{BG_PANEL}; border:2px solid {ACCENT_CYAN};
                width:18px; height:18px; margin:-6px 0; border-radius:9px;
            }}
            QSlider::handle:horizontal:hover {{
                background:{ACCENT_CYAN};
            }}
            QSlider::handle:horizontal:pressed {{
                background:{ACCENT_CYAN};
                width:22px; height:22px; margin:-8px 0; border-radius:11px;
            }}
        """)
        self.speed_slider.valueChanged.connect(self._on_speed)
        sv.addWidget(self.speed_slider)
        outer.addWidget(speed_w, stretch=1)

        outer.addWidget(vsep())

        self.time_lbl = QLabel("t = 0s  \u00b7  Tick 0")
        self.time_lbl.setStyleSheet(
            f"color:{TEXT_SEC}; font-family:{FONT_MONO}; font-size:11px; background:transparent;"
        )
        self.time_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        outer.addWidget(self.time_lbl)

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self.sim.pause()
            self.pause_btn.setText("\u25b6  Resume")
        else:
            self.sim.resume()
            self.pause_btn.setText("\u23f8  Pause")

    def _on_speed(self, v):
        self.sim.speed = float(v)
        self.speed_val.setText(f"{v}\u00d7")

    def update_time(self, state: SimState):
        self.time_lbl.setText(f"t = {state.time_s:.0f}s  \u00b7  Tick {state.tick}")


# =============================================================================
# SIGNAL BRIDGE
# =============================================================================
class SignalBridge(QObject):
    state_updated = pyqtSignal(object)


# =============================================================================
# MAIN DASHBOARD
# =============================================================================
class BMSDashboard(QMainWindow):
    def __init__(self, sim: SimLoop):
        super().__init__()
        self.sim    = sim
        self.bridge = SignalBridge()
        self.bridge.state_updated.connect(self._on_state)
        self.setWindowTitle("BMS Simulator — 8S8P NMC Pack")
        self.setMinimumSize(1280, 760)
        self.setStyleSheet(f"background-color:{BG_DARK};")
        self._build_ui()
        sim.register_callback(lambda s: self.bridge.state_updated.emit(s))

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = HeaderWidget()
        root.addWidget(self.header)

        content = QWidget()
        content.setStyleSheet(f"background-color:{BG_DARK};")
        grid = QGridLayout(content)
        grid.setContentsMargins(8, 8, 8, 6)
        grid.setSpacing(6)

        self.voltage_panel = CellVoltagePanel()
        self.socsoh_panel  = SOCSOHPanel()
        self.fault_panel   = FaultPanel()
        self.temp_panel    = TempHeatmapPanel()
        self.balance_panel = BalancingPanel()

        grid.addWidget(self._card(self.voltage_panel),  0, 0, 1, 2)
        grid.addWidget(self._card(self.socsoh_panel),   0, 2, 1, 1)
        grid.addWidget(self._card(self.fault_panel),    0, 3, 1, 1)
        grid.addWidget(self._card(self.temp_panel),     1, 0, 1, 2)
        grid.addWidget(self._card(self.balance_panel),  1, 2, 1, 2)

        grid.setRowStretch(0, 5)
        grid.setRowStretch(1, 4)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 2)
        grid.setColumnStretch(3, 1)

        root.addWidget(content, stretch=1)
        self.controls = ControlsPanel(self.sim)
        root.addWidget(self.controls)

    def _card(self, w: QWidget) -> QFrame:
        f = QFrame()
        f.setStyleSheet(
            f"background-color:{BG_PANEL}; border:1px solid {GRID_LINE}; border-radius:7px;"
        )
        vl = QVBoxLayout(f)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.addWidget(w)
        return f

    def _on_state(self, state: SimState):
        self.header.update(state)
        self.voltage_panel.update(state)
        self.socsoh_panel.update(state)
        self.temp_panel.update(state)
        self.balance_panel.update(state)
        self.fault_panel.update(state)
        self.controls.update_time(state)

    def closeEvent(self, event):
        self.sim.stop()
        event.accept()


# =============================================================================
# ENTRY POINT
# =============================================================================
def launch_dashboard(
    profile        : str   = "constant_1c",
    balancing_mode : str   = "passive",
    soc_init       : float = 1.0,
    speed          : float = 5.0,
):
    app = QApplication.instance() or QApplication(sys.argv)
    sim = SimLoop(
        profile=profile, balancing_mode=balancing_mode,
        soc_init=soc_init, soc_spread=0.04, duration_s=0, speed=speed,
    )
    window = BMSDashboard(sim)
    window.show()
    sim.start()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch_dashboard(profile="constant_1c", balancing_mode="passive", speed=5.0)