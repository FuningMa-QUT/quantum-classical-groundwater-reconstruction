"""Figure 1c and 1d: P06 injection-extraction benchmark configuration.

This script builds, runs, and plots the MODFLOW 6 P06 model.
It requires flopy and the MODFLOW 6 executable to be installed.
"""

from pathlib import Path
import sys
import locale
import flopy
import matplotlib.pyplot as plt
import numpy as np

# ── Fix Windows encoding issue ─────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────
workspace = Path.cwd() / "p06_workspace"
workspace.mkdir(parents=True, exist_ok=True)
figs_path = Path.cwd() / "figures"
figs_path.mkdir(parents=True, exist_ok=True)

sim_name = "p06_injection_extraction"
sim_ws = workspace / sim_name

# ── Model parameters ──────────────────────────────────────
length_units = "feet"
time_units = "days"
nlay, nrow, ncol = 1, 31, 31
delr, delc, delz = 900.0, 900.0, 20.0
top = 0.0
prsity = 0.35
k11 = 432.0  # ft/d
al = 100.0   # ft
trpt = 1.0
qwell = 86400.0  # ft^3/d
cwell = 100.0    # percent

# Time: injection 0-2.5yr, extraction 2.5-10yr
perlen = [912.5, 2737.5]
nper = len(perlen)
nstp = [365, 1095]
tsmult = [1.0, 1.0]

# Solver settings
nouter, ninner = 100, 300
hclose, rclose, relax = 1e-6, 1e-6, 1.0

# Initial conditions
strt = np.zeros((nlay, nrow, ncol), dtype=float)
idomain = np.ones((nlay, nrow, ncol), dtype=int)

# Well data: [layer, row, col, flow_rate, concentration]
spd_mf6 = {
    0: [[(0, 15, 15), qwell, cwell]],    # Injection
    1: [[(0, 15, 15), -qwell, 0.0]],     # Extraction
}

# Constant head boundaries: all four sides h=0
chdspd = []
for i in range(nrow):
    chdspd.append([(0, i, 0), 0.0])
    chdspd.append([(0, i, ncol - 1), 0.0])
for j in range(1, ncol - 1):
    chdspd.append([(0, 0, j), 0.0])
    chdspd.append([(0, nrow - 1, j), 0.0])
chdspd = {0: chdspd}

# ── Build MODFLOW 6 model ──────────────────────────────────
print("Building MODFLOW 6 model...")

mf6_exe = "G:/Modflow/mf6.4.2_win64/bin/mf6.exe"

sim = flopy.mf6.MFSimulation(
    sim_name=sim_name,
    sim_ws=sim_ws,
    exe_name=mf6_exe,
)

# Time discretization
tdis_rc = []
for i in range(nper):
    tdis_rc.append((perlen[i], nstp[i], tsmult[i]))
flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_rc, time_units=time_units)

# Groundwater flow model
gwf = flopy.mf6.ModflowGwf(
    sim,
    modelname="flow",
    save_flows=True,
)

# Separate IMS for GWF (must be registered first)
ims_gwf = flopy.mf6.ModflowIms(
    sim,
    filename="flow.ims",
    print_option="SUMMARY",
    outer_dvclose=hclose,
    outer_maximum=nouter,
    inner_maximum=ninner,
    inner_dvclose=hclose,
    rcloserecord=rclose,
    linear_acceleration="BICGSTAB",
    relaxation_factor=relax,
)
sim.register_ims_package(ims_gwf, [gwf.name])

# Discretization
flopy.mf6.ModflowGwfdis(
    gwf,
    length_units=length_units,
    nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc, top=top, botm=[-delz],
    idomain=idomain,
)

# Node property flow
flopy.mf6.ModflowGwfnpf(
    gwf,
    icelltype=0,
    k=k11,
    k33=k11,
    save_specific_discharge=True,
)

# Initial conditions
flopy.mf6.ModflowGwfic(gwf, strt=strt)

# Constant head
flopy.mf6.ModflowGwfchd(
    gwf,
    maxbound=len(chdspd[0]),
    stress_period_data=chdspd,
    pname="CHD-1",
)

# Well package - MUST set pname for GWT to reference
flopy.mf6.ModflowGwfwel(
    gwf,
    stress_period_data=spd_mf6,
    auxiliary="CONCENTRATION",
    pname="WEL-1",
)

# Output control
flopy.mf6.ModflowGwfoc(
    gwf,
    head_filerecord="flow.hds",
    budget_filerecord="flow.bud",
    saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
)

# Groundwater transport model
gwt = flopy.mf6.MFModel(
    sim,
    model_type="gwt6",
    modelname="transport",
)

# Separate IMS for GWT (must be registered second)
ims_gwt = flopy.mf6.ModflowIms(
    sim,
    filename="transport.ims",
    print_option="SUMMARY",
    outer_dvclose=hclose,
    outer_maximum=nouter,
    inner_maximum=ninner,
    inner_dvclose=hclose,
    rcloserecord=rclose,
    linear_acceleration="BICGSTAB",
    relaxation_factor=relax,
)
sim.register_ims_package(ims_gwt, [gwt.name])

# Transport discretization
flopy.mf6.ModflowGwtdis(
    gwt,
    nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc, top=top, botm=[-delz],
    idomain=idomain,
)

# Initial concentration
flopy.mf6.ModflowGwtic(gwt, strt=0.0)

# Advection
flopy.mf6.ModflowGwtadv(gwt, scheme="TVD")

# Dispersion
flopy.mf6.ModflowGwtdsp(gwt, alh=al, ath1=al * trpt, xt3d_off=True)

# Mobile storage
flopy.mf6.ModflowGwtmst(gwt, porosity=prsity)

# Source-sink mixing - references WEL-1 by pname
flopy.mf6.ModflowGwtssm(
    gwt,
    sources=[("WEL-1", "AUX", "CONCENTRATION")],
)

# Transport output control
flopy.mf6.ModflowGwtoc(
    gwt,
    budget_filerecord="transport.bud",
    concentration_filerecord="transport.ucn",
    saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "LAST")],
)

# Flow-transport exchange
flopy.mf6.ModflowGwfgwt(
    sim,
    exgtype="GWF6-GWT6",
    exgmnamea="flow",
    exgmnameb="transport",
)

# ── Write and run ──────────────────────────────────────────
print("Writing input files...")
sim.write_simulation(silent=True)

print("Running MODFLOW 6...")
sim.run_simulation(silent=False, report=True, encoding="latin-1")
print("Simulation completed successfully.")

# ── Read results for plotting ──────────────────────────────
print("Reading results...")

head_data = gwf.output.head()
heads = head_data.get_alldata()

conc_data = gwt.output.concentration()
concs = conc_data.get_alldata()

# ── Plot ──────────────────────────────────────────────────
print("Creating figure...")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

injection_time_idx = 0
extraction_time_idx = -1

titles = [
    "Injection phase (0-2.5 yr): flow directed outward",
    "Pumpback phase (2.5-10 yr): flow directed inward",
]
panels = ["(c)", "(d)"]

x_centers = np.arange(0, ncol) * delr + delr / 2
y_centers = np.arange(0, nrow) * delc + delc / 2
X, Y = np.meshgrid(x_centers, y_centers)

well_x = 15 * delr + delr / 2
well_y = 15 * delc + delc / 2

for idx, ax in enumerate(axes):
    if idx == 0:
        h = heads[injection_time_idx, 0, :, :]
        c = concs[injection_time_idx, 0, :, :] if len(concs) > 0 else None
    else:
        h = heads[extraction_time_idx, 0, :, :]
        c = concs[extraction_time_idx, 0, :, :] if len(concs) > 0 else None

    # Head fill
    levels = np.linspace(h.min(), h.max(), 25)
    cf = ax.contourf(
        X, Y, h,
        levels=levels,
        cmap="Blues",
        alpha=0.6,
        extend="both",
    )

    # Head contour lines
    ax.contour(
        X, Y, h,
        levels=levels[::3],
        colors="darkblue",
        linewidths=0.4,
        alpha=0.4,
    )

    # Flow direction from head gradient
    dhdx = np.zeros_like(h)
    dhdy = np.zeros_like(h)
    dhdx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2 * delr)
    dhdy[1:-1, :] = (h[2:, :] - h[:-2, :]) / (2 * delc)
    vx = -k11 * dhdx
    vy = -k11 * dhdy

    skip = 4
    ax.quiver(
        X[::skip, ::skip], Y[::skip, ::skip],
        vx[::skip, ::skip], vy[::skip, ::skip],
        color="darkred", alpha=0.7, scale=80000,
        width=0.004, headwidth=5,
    )

    # Concentration contours
    if c is not None and np.max(c) > 0:
        clevels = np.linspace(0.1, np.max(c), 8)
        ax.contour(
            X, Y, c,
            levels=clevels,
            colors="darkgreen",
            linewidths=1.2,
            alpha=0.7,
        )

    # Well marker
    ax.plot(well_x, well_y, marker="s", color="black",
            markersize=14, zorder=10, markeredgecolor="white",
            markeredgewidth=1.5)
    ax.plot(well_x, well_y, marker="x", color="white",
            markersize=8, zorder=11, linewidth=2)

    # Monitoring anchor examples
    if idx == 0:
        # Injection: downstream monitors
        for dist in [4, 7, 10]:
            mx = well_x + dist * delr
            if 1 < mx < (ncol - 1) * delr:
                for offset in [-1, 0, 1]:
                    my = well_y + offset * delc
                    ax.plot(mx, my, "o", color="#E65100", markersize=5,
                           alpha=0.6, markeredgecolor="white", markeredgewidth=0.5)
    else:
        # Extraction: capture-ring monitors
        for r in [4, 7]:
            theta = np.linspace(0, 2 * np.pi, 16)
            for t in theta:
                mx = well_x + r * delr * np.cos(t)
                my = well_y + r * delc * np.sin(t)
                if 1 < mx < (ncol - 1) * delr and 1 < my < (nrow - 1) * delc:
                    ax.plot(mx, my, "o", color="#E65100", markersize=5,
                           alpha=0.6, markeredgecolor="white", markeredgewidth=0.5)

    # Formatting
    ax.set_aspect("equal")
    ax.set_xlim(0, ncol * delr)
    ax.set_ylim(0, nrow * delc)
    ax.set_xlabel("x (ft)", fontsize=9)
    ax.set_ylabel("y (ft)", fontsize=9)
    ax.set_title(titles[idx], fontsize=11)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    # Panel label
    ax.text(0.02, 0.98, panels[idx], transform=ax.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold",
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
cbar = fig.colorbar(cf, cax=cbar_ax)
cbar.set_label("Hydraulic head (ft)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout(rect=[0, 0, 0.91, 1])

# Save
for fmt in ["png", "pdf"]:
    output_path = figs_path / "Figure_01_cd.{}".format(fmt)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    print("  Saved: {}".format(output_path))

plt.close(fig)
print("Done!")