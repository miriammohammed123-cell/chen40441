from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

c0 = 8         # initial concentration [g/kg]
T = 15           # temperature [°C]

L_mean = 100.0        # seed average length [μm]
sigma_L = 10.0        # seed length std dev [μm]
m_seed = 1.0          # seed mass [g]

rho_c = 1.11e-12      # crystal density [g/μm^3]
kv = np.pi / 4        # shape factor [-] (cylinder)

L_max = 500.0         # max length [μm]
dL = 1.0              # bin length [μm]
t_max = 86400         # simulation time [s]


# Solubility: c* = a * exp(b*T)
a = 3.37                # [g/kg]
b = 0.0359              # [1/°C]


# gets looks for the file in the folder and renames the headings
here = Path(__file__).parent
df = pd.read_csv(here / "SoF-Coursework-26-27-KineticData.csv")
df.columns = ["S", "G", "Nuc_rate"]


#Fitting function of G to S
S = df["S"].to_numpy()   # supersaturation [-]
G = df["G"].to_numpy()   # growth rate [µm/s]

#print (S)
#print (G)

#Defining piece wize groth function
def growth_function(S, k1, k2, k3, S_switch):
    """Piecewise growth rate function
       S < 1            : G = 0 (undersaturated warning)
       1 <= S < S_switch: G = k1 (S-1)^k2
       S >= S_switch    : G = k3*S + k4 
    """
    S = np.asarray(S, dtype=float)

    if np.any(S<1):
        warnings.warn("S<1, understaturation, G=0")

    S_part_1 = k1 * (S-1) ** k2
    k4 = (k1 * (S_switch - 1) ** k2) - k3*S_switch
    S_part_2 = k3 * S + k4

    return np.where(S<1, 0, np.where(S<S_switch, S_part_1, S_part_2))



#fitting Piece wize growth function parameters
fit_var_G, covar_G = curve_fit(
    growth_function, S, G,
    p0=[50, 2.5, 35, 1.4],                       # k1, k2, k3, S_switch
    bounds=([0, 0, 0, 1.05], [np.inf, 10, np.inf, 3.0]),
    max_nfev=20000,)

k1, k2, k3, S_switch = fit_var_G
k4 = k1 * (S_switch - 1)**k2 - k3 * S_switch
print(f"k1 = {k1:.2f}, k2 = {k2:.2f}, k3 = {k3:.2f}, k4 = {k4:.2f}, S_switch = {S_switch:.3f}")
print(f"for 1 < S < {S_switch:.3f} use: {k1:.2f}*(S-1)^{k2:.2f}, for S > {S_switch:.3f} use: {k3:.2f}*S + {k4:.2f}")


#R^2 for Growth function
G_pred = growth_function(S, *fit_var_G)          # predicted G at each S
SS_res = np.sum((G - G_pred)**2)                 # unexplained error
SS_tot = np.sum((G - np.mean(G))**2)             # total variation in data
R2 = 1 - SS_res / SS_tot
print(f"R² = {R2:.5f}")

x = []   # 1 / ln(S)^2
y = []   # ln(B)

for i in range(len(df)):
    if df["Nuc_rate"][i]>0 and df["S"][i]>1:
        x.append(1/(np.log(df["S"][i])**2))
        y.append(np.log(df["Nuc_rate"][i]))

x=np.array(x)
y=np.array(y)
slope,intercept=np.polyfit(x,y,1)
r=np.corrcoef(x,y)[0,1]
A=np.exp(intercept)
B=-slope
print(f"A = {A:.4e}")
print(f"B = {B:.4f}")

L = np.arange(dL / 2, L_max, dL)

# full normal pdf 
f = np.exp(-(L - L_mean) ** 2 / (2 * sigma_L ** 2)) / (sigma_L * np.sqrt(2 * np.pi))

m3=(rho_c * kv * np.sum(f * L ** 3 * dL))
# total number of seeds from the mass constraint
N_seed = m_seed /m3
n0 = N_seed * f          # crystals per μm

print(f"N_seed = {N_seed:.3e}")
print(f"mass   = {rho_c * kv * np.sum(n0 * L**3 * dL):.6f} g")
plt.figure(figsize=(6, 4))
plt.plot(L, n0)
plt.xlabel("Crystal length, L [μm]")
plt.ylabel("n(L,0) [crystals/μm]")
plt.title("Seed distribution")
plt.xlim(40, 160)
plt.savefig(here/ "seeds.png", dpi=200)
print("saved seeds.png")

# Solubility and supersaturation
def solubility(T):
    return a * np.exp(b * T)

def supersaturation(c, T):
    return c / solubility(T)

c_star = solubility(T)              # solubility at 15 °C [g/kg]
S0 = supersaturation(c0, T)         # initial supersaturation [-]

print(f"c*  = {c_star:.3f} g/kg")
print(f"S0  = {S0:.3f}")
print(f"max solute that can crystallise = {c0 - c_star:.3f} g/kg")

# ---------- Setup ----------
m_solvent = 1.0                 # solvent mass [kg]  (assumed basis, not in Table 1)
c_star = solubility(T)          # c* at 15 °C [g/kg]
N_bin = len(L)                  # number of size bins (500)


# ---------- Kinetics: links the two phases ----------
def kinetics(c):
    """Growth rate G [μm/s] and nucleation rate J [crystals/s] from concentration."""
    S = c / c_star
    if S > 1:
        G = float(growth_function(S, *fit_var_G))
        J = A * np.exp(-B / np.log(S)**2) * m_solvent
    else:
        G = 0.0
        J = 0.0
    return G, J


# ---------- Solid phase: population balance ----------
def solid_phase(f, G, J):
    """df/dt for every size bin."""
    dfdt = np.zeros(N_bin)
    for i in range(N_bin):
        if i == 0:
            flux_in = J                 # boundary condition at L = 0
        else:
            flux_in = G * f[i-1]        # growth in from the bin below
        flux_out = G * f[i]             # growth out to the bin above
        dfdt[i] = (flux_in - flux_out) / dL
    return dfdt


# ---------- Liquid phase: solute mass balance ----------
def liquid_phase(dfdt):
    """dc/dt: solute leaves the liquid at the rate the crystals gain mass."""
    mass_rate = 0.0                     # [g/s]
    for i in range(N_bin):
        mass_rate = mass_rate + rho_c * kv * L[i]**3 * dfdt[i] * dL
    dcdt = -mass_rate / m_solvent
    return dcdt


# ---------- Combine both phases for the solver ----------
def rhs(t, y):
    f = y[:N_bin]                       # solid phase state
    c = y[N_bin]                        # liquid phase state

    G, J = kinetics(c)
    dfdt = solid_phase(f, G, J)
    dcdt = liquid_phase(dfdt)

    dydt = np.zeros(N_bin + 1)
    for i in range(N_bin):
        dydt[i] = dfdt[i]
    dydt[N_bin] = dcdt
    return dydt


# ---------- Initial conditions ----------
y0 = np.zeros(N_bin + 1)
for i in range(N_bin):
    y0[i] = n0[i]                       # seeds
y0[N_bin] = c0                          # initial concentration


# ---------- Solve ----------
t_out = np.concatenate(([0.0], np.logspace(-1, np.log10(t_max), 499)))
sol = solve_ivp(rhs, [0, t_max], y0, method="LSODA",
                t_eval=t_out, rtol=1e-6, atol=1e-8)
print("solver finished:", sol.success)

f_end = sol.y[:N_bin, -1]               # final size distribution
c_t = sol.y[N_bin]                      # concentration over time

print(sol.y[:N_bin, -1])                                   # all 500 final PSD values [crystals/μm]
print(f"final concentration = {sol.y[N_bin, -1]:.4f} g/kg")

# ---------- Checks ----------
m_end = rho_c * kv * np.sum(f_end * L**3 * dL)
c_check = c0 - (m_end - m_seed) / m_solvent
print(f"final c = {c_t[-1]:.4f} g/kg, from crystal mass = {c_check:.4f} g/kg")
print(f"final S = {c_t[-1] / c_star:.4f}, crystal mass = {m_end:.4f} g")
print(f"f in the last bin at the end = {f_end[-1]:.3e}")

# ---------- Plot the size distribution at the final time ----------
f_end = sol.y[:N_bin, -1]               # all bins, last time point

plt.figure(figsize=(7, 5))
plt.plot(L, f_end, label=f"t = {t_out[-1]:.0f} s")
plt.xlabel("Crystal length, L [μm]")
plt.ylabel("f(t, L) [crystals/μm]")
plt.xlim(0, 250)
plt.legend()
plt.tight_layout()
plt.savefig(here / "psd_final.png", dpi=200)
print("saved psd_final.png")

for i in range(N_bin):
    print(f"L = {L[i]:6.1f} μm    f = {sol.y[i, -1]:.4e} crystals/μm")
print(f"final concentration = {sol.y[N_bin, -1]:.4f} g/kg")

# ---------- Method of moments ----------
def moments_rhs(t, z):
    """z = [m0, m1, m2, m3, c]. Returns dz/dt."""
    c = z[4]
    G, J = kinetics(c)

    dzdt = np.zeros(5)
    dzdt[0] = J                                   # dm0/dt = J
    for j in range(1, 4):
        dzdt[j] = j * G * z[j-1]                  # dmj/dt = j G m(j-1)
    dzdt[4] = -rho_c * kv * dzdt[3] / m_solvent   # dc/dt = -rho_c kv dm3/dt
    return dzdt

# initial moments from the seed distribution, then c0
z0 = np.zeros(5)
for j in range(4):
    for i in range(N_bin):
        z0[j] = z0[j] + L[i]**j * n0[i] * dL
z0[4] = c0

sol_mom = solve_ivp(moments_rhs, [0, t_max], z0, method="LSODA",
                    t_eval=t_out, rtol=1e-8, atol=1e-10)
print("moments solver finished:", sol_mom.success)

# moments of the PBE solution at every output time
pbe_mom = np.zeros((4, len(t_out)))
for k in range(len(t_out)):
    for j in range(4):
        for i in range(N_bin):
            pbe_mom[j, k] = pbe_mom[j, k] + L[i]**j * sol.y[i, k] * dL

# compare the two methods at the final time
print("final time comparison (PBE vs method of moments)")
for j in range(4):
    diff = 100 * (pbe_mom[j, -1] - sol_mom.y[j, -1]) / sol_mom.y[j, -1]
    print(f"m{j}: PBE = {pbe_mom[j, -1]:.4e}   MoM = {sol_mom.y[j, -1]:.4e}   difference = {diff:.2f} %")
print(f"c : PBE = {c_t[-1]:.4f}   MoM = {sol_mom.y[4, -1]:.4f} g/kg")
print(f"mean size m1/m0: PBE = {pbe_mom[1, -1] / pbe_mom[0, -1]:.1f}   MoM = {sol_mom.y[1, -1] / sol_mom.y[0, -1]:.1f} μm")

# plot both methods against time
plt.figure(figsize=(7, 5))
plt.subplot(1, 2, 1)
plt.plot(t_out, c_t, label="PBE")
plt.plot(t_out, sol_mom.y[4], "--", label="Method of moments")
plt.xscale("log")
plt.xlabel("Time [s]")
plt.ylabel("Concentration [g/kg]")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_out, pbe_mom[1] / pbe_mom[0], label="PBE")
plt.plot(t_out, sol_mom.y[1] / sol_mom.y[0], "--", label="Method of moments")
plt.xscale("log")
plt.xlabel("Time [s]")
plt.ylabel("Mean crystal length [μm]")
plt.legend()

plt.tight_layout()
plt.savefig(here / "moments_check.png", dpi=200)
print("saved moments_check.png")