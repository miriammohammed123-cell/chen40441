from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

c0 = 8.0              # initial concentration [g/kg]
T = 15.0              # temperature [°C]

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

# gets looks for the file in the folder and renames the headings
here = Path(__file__).parent
df = pd.read_csv(here / "SoF-Coursework-26-27-KineticData.csv")
df.columns = ["S", "G", "Nuc_rate"]

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

