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
print(f"r = {r:.6f}   r^2 = {r**2:.6f}")