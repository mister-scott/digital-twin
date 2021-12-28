import matplotlib.pyplot as plt

from component import *
from icecream import ic

ids = []
mosfet = MOSFET()
vds_vals = [0.01*i for i in range(1,501)]
for v in vds_vals:
    ids.append(mosfet.id_vd(vgs=3.0,vds=v,rounding=False))

plt.figure(dpi=120)
plt.plot(vds_vals,ids)
plt.xlabel("Vds (V)",fontsize=15)
plt.ylabel("Ids (A)",fontsize=15)
plt.show()