import glob
import matplotlib.pyplot as plt
import numpy as n
import h5py
import stuffr

fl=glob.glob("spec*.h5")
fl.sort()
n_t=len(fl)
RTI=n.zeros([n_t,500],dtype=n.float32)

ci=1
ts=[]
for i in range(len(fl)):
    h=h5py.File(fl[i],"r")
    S=h["S"][()]
    ts.append(stuffr.unix2date(h["t0"][()]))
    dops=h["dops"][()]
    rg=n.arange(500)
    fidx=n.where(n.abs(dops)<50.0)[0]
    pwr=n.abs(S[0,0,ci,:,:])
    RTI[i,:]=n.max(pwr,axis=0)
    #plt.pcolormesh(dops,rg,10.0*n.log10(n.abs(S[0,0,1,:,:].T)**2.0))
    #plt.show()
    h.close()

plt.pcolormesh(ts,n.arange(500)*0.6,10.0*n.log10(RTI.T))
plt.xlabel("Time (UT)")
plt.ylabel("Total propagation distance (km)")
plt.colorbar()
plt.show()
