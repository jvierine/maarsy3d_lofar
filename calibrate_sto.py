import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob
import sto_array as sa
import jcoord
import scipy.constants as c
def uv_coverage(x,y,pairs,N=1000):
    u=n.zeros(len(pairs))
    v=n.zeros(len(pairs))
    for pi in range(len(pairs)):
        u[pi]=x[pairs[pi][1]]-x[pairs[pi][0]]
        v[pi]=y[pairs[pi][1]]-y[pairs[pi][0]]

    urange=2*n.max([n.abs(n.max(u)),n.abs(n.min(u))])
    vrange=2*n.max([n.abs(n.max(v)),n.abs(n.min(v))])    
    uvrange=n.max([urange,vrange])
    du=uvrange/N
    uidx=n.array(n.round(u/du),dtype=int)
    vidx=n.array(n.round(v/du),dtype=int)
    uidx[uidx<0]=N+uidx[uidx<0]
    vidx[vidx<0]=N+vidx[vidx<0]    
 #   plt.plot(uidx,vidx,"x")
  #  plt.show()

    return(u,v,uidx,vidx)

lam=c.c/53.5e6 # wavelength in meters
def kvecs(N=400,maxdcos=0.3,k=2.0*n.pi/lam):
    l=n.linspace(-maxdcos,maxdcos,num=N)
    m=n.linspace(-maxdcos,maxdcos,num=N)    
    ll,mm=n.meshgrid(l,m)
    nn=n.sqrt(1-ll**2.0+mm**2.0)
    kvec_x = k*ll
    kvec_y = k*mm
    kvec_z = k*nn
    mask = n.sqrt(ll**2.0+mm**2.0) < maxdcos
    return(kvec_x,kvec_y,l,m,mask)

def find_angle(u,v,S,kvec_x,kvec_y,l,m,mask=1.0):
    meas = n.exp(1j*n.angle(S))

    MF = n.zeros(kvec_x.shape,dtype=n.complex64)
    for i in range(len(meas)):
        MF+=meas[i]*n.exp(-1j*(kvec_x*u[i] + kvec_y*v[i]))
    MF=MF*mask
        
    i,j=n.unravel_index(n.argmax(n.abs(MF)),kvec_x.shape)
    if False:
        plt.pcolormesh(n.abs(MF))
        plt.colorbar()
        plt.show()
    return({"l":l[i],"m":m[j],"MF":MF[i,j]/len(meas)})



xyz,xyza,latlonh=sa.get_antenna_xyz()


# ecef positions (m) of antenna modules
ecef=n.zeros([latlonh.shape[0],3])
for ai in range(latlonh.shape[0]):
    ecef[ai,:]=jcoord.geodetic2ecef(latlonh[ai,0],latlonh[ai,1],latlonh[ai,2])
    print(ecef[ai,:])

n_modules=ecef.shape[0]

# maarsy position
coords={"lat":69.29836217360676,
        "lon":16.04139069818655,
        "alt":1.0}

ecef_maarsy=jcoord.geodetic2ecef(coords["lat"],coords["lon"],coords["alt"])
ecef_sto=jcoord.geodetic2ecef(69.0089129954889,15.1424765870241,50)


# we know that the signal is combing in from the direction
ground_direction = ecef_sto-ecef_maarsy
ground_k=2.0*n.pi*ground_direction/n.linalg.norm(ground_direction)/lam




    
fl=glob.glob("spec*.h5")
fl.sort()

h=h5py.File(fl[0],"r")
S=h["S"][()]
if "module_pairs" not in h.keys():
    exit(0)
mp=h["module_pairs"][()]
h.close()

# these are the pairwise vectors used for interferometry
vecs=n.zeros([mp.shape[0],3])
for mi in range(mp.shape[0]):
    vecs[mi,:]=ecef[mp[mi,0],:]-ecef[mp[mi,1],:]

plt.plot(vecs[:,0],vecs[:,1],".")
plt.title("uv-coverage")
plt.xlabel("East-west baseline (m)")
plt.ylabel("North-south baseline (m)")
plt.savefig("memos/004-baselines.png")
plt.close()
#plt.show()
#print(mp.shape)
#exit(0)



def find_cal(xc,xhat0=n.zeros(n_modules-1)):
    phase_cal=n.zeros(n_modules)
    m=n.exp(1j*n.angle(xc))
    def ss(x):
        phase_cal[1:len(phase_cal)]=x
        MF=0.0+0.0j
        for i in range(xc.shape[0]):
            MF+=m[i]*n.exp(1j*(-phase_cal[mp[i,0]]+phase_cal[mp[i,1]]))*n.exp(-1j*(ground_k[0]*vecs[i,0] + ground_k[1]*vecs[i,1] + ground_k[2]*vecs[i,2]))
        #print(MF)
        return(-n.abs(MF))
    import scipy.optimize as so
    xhat=so.fmin(ss,xhat0,disp=False)
    xhat=so.fmin(ss,xhat,disp=False)
    xhat=so.fmin(ss,xhat,disp=False)
    xhat=n.mod(xhat,2*n.pi)
    print(ss(xhat),"%1.2f %1.2f %1.2f %1.2f %1.2f %1.2f %1.2f"%(xhat[0],xhat[1],xhat[2],xhat[3],xhat[4],xhat[5],xhat[6]))
    return(xhat)
    

#24-29, 51
#14,45
#45,55
cal=n.zeros([n_modules-1,2])
cals=n.zeros([len(fl),len(cal),2])
for fi,f in enumerate(fl):
    h=h5py.File(f,"r")
    S=h["S"][()]
    if "module_pairs" not in h.keys():
        continue
    mp=h["module_pairs"][()]

    ci=1
    pwr=n.abs(S[0,0,ci,:,:])+n.abs(S[0,1,ci,:,:])
    maarsy_rgi=n.argmax(pwr[51,:])
    if False:
        for i in range(S.shape[0]):
            plt.plot(S[i,0,ci,:,maarsy_rgi].real)
            plt.plot(S[i,0,ci,:,maarsy_rgi].imag)
            plt.show()
    
        plt.plot(S[:,0,ci,51,maarsy_rgi].real)
        plt.plot(S[:,0,ci,51,maarsy_rgi].imag)
        plt.plot(n.abs(S[:,0,ci,51,maarsy_rgi]))
        plt.xlabel("Baseline")
        plt.ylabel("Ground path XC")
        plt.show()

    cal[:,0]=find_cal(S[:,0,ci,51,maarsy_rgi],cal[:,0])
    cals[fi,:,0]=cal[:,0]
    cal[:,1]=find_cal(S[:,1,ci,51,maarsy_rgi],cal[:,1])
    cals[fi,:,1]=cal[:,1]
#    print(maarsy_rgi)
    h.close()


plt.subplot(211)
plt.pcolormesh(cals[:,:,0],cmap="hsv")
plt.xlabel("Antenna module")
plt.title("Pol 0")
plt.ylabel("Time")
cb=plt.colorbar()
cb.set_label("Phase calibration")
plt.subplot(212)
plt.pcolormesh(cals[:,:,1],cmap="hsv")
plt.title("Pol 1")
plt.xlabel("Antenna module")
plt.ylabel("Time")
cb=plt.colorbar()
cb.set_label("Phase calibration")
plt.show()
    
