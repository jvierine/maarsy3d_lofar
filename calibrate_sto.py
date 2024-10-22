import numpy as n
import matplotlib.pyplot as plt
import h5py
import glob
import sto_array as sa
import jcoord
import scipy.constants as c

from mpi4py import MPI

# use MPI to paralellize the processing steps
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("comm rank %d size %d"%(rank,size))


# maarsy position
coords={"lat":69.29836217360676,
        "lon":16.04139069818655,
        "alt":1.0}

ecef_maarsy=jcoord.geodetic2ecef(coords["lat"],coords["lon"],coords["alt"])
ecef_sto=jcoord.geodetic2ecef(69.0089129954889,15.1424765870241,50)

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
def kvecs(N=400,maxdcos=1.0,k=2.0*n.pi/lam):

    ecef_sto=jcoord.geodetic2ecef(69.0089129954889,15.1424765870241,50)
    ecef_sto_no=jcoord.geodetic2ecef(69.0089129954889+0.1,15.1424765870241,50)-ecef_sto
    ecef_sto_ea=jcoord.geodetic2ecef(69.0089129954889,15.1424765870241+0.1,50)-ecef_sto
    u_e=n.cross(ecef_sto_no,ecef_sto)
    u_e=u_e/n.linalg.norm(u_e)
    u_n=-n.cross(u_e,ecef_sto)
    u_n=u_n/n.linalg.norm(u_n)
    u_u=ecef_sto/n.linalg.norm(ecef_sto)

    # print(u_e)
    #   print(u_n)
    #  print(u_u)
    #    print(n.cross(u_n,u_u))
    #    print(n.dot(ecef_sto,u_u))
#    print(n.cross(ecef_sto,u_u))
 #   exit(0)
    
    
    
    l=n.linspace(-maxdcos,maxdcos,num=N)
    m=n.linspace(-maxdcos,maxdcos,num=N)    
    ll,mm=n.meshgrid(l,m)
    nn=n.sqrt(1-ll**2.0+mm**2.0)
    kvec=n.zeros([ll.shape[0],ll.shape[0],3],dtype=n.float32)
    kvec[:,:,0] = -(ll*u_n[0]+mm*u_e[0]+nn*u_u[0])
    kvec[:,:,1] = -(ll*u_n[1]+mm*u_e[1]+nn*u_u[1])
    kvec[:,:,2] = -(ll*u_n[2]+mm*u_e[2]+nn*u_u[2])
    norm=n.sqrt(kvec[:,:,0]**2.0+kvec[:,:,1]**2.0+kvec[:,:,2]**2.0)
    kvec=k*kvec/norm[:,:,None]


    ecef_maarsy=jcoord.geodetic2ecef(coords["lat"],coords["lon"],90e3)
    ecef_sto=jcoord.geodetic2ecef(69.0089129954889,15.1424765870241,50)
    ink=(ecef_sto-ecef_maarsy)

    ink=ink/n.linalg.norm(ink)
    angle=n.arccos((kvec[:,:,0]*ink[0]+kvec[:,:,1]*ink[1]+kvec[:,:,2]*ink[2])/n.sqrt(kvec[:,:,0]**2.0+kvec[:,:,1]**2.0+kvec[:,:,2]**2.0))

  #  plt.pcolormesh((kvec[:,:,0]**2.0+kvec[:,:,1]**2.0+kvec[:,:,2]**2.0))
 #   plt.colorbar()
#    plt.show()
    
#    knorm=n.sqrt(kvec[:,:,0]**2.0+kvec[:,:,1]**2.0+kvec[:,:,2]**2.0)
 #   k=2.0*n.pi*kvec/knorm[:,:,None]/lam
    mask = ((n.sqrt(ll**2.0+mm**2.0) < maxdcos) & (n.abs(angle) < n.pi*30/180.0))
    return(kvec,mask)

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

def towards_maarsy_k2(daz=60,dele=80,N=100):
    
    az,el,r=jcoord.geodetic_to_az_el_r(69.0089129954889,15.1424765870241,50, coords["lat"],coords["lon"], 1)
    azs=az+n.linspace(-daz/2,daz/2,num=N)
    els=el+n.linspace(0,dele,num=N)

    k=n.zeros([N,N,3],dtype=n.float32)
    print(azs.shape)
    for i in range(N):
        for j in range(N):
            k[i,j,:]=-(2.0*n.pi/lam)*jcoord.azel_ecef(69.0089129954889,15.1424765870241,50,azs[i],els[j])
    return(k,azs,els)

def towards_maarsy_k(minh=0,maxh=500,delta_mer=400,N=200):
    """
    create k-vectors from Sto towards maarsy at 80 km altitude
    """

    ecef_maarsy=jcoord.geodetic2ecef(coords["lat"],coords["lon"],0)
    up_u=ecef_maarsy/n.linalg.norm(ecef_maarsy)

    ecef_maarsy_no=jcoord.geodetic2ecef(coords["lat"]+1,coords["lon"],90e3)
    no_u=ecef_maarsy_no-ecef_maarsy
    no_u=no_u/n.linalg.norm(no_u)

    north=n.linspace(-delta_mer*1e3/2,delta_mer*1e3/2,num=N)
    up=n.linspace(minh*1e3,maxh*1e3,num=N)

    nn,uu=n.meshgrid(north,up)
#    nn=nn.flatten()
 #   uu=uu.flatten()    
    ecef_sto=jcoord.geodetic2ecef(69.0089129954889,15.1424765870241,50)
    
    # k vectors coming from maarsy volume to sto
    k = n.zeros([N,N,3],dtype=n.float32)
    k[:,:,0] = ecef_sto[0]-(nn*no_u[0] + uu*up_u[0] + ecef_maarsy[0])
    k[:,:,1] = ecef_sto[1]-(nn*no_u[1] + uu*up_u[1] + ecef_maarsy[1])
    k[:,:,2] = ecef_sto[2]-(nn*no_u[2] + uu*up_u[2] + ecef_maarsy[2])    
    

    knorm=n.sqrt(k[:,:,0]**2.0+k[:,:,1]**2.0+k[:,:,2]**2.0)
    k=2.0*n.pi*k/knorm[:,:,None]/lam
    return(k,north,up)
mask=1
#k,north,up=towards_maarsy_k(delta_up=180,delta_mer=100,N=100)


#@k,north,up=towards_maarsy_k2(N=100)
k,azs,els=towards_maarsy_k2(N=200)

#k,mask=kvecs(N=200)


xyz,xyza,latlonh=sa.get_antenna_xyz()



# ecef positions (m) of antenna modules
ecef=n.zeros([latlonh.shape[0],3])
for ai in range(latlonh.shape[0]):
    ecef[ai,:]=jcoord.geodetic2ecef(latlonh[ai,0],latlonh[ai,1],latlonh[ai,2])
    print(ecef[ai,:])

n_modules=ecef.shape[0]



# we know that the signal is combing in from the direction
ecef_maarsy=jcoord.geodetic2ecef(coords["lat"],coords["lon"],coords["alt"])
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


def image(XC):
    ho=h5py.File("stocal.h5","r")

    phase0=n.concatenate(([0],ho["cal_pol0"][()]))
    phase1=n.concatenate(([0],ho["cal_pol1"][()]))
    print(phase0)
    print(phase1)

    for fi in range(rank,len(fl),size):
        f=fl[fi]
        h=h5py.File(f,"r")
        S=h["S"][()]
        t0=h["t0"][()]
        if "module_pairs" not in h.keys():
            continue
        mp=h["module_pairs"][()]
        ci=1
        pwr=n.abs(S[0,0,ci,:,:])+n.abs(S[0,1,ci,:,:])
        nf=n.nanmedian(10.0*n.log10(pwr))
        print(nf)

        db=10.0*n.log10(pwr)
        db0=n.copy(db)

        db=db-nf        
        db[db<8]=0
        idx=n.where(db > 0)
        if False:
            plt.pcolormesh(db.T,vmax=30,vmin=0)
            plt.colorbar()
            plt.show()

        
        MF_all=n.zeros([k.shape[0],k.shape[0]],dtype=n.float32)
        MF_rg=n.zeros([k.shape[0],k.shape[0]],dtype=n.float32)        
        for i in range(len(idx[0])):
            print(idx[0][i])
            print(idx[1][i])
            MF=n.zeros([k.shape[0],k.shape[0]],dtype=n.complex64)
            print(idx[0][i])


            bg0=n.median(n.mean(S[:,0,ci,:,:],axis=1),axis=1)
            bg1=n.median(n.mean(S[:,1,ci,:,:],axis=1),axis=1)

            xc0=S[:,0,ci,idx[0][i],idx[1][i]]-bg0
            xc1=S[:,1,ci,idx[0][i],idx[1][i]]-bg1
            
            
            for xci in range(S.shape[0]):
#                print(xc[xci])
                print(vecs.shape)
                print(k.shape)
                MF+=xc0[xci]*n.exp(1j*(-phase0[mp[xci,0]]+phase0[mp[xci,1]]))*n.exp(-1j*(k[:,:,0]*vecs[xci,0] + k[:,:,1]*vecs[xci,1] + k[:,:,2]*vecs[xci,2]))*mask
                MF+=xc1[xci]*n.exp(1j*(-phase1[mp[xci,0]]+phase1[mp[xci,1]]))*n.exp(-1j*(k[:,:,0]*vecs[xci,0] + k[:,:,1]*vecs[xci,1] + k[:,:,2]*vecs[xci,2]))*mask
            MF2=n.copy(MF)
            
#            MF2.shape=(k.shape[0],k.shape[0])
            mi,mj=n.unravel_index(n.argmax(n.abs(MF2)),MF2.shape)
            MF_all[mi,mj]+=n.max(n.abs(MF2))
            MF_rg[mi,mj]=idx[1][i]
            if True:
                plt.pcolormesh(azs,els,n.abs(MF.T))
                plt.xlabel("Azimuth (deg)")
                plt.ylabel("Elevation (deg)")                
#                plt.plot(north[mj]/1e3,up[mi]/1e3+90,"x",color="black")
#                plt.xlabel("Up (km)")
 #               plt.ylabel("North-South (km)")
  #              plt.title("Position relative to MAARSY meridian")
                plt.show()
        MF_all=n.copy(MF_all)
        plt.figure(figsize=(16,9))
        plt.subplot(121)
        dB=10.0*n.log10(MF_all.T+1)
        peak=n.max(dB)
        plt.pcolormesh(azs,els,dB)
        plt.xlabel("Azimuth (deg)")
        plt.ylabel("Elevation (deg)")                
        import stuffr
        #
#        plt.pcolormesh(north/1e3,up/1e3+90,MF_rg,cmap="turbo")#10.0*n.log10(MF_all+1))
        plt.colorbar()
#        plt.plot(north[mj]/1e3+90,up[mi]/1e3,"x",color="black")
        #plt.ylabel("North (km)")
        #plt.xlabel("East (km)")
        #plt.title("Position relative to MAARSY meridian")
        plt.subplot(122)
        plt.pcolormesh(db0.T)
        plt.xlabel("Doppler gate")
        plt.ylabel("Range gate")
        plt.title("bg %s"%(stuffr.unix2datestr(t0)))

        plt.colorbar()        
        plt.savefig("image-%06d.png"%(fi))
        plt.close()
#        plt.show()
            
#        pwr=n.abs(S[0,0,ci,:,:])+n.abs(S[0,1,ci,:,:])
 #       maarsy_rgi=n.argmax(pwr[51,:])
#    m=n.exp(1j*n.angle(xc))
#    def ss(x):
        #phase_cal[1:len(phase_cal)]=x
 #       MF=0.0+0.0j
  #      for i in range(xc.shape[0]):
  
        #print(MF)
    #    return(-n.abs(MF))
  #  import scipy.optimize as so


image(0)





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

def cal_sto():
    """
    assume that direct path is coming from MAARSY horizon direction
    """
    #plt.show()
    #print(mp.shape)
    #exit(0)
    
    #24-29, 51
    #14,45
    #45,55
    cal=n.zeros([n_modules-1,2])
    cals=n.zeros([len(fl),len(cal),2])
    bgs=n.zeros([len(fl),int(n_modules*(n_modules-1)/2),2],dtype=n.complex64)
    for fi,f in enumerate(fl):
        h=h5py.File(f,"r")
        S=h["S"][()]
        if "module_pairs" not in h.keys():
            continue
        mp=h["module_pairs"][()]

        ci=1
        pwr=n.abs(S[0,0,ci,:,:])+n.abs(S[0,1,ci,:,:])
        maarsy_rgi=n.argmax(pwr[51,:])

        for pi in range(S.shape[0]):
            bgs[fi,pi,0]=n.mean(S[pi,0,ci,:,100:200])
            bgs[fi,pi,1]=n.mean(S[pi,1,ci,:,100:200])
    #    n.mean(S[,0,ci,:,100:200])
     #   plt.pcolormesh(10.0*n.log10(pwr))
      #  plt.show()
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


    ho=h5py.File("stocal.h5","w")
    ho["cal_pol0"]=n.angle(n.mean(n.exp(1j*cals[:,:,0]),axis=0))
    ho["cal_pol1"]=n.angle(n.mean(n.exp(1j*cals[:,:,1]),axis=0))
    ho.close()

    plt.subplot(211)
    plt.pcolormesh(n.angle(bgs[:,:,0]),cmap="hsv")
    plt.xlabel("Antenna module")
    plt.title("Pol 0")
    plt.ylabel("Time")
    cb=plt.colorbar()
    cb.set_label("RFI phase")
    plt.subplot(212)
    plt.pcolormesh(n.angle(bgs[:,:,1]),cmap="hsv")
    plt.xlabel("Antenna module")
    plt.title("Pol 1")
    plt.ylabel("Time")
    cb=plt.colorbar()
    cb.set_label("RFI phase")
    plt.show()

    plt.subplot(211)
    plt.pcolormesh(n.abs(bgs[:,:,0]))
    plt.xlabel("Antenna module")
    plt.title("Pol 0")
    plt.ylabel("Time")
    cb=plt.colorbar()
    cb.set_label("RFI power")
    plt.subplot(212)
    plt.pcolormesh(n.abs(bgs[:,:,1]))
    plt.xlabel("Antenna module")
    plt.title("Pol 1")
    plt.ylabel("Time")
    cb=plt.colorbar()
    cb.set_label("RFI power")
    plt.show()



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
    
