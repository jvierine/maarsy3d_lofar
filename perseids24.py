import numpy as n
import matplotlib.pyplot as plt
import readgdf
import scipy.interpolate as sint
import h5py
import stuffr
import scipy.signal as ss
import itertools

#
# Analyze LOFAR measurements of MAARSY using the 16-bit complementary code,
# 1 ms IPP and 2 us bit length (meso17)
#

from mpi4py import MPI

# use MPI to paralellize the processing steps
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("comm rank %d size %d"%(rank,size))

# the frequency shift needed to center the signal to DC
dfreq=(53.515625-53.5)*1e6

def get_codes(interp=0,plot_filtered_code=False):
    """
    16-bit complementary code pair used at MAARSY
    Possibly interpolate the code if interp>1
    when interpolating, the code will be low pass filtered and 
    shifted by interp samples forward to take into account 
    the filter impulse response.
    see with plot_filtered_code=True
    """
    code_strings=["1000110110000010","0100000101001110"]
    codes=[]
    code=[]
    for i in range(16):
        a=int(code_strings[0][i])
        if(a == 0):
            a=-1.0
        else:
            a=1.0
        code.append(a)
    code=n.array(code,dtype=n.complex64)
    codes.append(code)
    code=[]
    for i in range(16):
        a=int(code_strings[1][i])
        if(a == 0):
            a=-1.0
        else:
            a=1.0
        code.append(a)
    code=n.array(code,dtype=n.complex64)        
    codes.append(code)
    code_len=len(code)
    
    if interp > 1:
        # length of new code
        N=interp*len(code)
        
        # cutoff frequency
        omega_c = 2*(n.pi/interp)
        m = n.arange(N)-N/2.0 + 1e-6

        # impulse response of LPF
        lpf = (n.sin(omega_c*m)/m)*ss.hann(len(m))
        #plt.plot(lpf)
        #plt.show()
        LPF=n.fft.fft(lpf,N)
        interpolated_codes=[]
        for c in codes:
            interpolated_code=n.zeros(N,dtype=n.complex64)
            for i in range(len(c)):
                interpolated_code[(i*interp):(i*interp+interp)]=c[i]
#            interpolated_code2=n.roll(n.fft.ifft(n.fft.fft(interpolated_code)*LPF),-int(N/2))
            interpolated_code2=interpolated_code
 #           interpolated_code2=n.roll(n.fft.ifft(n.fft.fft(interpolated_code)*LPF),-int(N/2))            
            if plot_filtered_code:
                plt.plot(interpolated_code.real,".")
                plt.plot(interpolated_code2.real,".")            
                plt.show()
            interpolated_codes.append(interpolated_code2)
        codes=interpolated_codes
    return(codes)

def resample(z,sr_in=3*195312.5,sr_out=0.5e6):
    """
    resample vector from sr_in to sr_out
    """
    tin=n.arange(len(z))/sr_in
    n_in=len(z)
    T_max=n_in/sr_in
    n_out=int(n.floor(T_max*sr_out))

    # cutoff frequency
    fc=0.8*sr_out/2.0
    
    omega_c = 2*n.pi*fc/sr_in
    w = 20
    m=n.arange(-10,10)+1e-6
    # impulse response of LPF
    lpf = (n.sin(omega_c*m)/m)*ss.hann(len(m))
    if False:
        freqs=n.fft.fftshift(n.fft.fftfreq(len(m),d=1/sr_in))
        plt.plot(freqs,10.0*n.log10(n.abs(n.fft.fftshift(n.fft.fft(lpf)))**2.0))
        plt.show()
    zo=n.convolve(z,lpf,mode="same")
 #   LPF=n.fft.fft(lpf,N)
    
    
    tin=n.arange(n_in)/sr_in
    tout=n.arange(n_out)/sr_out
    zfun=sint.interp1d(tin,zo)
    return(zfun(tout))


def process_dir(dirname="/data1/maarsy3d/imaging/data-1719924302.3445",beamlets=[[1,2,0],
                                                                                 [4,5,3],
                                                                                 [7,8,6],
                                                                                 [10,11,9],
                                                                                 [13,14,12],
                                                                                 [16,17,15],
                                                                                 [19,20,18]]):
                                                                    

    beamlets=n.array(beamlets)
    n_modules=beamlets.shape[0]
    n_beamlets=beamlets.shape[1]
    #% 48°/60°  Gr1 - Gr8 combined
    #
    #beamctl --antennaset=LBA_INNER --band=30_90 --rcus=0:95 --subbands=265:284 --beamlets=0:19 --digdir=1.0472,0.96,AZELGEO &
#    beamlets=[10,8,9]

    # number of samples per ipp in the resampled data
    n_rgo=500
    rg=0.3

    codes=get_codes(interp=0,plot_filtered_code=False)
    codes_f=[]
    codes_f.append(n.conj(n.fft.fft(codes[0],n_rgo)))
    codes_f.append(n.conj(n.fft.fft(codes[1],n_rgo)))

#    n_modules=len(beamlets)
    # number of cross-spectra
    n_xspec = int(n_modules*(n_modules-1)/2)
    
    module_pairs=n.array(list(itertools.combinations(n.arange(n_modules),2)))
    
    d=readgdf.readgdf(dirname)
    b=d.get_ubounds(beamlets[0,0])

    ipp=1e-3
#    n_beamlets=len(beamlets)
    sr=200e6/1024
    sr3=sr*3
    samples_per_ipp=ipp*sr
    n_rg=int(n.floor(n_beamlets*samples_per_ipp))

    n_ipp=1024
    n_ipph=512
    n_frames=20

    # 2*f*v/3e8 = df
    dops=n.fft.fftshift(n.fft.fftfreq(n_ipph,d=2*1e-3))
    # use only these indices
    fidx=n.where(n.abs(dops)<50)[0]

    n_timesteps=int(n.floor((b[1]-b[0])/(n_ipp*n_frames*samples_per_ipp)))

    #ut0=1719568825.962887763977
    # start at next full second
    ut0 = n.ceil(d.timestamp)
    sample0= int(ut0*d.sample_rate)

    ipp=1e-3
    for ti in range(rank,n_timesteps,size):
        this_ut0=ti*n_ipp*ipp*n_frames + ut0
        S=n.zeros([n_xspec,2,2,n_ipph,n_rgo],dtype=n.complex64)

        X=n.zeros([n_modules,2,2,n_ipph,n_rgo],dtype=n.complex64)
        for fi in range(n_frames):
            fi0=int(n.round(fi*n_ipp*samples_per_ipp+ti*n_frames*n_ipp*samples_per_ipp))
            for mi in range(n_modules):
                print("frame %d module %d"%(fi,mi))
                # read all ipps
                n_read=int(n.ceil(n_ipp*samples_per_ipp))+10

                tv=n.arange(n_beamlets*n_read)/sr/n_beamlets
                # shift frequency
                csin=n.exp(1j*2*n.pi*tv*dfreq)
                x,y=d.read_ubeamlets(sample0+fi0,n_read,beamlets=beamlets[mi])
#                print(n_read)
 #               print(n_beamlets)
  #              print(len(tv))
   #             print(len(x))

                x=resample(x*csin)
                y=resample(y*csin)

                i0=0.0

                codei=0
                # number of ipps
                for i in range(int(n_ipph)):
                    # complementary code decode by summing
                    for j in range(2):
                        # range aliased and not range aliased
                        X[mi,0,0,i,:]+=n.fft.ifft(codes_f[codei%2]*n.fft.fft(x[int(i0):(int(i0)+n_rgo)]))
                        X[mi,1,0,i,:]+=n.fft.ifft(codes_f[codei%2]*n.fft.fft(y[int(i0):(int(i0)+n_rgo)]))
                        X[mi,0,1,i,:]+=n.fft.ifft(codes_f[(codei+1)%2]*n.fft.fft(x[int(i0):(int(i0)+n_rgo)])) 
                        X[mi,1,1,i,:]+=n.fft.ifft(codes_f[(codei+1)%2]*n.fft.fft(y[int(i0):(int(i0)+n_rgo)]))
                        codei+=1
                        i0+=n_rgo

            for xi in range(n_xspec):
                for i in range(n_rgo):
                    for poli in range(2):
                        for codi in range(2):
                            S[xi,poli,codi,:,i]+=n.fft.fftshift(n.fft.fft(X[module_pairs[xi][0],poli,codi,:,i]))*n.conj(n.fft.fftshift(n.fft.fft(X[module_pairs[xi][1],poli,codi,:,i])))


        for xi in range(n_xspec):
            plt.figure(figsize=(12,12))
            plt.subplot(221)
            an0=0.0#n.median(S[xi,0,0,:,:].flatten())
            an1=0.0#n.median(S[xi,1,0,:,:].flatten())
            print(an0)
            print(an1)            
            #            dops,
            rgv=n.arange(n_rgo)*rg + 40 + 150
            plt.pcolormesh(dops,rgv,n.transpose(n.angle( (S[xi,0,0,:,:]-an0) + (S[xi,1,0,:,:]-an1))),cmap="hsv")
            plt.xlim([-50,50])

            plt.xlabel("Doppler (Hz)")
            plt.ylabel("Total range (km)")            
            plt.title("%s %d-%d"%(stuffr.unix2datestr(this_ut0),module_pairs[xi][0],module_pairs[xi][1]))
            plt.colorbar()            
            plt.subplot(222)
            an0=0.0#n.median(S[xi,0,1,:,:].flatten())
            an1=0.0#n.median(S[xi,1,1,:,:].flatten())
            
#            an0=n.median(n.mean(S[xi,0,1,:,:],axis=1))
 #           an1=n.median(n.mean(S[xi,1,1,:,:],axis=1))
            print(an0)
            print(an1)            
            #            dops,
            rgv=n.arange(n_rgo)*rg  + 40#+ 150
            plt.pcolormesh(dops,rgv,n.transpose(n.angle( (S[xi,0,1,:,:]-an0) + (S[xi,1,1,:,:]-an1))),cmap="hsv")
            plt.xlim([-50,50])
            plt.xlabel("Doppler (Hz)")
            plt.ylabel("Total range (km)")            
#            plt.title("%s %d-%d"%(stuffr.unix2datestr(this_ut0),module_pairs[xi][0],module_pairs[xi][1]))
            plt.colorbar()            
            plt.subplot(223)
            dB=10.0*n.log10(n.transpose(n.abs(S[xi,0,0,:,:] + S[xi,1,0,:,:])))
            
            dB=dB-n.median(dB)
            rgv=n.arange(n_rgo)*rg + 40 + 150
            plt.pcolormesh(dops,rgv,dB,vmin=-3,vmax=30,cmap="turbo")
            plt.xlim([-50,50])            
            plt.xlabel("Doppler (Hz)")
            plt.ylabel("Total range (km)")
            cb=plt.colorbar()
            cb.set_label("dB")
            plt.subplot(224)
            dB=10.0*n.log10(n.transpose(n.abs(S[xi,0,1,:,:] + S[xi,1,1,:,:])))
            
            dB=dB-n.median(dB)
            rgv=n.arange(n_rgo)*rg + 40
            plt.pcolormesh(dops,rgv,dB,vmin=-3,vmax=30,cmap="turbo")
            plt.xlim([-50,50])            
            plt.xlabel("Doppler (Hz)")
            plt.ylabel("Total range (km)")
            cb=plt.colorbar()
            cb.set_label("dB")
            plt.tight_layout()
            plt.savefig("maarsy3d_%03d_%06d.png"%(xi,ti),dpi=150)
            plt.close()


        ho=h5py.File("spec_%06d.h5"%(ti),"w")
        ho["ti"]=ti
        ho["S"]=S
        ho["t0"]=this_ut0
        ho["dops"]=dops
        ho["n_rgo"]=n_rgo
        ho["beamlets"]=beamlets
        ho.close()


def explore_data():
    import glob
    dirlist=glob.glob("/data2/perseids2024/data-*")
    dirlist.sort()
    for d in dirlist[0:1]:
        print(d)
        try:
            d=readgdf.readgdf(d)
            b=d.get_ubounds(0)
            print("Data coverage %s to %s"%(stuffr.unix2datestr(b[0]/d.sample_rate),stuffr.unix2datestr(b[1]/d.sample_rate)))
            step=1000000
            for i in range(int(n.floor(b[1]-b[0]-4)/step)):
                x,y=d.read_ubeamlets(b[0]+i*step,4,[0])
                print(x.real)
                print(x.imag)                

        except:
            import traceback as tb
            tb.print_exc()
            print("bad dir %s"%(d))

    

if __name__ == "__main__":
    import glob    
#    explore_data()
    dirlist=glob.glob("/data2/perseids2024/data-*")
    dirlist.sort()
    for d in dirlist:
        process_dir(d)
    

