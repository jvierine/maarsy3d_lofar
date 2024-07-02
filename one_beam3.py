import numpy as n
import matplotlib.pyplot as plt
import readgdf
import scipy.interpolate as sint
import h5py
import stuffr

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("comm rank %d size %d"%(rank,size))


print(readgdf.freq(n.arange(265,284))/1e6)

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


def resample(z,sr_in=3*195312.5,sr_out=500e3):
    tin=n.arange(len(z))/sr_in
    n_in=len(z)
    T_max=n_in/sr_in
    n_out=int(n.floor(T_max*sr_out))
    
    tin=n.arange(n_in)/sr_in
    tout=n.arange(n_out)/sr_out
    zfun=sint.interp1d(tin,z)
    return(zfun(tout))


if __name__ == "__main__":
    #% 48°/60°  Gr1 - Gr8 combined
    #
    #beamctl --antennaset=LBA_INNER --band=30_90 --rcus=0:95 --subbands=265:284 --beamlets=0:19 --digdir=1.0472,0.96,AZELGEO &
#    beamlets=[10,8,9]


    codes=get_codes(interp=0,plot_filtered_code=True)
    codes_f=[]
    codes_f.append(n.conj(n.fft.fft(codes[0],500)))
    codes_f.append(n.conj(n.fft.fft(codes[1],500)))
    
    beamlets=[9,10,8]    
    dirname="/data1/maarsy3d/data-1719568825.7878"
    d=readgdf.readgdf(dirname)
    b=d.get_bounds(9)

    ipp=1e-3
    n_beamlets=len(beamlets)
    sr=200e6/1024

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

    # interpolated ipp
    n_rgo=500

    ut0=1719568825.962887763977

    ipp=1e-3
    for ti in range(rank,n_timesteps,size):
        this_ut0=ti*n_ipp*ipp*n_frames + ut0
        S=n.zeros([2,2,n_ipph,n_rgo],dtype=n.float32)

        for fi in range(n_frames):
            X=n.zeros([2,2,n_ipph,n_rgo],dtype=n.complex64)
            fi0=int(n.round(fi*n_ipp*samples_per_ipp+ti*n_frames*n_ipp*samples_per_ipp))

            # read all ipps
            n_read=int(n.ceil(n_ipp*samples_per_ipp))+10

            tv=n.arange(n_beamlets*n_read)/sr/n_beamlets
            # shift frequency
            csin=n.exp(1j*2*n.pi*tv*dfreq)
            x,y=d.read_beamlets(fi0,n_read,beamlets=beamlets)

            x=resample(x*csin)
            y=resample(y*csin)

            i0=0.0

            codei=0
            for i in range(int(n_ipph)):
                for j in range(2):
                    X[0,0,i,:]+=n.fft.ifft(codes_f[codei%2]*n.fft.fft(x[int(i0):(int(i0)+n_rgo)]))
                    X[1,0,i,:]+=n.fft.ifft(codes_f[codei%2]*n.fft.fft(y[int(i0):(int(i0)+n_rgo)]))
                    X[0,1,i,:]+=n.fft.ifft(codes_f[(codei+1)%2]*n.fft.fft(x[int(i0):(int(i0)+n_rgo)])) 
                    X[1,1,i,:]+=n.fft.ifft(codes_f[(codei+1)%2]*n.fft.fft(y[int(i0):(int(i0)+n_rgo)]))
                    codei+=1
                    i0+=n_rgo

            for i in range(n_rgo):
                for poli in range(2):
                    for codi in range(2):
                        S[poli,codi,:,i]+=n.abs(n.fft.fftshift(n.fft.fft(X[poli,codi,:,i])))**2.0

        snr=n.copy(S)
        nfloorx=n.nanmedian(snr[0,:,:,:])
        nfloory=n.nanmedian(snr[1,:,:,:])    
        snr[0,:,:]=(snr[0,:,:]-nfloorx)/nfloorx
        snr[1,:,:]=(snr[1,:,:]-nfloory)/nfloory    

        dB=10.0*n.log10(snr)

        db0,db1=n.nanpercentile(dB.flatten(),[5,99])
        dB[snr<=0]=-60

        plt.figure(figsize=(2*8,2*6.4))
        plt.subplot(221)
        plt.pcolormesh(dops,n.arange(n_rgo)*0.6,dB[0,0,:,:].T,vmin=db0)
        cb=plt.colorbar()
        cb.set_label("SNR (dB)")
        plt.xlim([-50,50])
        plt.title("X Not aliased (nfloorx/nfloory=%1.2g)"%(nfloorx/nfloory))
        plt.xlabel("Doppler (Hz)")
        plt.ylabel("Total range (km)")

        plt.subplot(222)
        plt.pcolormesh(dops,n.arange(n_rgo)*0.6,dB[1,0,:,:].T,vmin=db0)
        cb=plt.colorbar()
        cb.set_label("SNR (dB)")
        plt.xlim([-50,50])
        plt.title("Y Not aliased %s"%(stuffr.unix2datestr(this_ut0)))
        plt.xlabel("Doppler (Hz)")
        plt.ylabel("Total range (km)")

        plt.subplot(223)
        plt.pcolormesh(dops,n.arange(n_rgo)*0.6,dB[0,1,:,:].T,vmin=db0)
        cb=plt.colorbar()
        cb.set_label("SNR (dB)")
        plt.title("X Range aliased")
        plt.xlim([-50,50])    
        plt.xlabel("Doppler (Hz)")
        plt.ylabel("Total range (km)")

        plt.subplot(224)
        plt.pcolormesh(dops,n.arange(n_rgo)*0.6,dB[1,1,:,:].T,vmin=db0)
        cb=plt.colorbar()
        cb.set_label("SNR (dB)")
        plt.title("Y Range aliased")
        plt.xlim([-50,50])    
        plt.xlabel("Doppler (Hz)")
        plt.ylabel("Total range (km)")

        plt.tight_layout()
        plt.savefig("maarsy3d_decoded-%05d.png"%(ti),dpi=150)
        plt.close()

        ho=h5py.File("spec-%06d.h5"%(ti),"w")
        ho["ti"]=ti
        ho["S"]=S
        ho["t0"]=this_ut0
        ho["dops"]=dops
        ho["n_rgo"]=n_rgo
        ho.close()
        
    

