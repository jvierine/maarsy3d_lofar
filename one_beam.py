import numpy as n
import matplotlib.pyplot as plt
import readgdf

print(readgdf.freq(n.arange(265,284))/1e6)

dfreq=(53.515625-53.5)*1e6

if __name__ == "__main__":
    #% 48°/60°  Gr1 - Gr8 combined
    #
    #beamctl --antennaset=LBA_INNER --band=30_90 --rcus=0:95 --subbands=265:284 --beamlets=0:19 --digdir=1.0472,0.96,AZELGEO &
    beamlets=[8,9,10]
    dirname="/data1/maarsy3d/data-1719568825.7878"
    d=readgdf.readgdf(dirname)
    b=d.get_bounds(9)

    ipp=1e-3
    sr=200e6/1024

    samples_per_ipp=ipp*sr
    print(samples_per_ipp)
    n_rg=int(n.floor(samples_per_ipp))

    n_ipp=512
    n_frames=100

    PI=n.zeros([n_frames*n_ipp,n_rg],dtype=n.float32)
    pidx=0

    n_spectra=int((b[1]-b[0])/(n_ipp*n_frames*samples_per_ipp))
    print(n_spectra)

    # 2*f*v/3e8 = df
    vels=3e8*n.fft.fftshift(n.fft.fftfreq(n_ipp,d=1e-3))/2/53.5e6
    
    for si in range(n_spectra):
        X=n.zeros([n_ipp,n_rg],dtype=n.complex64)
        Y=n.zeros([n_ipp,n_rg],dtype=n.complex64)
        SX=n.zeros([n_ipp,n_rg],dtype=n.float32)
        SY=n.zeros([n_ipp,n_rg],dtype=n.float32)

        for fi in range(n_frames):
            fi0=int(n.round(fi*n_ipp*samples_per_ipp + si*n_ipp*samples_per_ipp*n_frames))

            # read all ipps
            n_read=int(n.ceil(n_ipp*samples_per_ipp))
            tv=n.arange(n_read)/sr
            # shift frequency
            csin=n.exp(1j*2*n.pi*tv*dfreq)
            x,y=d.read(fi0,n_read,beamlet=9)

            x=x*csin
            y=y*csin

            i0=0.0
            for i in range(n_ipp):
                X[i,:]=x[int(i0):(int(i0)+n_rg)]
                Y[i,:]=y[int(i0):(int(i0)+n_rg)]
                PI[pidx,:]=n.abs(x[int(i0):(int(i0)+n_rg)])**2.0+n.abs(y[int(i0):(int(i0)+n_rg)])**2.0
                pidx+=1
                i0+=samples_per_ipp        

            for i in range(n_rg):
                SX[:,i]+=n.abs(n.fft.fftshift(n.fft.fft(X[:,i])))**2.0
                SY[:,i]+=n.abs(n.fft.fftshift(n.fft.fft(Y[:,i])))**2.0

            P=n.abs(X)**2+n.abs(Y)**2.0

            if False:
                dB=10.0*n.log10(P)
                db0,db1=n.percentile(dB.flatten(),[1,90])
                plt.pcolormesh(n.arange(n_ipp),n.arange(n_rg),dB.T,vmin=db0,vmax=db1)
                plt.xlabel("Time (IPP)")
                plt.ylabel("Range gate")
                plt.colorbar()
                plt.show()

        dB=10.0*n.log10(SX+SY)
        db0,db1=n.percentile(dB.flatten(),[1,99])
        plt.figure(figsize=(2*8,6.4))
        plt.subplot(121)
        plt.pcolormesh(vels,n.arange(n_rg),dB.T)
        plt.xlabel("Doppler (m/s)")
        plt.ylabel("Range gate")
        plt.colorbar()
        plt.subplot(122)
        dB=10.0*n.log10(PI)
        db0,db1=n.percentile(dB.flatten(),[1,90])
        plt.pcolormesh(dB.T,vmin=db0,vmax=db1)
        plt.xlabel("Time (IPP)")
        plt.ylabel("Range gate")
        plt.colorbar()
        plt.show()
        
    

