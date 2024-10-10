import numpy as n
import matplotlib.pyplot as plt
import os
import glob
import re
import bz2

def freq(subband_idx):
    """
       http://kaira.sgo.fi/2013/04/converting-between-subband-and-frequency.html
    """
    return(subband_idx*200e6/1024)

class readgdf:
    def __init__(self,dirname):


        
        

        log=open("%s/sampler.log"%(dirname),"r")
        ll=log.readline();ll=log.readline();ll=log.readline();
        utsec=int(log.readline().split(" ")[1])
        extra_samp=int(log.readline().split(" ")[1])
        sr=n.float128(200000000.0)/n.float128(1024.0)
        self.samples_since_1970 = n.int64(n.round(n.float128(utsec)*sr)) + extra_samp
        ll=log.readline()
        self.timestamp=float(ll.split(" ")[1])
        self.debug=False
        if self.debug:
            print("initial timestamp %1.2f (unix) samples since 1970 %d"%(self.timestamp,self.samples_since_1970))

        #self.n_per_file=n_per_file
        # n_per_file=390625
        self.beamlet_dirs=glob.glob("%s/???"%(dirname))
        self.beamlet_dirs.sort()
        self.beamlets=[]
        self.beamlet_x_files=[]
        self.beamlet_y_files=[]
        self.beamlet_x_filetables=[]
        self.beamlet_y_filetables=[]
        self.beamlet_x_filenums=[]
        self.beamlet_y_filenums=[]

        self.sample_rate = 200e6/1024
        
        self.min_sample=[]
        self.max_sample=[]
        self.n_per_file=-1#n_per_file
        for bi,bd in enumerate(self.beamlet_dirs):
            # extract beamlet number
            self.beamlets.append(int(re.search(".*/(...)$",bd).group(1)))
            xf=glob.glob("%s/x/data*gdf*"%(bd))
            xf.sort()

            if self.n_per_file == -1:
                z=self.read_raw(xf[0])
                self.n_per_file=int(len(z)/2)
                print("detected %d samples per file"%(self.n_per_file))
            filetables_x={}
            filenums_x=[]
            for fi in range(len(xf)):
                filenum=int(re.search(".*/data-(......).gdf.*$",xf[fi]).group(1))-1
                filetables_x[filenum]=xf[fi]
#                print("%d %s"%(filenum,xf[fi]))
                filenums_x.append(filenum)
                
            self.beamlet_x_files.append(xf)
            yf=glob.glob("%s/y/data*gdf*"%(bd))            
            yf.sort()
            filetables_y={}
            filenums_y=[]
            for fi in range(len(yf)):
                filenum=int(re.search(".*/data-(......).gdf.*$",yf[fi]).group(1))-1
                filetables_y[filenum]=yf[fi]
                #print("%d %s"%(filenum,yf[fi]))                
                filenums_y.append(filenum)
                
            self.beamlet_x_filetables.append(filetables_x)
            self.beamlet_y_filetables.append(filetables_y)            
            
            self.beamlet_y_files.append(yf)
            
            self.min_sample.append(n.min(filenums_x))
            # the last file will be borked
            if len(xf)>0:
                self.max_sample.append((n.max(filenums_x)-1)*self.n_per_file)
            else:
                self.max_sample.append(0)
            if self.debug:
                print("beamlet %d 0-%d samples available"%(self.beamlets[bi],self.max_sample[bi]))
        if self.debug:
            print(self.beamlets)
        self.beamlet_idx_to_beamletdir=n.argsort(n.array(self.beamlets))

    def get_bounds(self,beamlet):
        return((self.min_sample[beamlet],self.max_sample[beamlet]))
    
    def get_ubounds(self,beamlet):
        return((self.min_sample[beamlet]+self.samples_since_1970,self.max_sample[beamlet]+self.samples_since_1970))

    def readu(self,i0,N,beamlet=0):
        """
        read N samples starting a i0 samples since 1970
        """
        return(self.read(i0-self.samples_since_1970,N,beamlet))

    def read_ubeamlets(self,i0,N,beamlets=[8,9,10]):    
        """
        Read N samples starting a i0 samples since 1970 with multiple beamlets
        """
        return(self.read_beamlets(i0-self.samples_since_1970,N,beamlets))


    def read_raw(self,fname):
        postfix=re.search(".*(.gdf.*)$",fname).group(1)
        if postfix == ".gdf":
            # read raw binary from file
            x=n.fromfile(fname,dtype="<i2")
            return(x)
        elif postfix == ".gdf.bz2":
            # read bzipped files decompressing on the fly
            fh=bz2.BZ2File(fname,"r").read()
            x=n.frombuffer(fh,dtype="<i2")
            return(x)
        else:
            print("file format now known")
            return(None)
    
    def read(self,i0,N,beamlet=0):
        """
        read N samples starting at sample i0
        """
        if self.debug:
            print("read %d samples"%(N))
        xf=self.beamlet_x_filetables[self.beamlet_idx_to_beamletdir[beamlet]]
        yf=self.beamlet_y_filetables[self.beamlet_idx_to_beamletdir[beamlet]]
        
        filen=int(n.floor(i0/self.n_per_file))
        if self.debug:
            print(xf[filen])

        # first sample index
        idx0=i0-int(n.floor(i0/self.n_per_file))*self.n_per_file

        n_left_in_file=self.n_per_file-idx0

        if self.debug:
            print(idx0)
            print(n_left_in_file)

        xout=n.zeros(N,dtype=n.complex64)
        yout=n.zeros(N,dtype=n.complex64)
        
        n_left=N
        outi=0
        while n_left != 0:
            if filen >= len(xf):
                if self.debug:
                    print("out of bounds. no more data left")
                raise Exception

            if filen in xf.keys():
                x=self.read_raw(xf[filen])
            else:
                print("file %d not found!"%(filen))
                x=n.zeros(2*self.n_per_file,dtype="<i2")
            if filen in yf.keys():
                #x=self.read_raw(xf[filen])                
                #y=n.fromfile(yf[filen],dtype="<i2")
                y=self.read_raw(yf[filen])                
            else:
                print("file %d not found!"%(filen))
                y=n.zeros(2*self.n_per_file,dtype="<i2")                
            

            if self.debug:
                print("outi %d %d samples left %d in file"%(outi,n_left,n_left_in_file))
            if n_left_in_file > n_left:
                # there is more left in the file than we need
#                print(len(x[idx0:(idx0+n_left)]))
 #               print(x[idx0:(idx0+n_left)])
                file_idx_re = n.arange(idx0,idx0+n_left)*2
                file_idx_im = n.arange(idx0,idx0+n_left)*2+1
                xout[outi:(outi+n_left)]=x[file_idx_re]+x[file_idx_im]*1j
                yout[outi:(outi+n_left)]=y[file_idx_re]+x[file_idx_im]*1j
                n_left-=n_left
                
            elif n_left_in_file <= n_left:
                if self.debug:
                    print("outi %d idx0 %d"%(outi,idx0))
                # there is more less left in the file than we need
                # read everything
                file_idx_re = n.arange(idx0,self.n_per_file)*2
                file_idx_im = n.arange(idx0,self.n_per_file)*2+1
                n_read=self.n_per_file-idx0
                xout[outi:(outi+n_read)]=x[file_idx_re]+x[file_idx_im]*1j
                yout[outi:(outi+n_read)]=y[file_idx_re]+x[file_idx_im]*1j
                # remove what we have consumed
                n_left-=n_read
                outi+=n_read

                # we start reading at the start of the next file
                idx0=0
                # we have a whole new file
                n_left_in_file=self.n_per_file
                filen+=1
                if self.debug:
                    print("opening new file %s"%(xf[filen]))

        return(xout,yout)
        

    def read_beamlets(self,i0,N,beamlets=[8,9,10]):
        """
        use fft to combine beamlets
        """
        n_beamlets=len(beamlets)
        beamlets=n.array(beamlets,dtype=int)
        n_samples=n_beamlets*N
        x=n.zeros(n_samples,dtype=n.complex64)
        y=n.zeros(n_samples,dtype=n.complex64)

        X=n.zeros([N,n_beamlets],dtype=n.complex64)
        Y=n.zeros([N,n_beamlets],dtype=n.complex64)

        for i in range(n_beamlets):
            X[:,i],Y[:,i]=self.read(i0,N,beamlet=beamlets[i])
        XX=n.fft.ifft(X,axis=1)
        YY=n.fft.ifft(Y,axis=1)
        xout=XX.reshape((n_samples,))
        yout=YY.reshape((n_samples,))
        return(xout,yout)

if __name__ == "__main__":
    dirname="/data1/maarsy3d/imaging/data-1719924302.3445"
    d=readgdf(dirname)
    print(d.get_bounds(0))
    x,y=d.read_beamlets(10,1000000,beamlets=[0])
    plt.subplot(121)
    plt.plot(x.real)
    plt.plot(x.imag)
    plt.subplot(122)
    plt.plot(y.real)
    plt.plot(y.imag)
    plt.show()
    
