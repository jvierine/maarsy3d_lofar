import numpy as n
import matplotlib.pyplot as plt
import os
import glob
import re

def freq(subband_idx):
    """
    http://kaira.sgo.fi/2013/04/converting-between-subband-and-frequency.html
    """
    return(subband_idx*200e6/1024)

class readgdf:
    def __init__(self,dirname,n_per_file=390625):
        self.n_per_file=n_per_file
        self.beamlet_dirs=glob.glob("%s/???"%(dirname))
        self.beamlets=[]
        self.beamlet_x_files=[]
        self.beamlet_y_files=[]
        self.min_sample=[]
        self.max_sample=[]
        for bi,bd in enumerate(self.beamlet_dirs):
            # extract beamlet number
            self.beamlets.append(int(re.search(".*/(...)$",bd).group(1)))
            xf=glob.glob("%s/x/data*gdf"%(bd))
            xf.sort()
            self.beamlet_x_files.append(xf)
            yf=glob.glob("%s/y/data*gdf"%(bd))
            yf.sort()
            self.beamlet_y_files.append(yf)
            self.min_sample.append(0)
            # the last file will be borked
            if len(xf)>0:
                self.max_sample.append((len(xf)-1)*self.n_per_file)
            else:
                self.max_sample.append(0)
            print("beamlet %d 0-%d samples available"%(self.beamlets[bi],self.max_sample[bi]))
        
        print(self.beamlets)
        self.beamlet_idx_to_beamletdir=n.argsort(n.array(self.beamlets))

    def get_bounds(self,beamlet):
        return((self.min_sample[beamlet],self.max_sample[beamlet]))
        
    def read(self,i0,N,beamlet=0):
        """
        read N samples starting at sample i0
        """
        print("read %d samples"%(N))
        xf=self.beamlet_x_files[self.beamlet_idx_to_beamletdir[beamlet]]
        yf=self.beamlet_y_files[self.beamlet_idx_to_beamletdir[beamlet]]

        
        filen=int(n.floor(i0/self.n_per_file))
        print(xf[filen])

        # first sample index
        idx0=i0-int(n.floor(i0/self.n_per_file))*self.n_per_file

        n_left_in_file=self.n_per_file-idx0
        
        print(idx0)
        print(n_left_in_file)

        xout=n.zeros(N,dtype=n.complex64)
        yout=n.zeros(N,dtype=n.complex64)
        
        n_left=N
        outi=0
        while n_left != 0:
            if filen >= len(xf):
                print("out of bounds. no more data left")
                raise Exception
            
            x=n.fromfile(xf[filen],dtype="<i2")
            y=n.fromfile(yf[filen],dtype="<i2")

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
                print("opening new file %s"%(xf[filen]))

                

                            
        return(xout,yout)
        
        

if __name__ == "__main__":
    dirname="/data1/maarsy3d/data-1719568825.7878"
    d=readgdf(dirname)
    print(d.get_bounds(0))
    x,y=d.read(10,1000000)
    plt.subplot(121)
    plt.plot(x.real)
    plt.plot(x.imag)
    plt.subplot(122)
    plt.plot(y.real)
    plt.plot(y.imag)
    plt.show()
    
