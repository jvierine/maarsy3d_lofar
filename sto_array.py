import numpy as n
import matplotlib.pyplot as plt

def get_antenna_xyz():
    c=n.genfromtxt("memos/002-positions.csv",delimiter=",")
    xyz=n.zeros([8,3])
    latlon=n.zeros([8,3])    
    xyza=n.zeros([48,3])    
    for li in range(c.shape[0]):
        #    print(c[li])
        if n.isnan(c[li,0]) == False:
#            print(c[li,:])
            ai=int(c[li,0])
            x=c[li,2]
            y=c[li,3]
            z=n.mean(c[(li+1):(li+7),4])
            xyz[ai-1,:]=[x,y,z]
            latlon[ai-1,0]=c[li,5]
            latlon[ai-1,1]=c[li,6]
            latlon[ai-1,2]=z
        if n.isnan(c[li,1]) == False:
            ai=int(c[li,1])
            xyza[ai-1,0]=c[li,2]
            xyza[ai-1,1]=c[li,3]
            xyza[ai-1,2]=c[li,4]
            print(ai)
            print(c[li,2],c[li,3],c[li,4])

            
            #        print(c[li,0])
    # local enu, array enu, lat-lon-height (WGS84)
    return(xyz,xyza,latlon)


if __name__ == "__main__":

    xyz,xyza,latlon=get_antenna_xyz()
    #plt.subplot(121)
    plt.plot(xyz[:,0],xyz[:,1],".")
    plt.plot(xyza[:,0],xyza[:,1],".")
    for i in range(xyz.shape[0]):
        plt.text(xyz[i,0],xyz[i,1]+7,"%d"%(i+1))
    plt.xlabel("East-West (m)")
    plt.ylabel("North-South (m)")

    #plt.subplot(122)
    #plt.plot(latlon[:,1],latlon[:,0],".")
    plt.show()
