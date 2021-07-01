#################################MCKmeans v.1.1###############################################################
#################################Wenshen Song##############################################################
import numpy as np
import matplotlib.pyplot as plt
import sys
import re
import math
import pandas as pd
import random
import os
import time
from mpl_toolkits.basemap import Basemap

from pyspark import SparkContext

global f


def addPoints(p1,p2): #return sum of 2 points
    return (p1[0]+p2[0],p1[1]+p2[1])

def EuclideanDistance(p1,p2): #return the euclidean distance of 2 points using lat-lon coordinates
    return np.linalg.norm(np.array(p1)-np.array(p2))

def GreatCircleDistance(p1,p2): #return great circle distance of 2 points(lat,lon) with R=1
    lat1=p1[0]*math.pi/180.0
    lon1=p1[1]*math.pi/180.0
    lat2=p2[0]*math.pi/180.0
    lon2=p2[1]*math.pi/180.0
    return 2.0*math.asin(np.sqrt(math.sin((lat1-lat2)/2.0)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon1-lon2)/2.0)**2))#haversine formula
     
def closestPoint(p,centers): #update each point p for closest cluster centers with index
    p=np.array(p)
    centers=np.array(centers)
    index=0
    closest=float("+inf")
    for i,c in enumerate(centers):
        if f=="EuclideanDistance":
            dist=EuclideanDistance(p,c)
        elif f=="GreatCircleDistance":
            dist=GreatCircleDistance(p,c)
     
        if dist < closest:
            closest=dist
            index=i

    return index



    

if __name__=="__main__":
    start_time = time.time()
    
    if len(sys.argv) != 5:
        print >> sys.stderr, "Usage: kmeans <file> <k> <convergeDist> <dist_function: e for Euclidean, g for GreatCircle>"
        exit()
    
    sc=SparkContext()
    if sys.argv[4]=="e":
        f="EuclideanDistance"
    elif sys.argv[4]=="g":
        f="GreatCircleDistance"
    else:
        print >> sys.stderr, "wrong choice of distance"
        exit()

    data=sc.textFile(sys.argv[1]).filter(lambda x: len(x)>0).map(lambda x:re.split('[,;\s]\s*',x)).map(lambda x:(float(x[0]),float(x[1])))
    N=data.count()
    data.persist()# read, parse into [lat,lon] and persist the processed data




    #############################SAMPLE RUN################################# 
    N_sample=N/1000 #we take 1/1000 points from data as a sample
    sample_temp=data.takeSample(False,N_sample) 
    sample_data=sc.parallelize(sample_temp) #restore the sample as rdd then persist
    sample_data.persist()
    


    num_MC=50; #number of Monte Carlo steps
    target_f=float("+inf") #target function for clustering, initialize with +inf

    for i in range(num_MC):

        K=int(sys.argv[2])#number of cluster K
        convergeDist=0.0#converge distance
        k_centers=data.takeSample(False,K) #initialize K random centers without replacement and with seed=1
        prior_k_centers=np.array(k_centers)  # initialized k centers before sample kmeans iteration


        plt.figure(1)  
        #m=Basemap(projection='merc',llcrnrlat=28,urcrnrlat=47,llcrnrlon=-126,urcrnrlon=-105, lat_ts=20,resolution='l') #west US
        #m=Basemap(projection='merc',llcrnrlat=18,urcrnrlat=52,llcrnrlon=-131,urcrnrlon=-58, lat_ts=20,resolution='l')  #US
        m=Basemap(projection='merc',llcrnrlat=-70,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180, lat_ts=20,resolution='c')  #worldwide

        m.drawcoastlines(linewidth=0.1, linestyle='solid', color='k')
        #m.drawstates(linewidth=0.1, linestyle='solid', color='k')
        m.drawcountries(linewidth=0.1, linestyle='solid', color='k')
        m.drawmapboundary()
        parallels = np.arange(-80.,80.1,10.)
        m.drawparallels(parallels,labels=[1,0,0,0],fontsize=8,linewidth=0.1, color='grey')
        meridians = np.arange(-180.,180.1,30.)
        m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=8,linewidth=0.1, color='grey')  

        cx, cy = m(prior_k_centers[:,1],prior_k_centers[:,0])
        m.scatter(cx,cy,s=50,marker='X',alpha=0.8)
        name=sys.argv[1].split("/")[-1].split(".")[0]+"_"+f+"_k="+str(K)+"_num_MC="+str(num_MC)+"_prior_k_centers"
        plt.title(name,fontsize=10)
        DIR=os.getcwd()
        plt.savefig(DIR+"/"+name+".png",dpi=600, bbox_inches='tight')




        tempDist=float("+inf") #sum of distance between old centers and new centers, initialize with +inf

        c=0 # kmeans interation number c

        while (tempDist > convergeDist and c<20): #control the iteration of each kmeans under 20
            c=c+1
            closest=sample_data.map(lambda p:(closestPoint(p,k_centers),(np.array(p),1) ) ) #for every point update its index and generate (index,(p,1)) key-value pair
            pstats=closest.reduceByKey(lambda x1,x2: addPoints(x1,x2)) #reduce by key(index), sum the point coordinate and number of points
            new_centers=pstats.map(lambda x: (x[0],x[1][0]/x[1][1])).collect() #for each index calculate the mean of points, which is the new center 
                   
            tempDist=sum(eval(f)(k_centers[ik],p) for (ik, p) in new_centers) # the sum of distance of old centers to new centers
            for (ik, p) in new_centers:
                k_centers[ik]=p # the converged k_centers are used as initial

        posterior_k_centers=np.array(k_centers)  #k centers after sample kmeans iteration 



        point_centers=closest.map(lambda x: (0,eval(f)(x[1][0],k_centers[x[0]])**2) ) #calculate the squared distance of each point to its cluster center
        new_target_f=point_centers.reduceByKey(lambda x1,x2: x1+x2).map(lambda x:x[1]).collect() #sum the squared distance
        new_target_f=np.array(new_target_f[0])
        
        if new_target_f < target_f:
            target_f=new_target_f #save the smallest target_f
            opt_k_centers=np.array(k_centers) # save the k_centers with smallest target_f

 
            

           


        plt.figure(2)
        #m=Basemap(projection='merc',llcrnrlat=28,urcrnrlat=47,llcrnrlon=-126,urcrnrlon=-105, lat_ts=20,resolution='l') #west US
        #m=Basemap(projection='merc',llcrnrlat=18,urcrnrlat=52,llcrnrlon=-131,urcrnrlon=-58, lat_ts=20,resolution='l')  #US
        #m=Basemap(projection='merc',llcrnrlat=-70,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180, lat_ts=20,resolution='c')  #worldwide

        m.drawcoastlines(linewidth=0.1, linestyle='solid', color='k')
        #m.drawstates(linewidth=0.1, linestyle='solid', color='k')
        m.drawcountries(linewidth=0.1, linestyle='solid', color='k')
        m.drawmapboundary()
        parallels = np.arange(-80.,80.1,10.)
        m.drawparallels(parallels,labels=[1,0,0,0],fontsize=8,linewidth=0.1, color='grey')
        meridians = np.arange(-180.,180.1,30.)
        m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=8,linewidth=0.1, color='grey')  



        cx, cy = m(posterior_k_centers[:,1],posterior_k_centers[:,0])
        m.scatter(cx,cy,s=50,marker='X',alpha=0.8)


    ox, oy = m(opt_k_centers[:,1],opt_k_centers[:,0])
    m.scatter(ox,oy,s=80,marker='o',c='red',alpha=1,label="optimized k_centers with targer_f="+str(target_f))      

    name=sys.argv[1].split("/")[-1].split(".")[0]+"_"+f+"_k="+str(K)+"_num_MC="+str(num_MC)+"_posteriror_k_centers"
    plt.title(name,fontsize=10)

    plt.legend(loc="lower left", markerscale=1., fontsize=10) 
    plt.savefig(DIR+"/"+name+".png",dpi=600, bbox_inches='tight')      
    



    ##########################################################################
    sample_time=time.time()










    #############################FULL RUN###################################
    K=int(sys.argv[2])#number of cluster K
    convergeDist=float(sys.argv[3])#converge distance which should be set to 0.1
    tempDist=float("+inf") #sum of distance between old centers and new centers, initialize with +inf
    
    k_centers=opt_k_centers

    while tempDist > convergeDist:
        closest=data.map(lambda p:(closestPoint(p,k_centers),(np.array(p),1) ) ) #for every point update its index and generate (index,(p,1)) key-value pair
        pstats=closest.reduceByKey(lambda x1,x2: addPoints(x1,x2)) #reduce by key(index), sum the point coordinate and number of points
        new_centers=pstats.map(lambda x: (x[0],x[1][0]/x[1][1])).collect() #for each index calculate the mean of points, which is the new center 
               
        tempDist=sum(eval(f)(k_centers[ik],p) for (ik, p) in new_centers) # the sum of distance of old centers to new centers
       
        for (ik, p) in new_centers:
            k_centers[ik]=p


    k_centers=np.array(k_centers)
    final_data = data.map(lambda p:(p[0],p[1],closestPoint(p, k_centers))).collect()#final clustered data with [lat,lon,label]
    snum=450000 #number of points used for plotting
    plot_data = data.map(lambda p:(p[0],p[1],closestPoint(p, k_centers))).takeSample(False,snum,1)#data selected used for plotting

    point_centers=closest.map(lambda x: (0,eval(f)(x[1][0],k_centers[x[0]])**2) ) #calculate the squared distance of each point to its cluster center
    target_f=point_centers.reduceByKey(lambda x1,x2: x1+x2).map(lambda x:x[1]).collect() #sum the squared distance
    target_f=np.array(target_f[0])


    sc.stop()
    end_time=time.time()

    DIR=os.getcwd()
    ftime=open(DIR+"/runtime",'a')
    name=sys.argv[1].split("/")[-1].split(".")[0]+"_"+f+"_k="+str(K)+"_plot_point="+str(snum)+"_cd="+str(convergeDist)
    print>>ftime, name,"    ",round((sample_time - start_time)/3600.,5),round((end_time - sample_time)/3600.,5),round((end_time - start_time)/3600.,5),target_f   
    ftime.close() 


    plot_data=np.array(plot_data)


    plt.figure(3)
    #m=Basemap(projection='merc',llcrnrlat=28,urcrnrlat=47,llcrnrlon=-126,urcrnrlon=-105, lat_ts=20,resolution='l') #west US
    #m=Basemap(projection='merc',llcrnrlat=18,urcrnrlat=52,llcrnrlon=-131,urcrnrlon=-58, lat_ts=20,resolution='l')  #US
    m=Basemap(projection='merc',llcrnrlat=-70,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180, lat_ts=20,resolution='c')  #worldwide

    px, py = m(plot_data[:,1],plot_data[:,0])
    cx, cy = m(k_centers[:,1],k_centers[:,0])

    m.drawcoastlines(linewidth=0.5, linestyle='solid', color='k')
    m.drawstates(linewidth=0.1, linestyle='solid', color='k')
    m.drawcountries(linewidth=0.5, linestyle='solid', color='k')
    m.drawmapboundary()
    parallels = np.arange(-80.,80.1,10.)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=8,linewidth=0.5, color='grey')
    meridians = np.arange(-180.,180.1,30.)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=8,linewidth=0.5, color='grey')
    

    for i in list(set(plot_data[:,2])):
        ci=np.where(plot_data[:,2]==i)
        m.scatter(px[ci],py[ci],s=1,marker='.',label=np.size(ci),alpha=1)
    
    
    m.scatter(cx,cy,s=50,marker='X',color='r',alpha=1)
    name=sys.argv[1].split("/")[-1].split(".")[0]+"_"+f+"_k="+str(K)+"_plot_point="+str(snum)+"_cd="+str(convergeDist)
    
    plt.title(name,fontsize=10)
    plt.legend(loc="lower left", markerscale=10., fontsize=8)
    plt.figtext(0.65,0.2,"target_f="+str(round(target_f,2)),fontsize=10)

    plt.savefig(DIR+"/"+name+".png",dpi=600, bbox_inches='tight')

    plt.show()












