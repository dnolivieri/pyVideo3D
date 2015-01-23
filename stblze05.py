#!/usr/bin/env python

#   stblze05.py:
##     dnolivieri.net: (updated 15-dic-2014)
#
#      does stabilization in one plane.
#

"""
  Notes...
     ordering has been a pain in the neck!

     maxLoc has the form (x,y), 
     The matrices for the images have (rows, cols), which is (y,x), 
     So be careful in doing the element arithmetic.
   
     * rectangle also uses the (x,y) which is opposite of the 
       image ordering...

For making videos later, use this...
  ffmpeg -r 10 -b 1800 -i %03d.jpg test1800.mp4
"""
import sys
import cv2.cv as cv
import cv2
import random
import numpy as np
from operator import itemgetter, attrgetter
import time
import math
import os
import os.path
import glob
import re
from numpy.fft import ifft,fftshift
import fftw3
import fftw3f
import fftw3l
import fftw3.lib
import fftw3l.lib
import fftw3f.lib
import fftw3.planning
import fftw3l.planning
import fftw3f.planning
import scipy.fftpack as fft

import dfsFiles as dF


def scipyFFT(ybar):
    # Now take FFT
    yF =np.abs(fft.fft2(ybar))/ybar.size
    yF = fft.fftshift( yF )
    return yF


def fastFFTw(ybar):
    #print ybar.shape, ybar.shape[0], ybar.shape[1]/2+1
    b = np.zeros((ybar.shape[0],ybar.shape[1]/2+1), dtype=np.typeDict['singlecomplex'])
    a = np.asarray(ybar, dtype=np.float32)
    p = fftw3f.Plan(a,b,'forward')
    p()
    #print  np.size(b)
    yF=b/np.prod(a.shape)

    #yF = np.abs(fftshift(yF))
    #print yF[:,1]
    return yF



def phase_correlation(x1, x2): 
    
    # obtain Numpy matrices of the images. 
    y1 = np.asarray(x1, dtype=np.float32)
    y2 = np.asarray(x2, dtype=np.float32)

    #print y1.shape, y2.shape

    # obtain the FFT of img1 and img2 (using np)
    """
    z1 = scipyFFT( y1 );
    z2 = scipyFFT( y2 );
    """

    fmethod=0
    if (fmethod==0):
        z1 = fft.fft2( y1 );
        z2 = fft.fft2( y2 );
    elif (fmethod==1):
        z1 = fastFFTw(y1)
        z2 = fastFFTw(y2)


    # obtain the cross power spectrum */
    res = np.zeros( z1.shape, complex)
    res.real = (z2.real * z1.real) - (z2.imag * (-z1.imag))
    res.imag = (z2.real * (-z1.imag)) + (z2.imag * z1.real)
    tmp = np.abs ( res )
    res.real /= tmp 
    res.imag /= tmp
  
    #/* obtain the phase correlation array */
    res = 10*np.abs(  fft.ifft2( res )  )

    return res


def MaxLoc():
    pass

def formCompositeImage(z1, z2, maxLoc):
    """
     quadrant 1:  x [ 0, c/2], y [r, r/2]
     quadrant 2:  x [ c, c/2], y [r, r/2]
     quadrant 3:  x [ c, c/2], y [0, r/2]
     quadrant 4:  x [ 0, c/2], y [0, r/2]
    """

    print "z1.rows=",z1.rows
    print "z1.cols=",z1.cols
    T = (z1.rows + z2.rows, z1.cols + z2.cols)
    M = np.zeros( T )

    r = z1.rows
    c = z1.cols
    x = maxLoc[0]
    y = maxLoc[1]

    print x, c/2
    if (x<=c/2 ) and (y <r) and (y>=r/2): 
        print "quadrant 1"
        rTop = z1.rows - maxLoc[1]
        rBot = rTop + z1.rows
        cLeft = maxLoc[0] 
        cRight= maxLoc[0] + z1.cols
        z1prm = np.asarray(z1, dtype=np.float32) 
        M[ rTop:rBot, cLeft: cRight ] = z1
        #M[0:z2.rows,0:z2.cols] = z2
        M[0:z2.rows,z1.cols:z1.cols+z2.cols] = z2

    elif (x<=c) and (x>=c/2) and  (y<=r) and (y>=r/2):
        print "quandrant 2"
        rTop = z1.rows - maxLoc[1]
        rBot = rTop + z1.rows
        cLeft = z1.cols - maxLoc[0] 
        cRight= cLeft + z1.cols

        z1prm = np.asarray(z1, dtype=np.float32) 
        M[ rTop:rBot, cLeft: cRight ] = z1
        M[0:z2.rows,0:z2.cols] = z2

    elif (x<=c) and (x>=c/2) and (y<=r/2):
        print "quadrant 3"
        rTop = maxLoc[1]
        rBot = rTop + z1.rows
        cLeft = z1.cols - maxLoc[0] 
        cRight= cstabiLeft + z1.cols

        z1prm = np.asarray(z1, dtype=np.float32) 
        M[ rTop:rBot, cLeft: cRight ] = z1
        M[0:z2.rows,0:z2.cols] = z2

    elif (x<=c/2) and (y<=r/2):
        print "quadrant 4"
        rTop = maxLoc[1]
        rBot = rTop + z1.rows
        cLeft = maxLoc[0] 
        cRight= maxLoc[0] + z1.cols
        z1prm = np.asarray(z1, dtype=np.float32) 
        M[0:z2.rows,0:z2.cols] = z2
        M[ rTop:rBot, cLeft: cRight ] = z1        

    else:
        print "fuck off"


    q = cv.fromarray( np.ascontiguousarray(M) )
    cv.ConvertScale(q, q,1/255.0) 
    return q
    

def getMaxLoc(z1, z2):
    #phase locatation
    p = cv.CreateMat( z1.rows, z1.cols, cv.CV_32FC1)
    p = phase_correlation( z1, z2)
    pm = cv.CreateMat( p.shape[0], p.shape[1], cv.CV_32FC1)

    pm = cv.fromarray( np.ascontiguousarray(p) )
    minVal,maxVal,minLoc,maxLoc = cv.MinMaxLoc(pm) 
    #print "maxLoc=", maxLoc

    r = z1.rows
    c = z1.cols
    x = maxLoc[0]
    y = maxLoc[1]
    quadrant=0

    xp=0
    yp=0
    #print x, c/2
    if (x<=c/2 ) and (y <r) and (y>=r/2): 
        #print "quadrant 1"
        quadrant=1
        xp = x
        yp = y-r
    elif (x<=c) and (x>=c/2) and  (y<=r) and (y>=r/2):
        #print "quandrant 2"
        quandrant=2
        xp = x-c
        yp = y-r
    elif (x<=c) and (x>=c/2) and (y<=r/2):
        #print "quadrant 3"
        quadrant=3
        xp=x-c
        yp=y
    elif (x<=c/2) and (y<=r/2):
        #print "quadrant 4"
        quadrant=4
        xp = x
        yp = y
    else:
        print "fuck off"

    print quadrant, maxLoc[0], maxLoc[1], xp, yp
    zcorr=(xp,yp)
    #cv.ShowImage("z1",z1)
    #cv.ShowImage("z2",z2)
    #cv.Circle(pm, maxLoc, 5, (255,0,0),2)
    #cv.ShowImage("p", pm)


    #pt1=(rTop,cLeft)
    #pt2=(rBot,cRight)
    #cv.Rectangle(q, pt1, pt2, 255)
    #q=formCompositeImage(z1, z2, maxLoc)
    #cv.ShowImage("q", q)


    #cv.WaitKey()
    return zcorr




def WriteImFile(resdir, i, z):
    prnt="z"
    """
    if (i<10):
        zbar=prnt+"00"+str(i)+".jpg"
    else:
        zbar=prnt+"0"+str(i)+".jpg"            
    """
    if (i<10):
        zbar=prnt+"00"+str(i)+".tif"
    else:
        zbar=prnt+"0"+str(i)+".tif"            
        
    zx=cv.CreateMat( z.rows, z.cols, cv.CV_8UC3   )
    cv.ConvertScale( z,zx, 255)
    # change the size...
    s = cv.CreateImage( (q.cols, q.rows), 8, 3 )       
    sbar = cv.GetSubRect(zx, (0,0,q.cols,q.rows) )
    cv.SetImageROI(s,(0,0,q.cols,q.rows))
    cv.Copy(sbar, s)
    cv.SaveImage(resdir+"/"+zbar, zx)
    #cv.imwrite(resdir+"/"+zbar, s)


def OutputVideo(resdir):

    for subdir in dF.DFS(resdir):
        imgs = glob.glob(os.path.join(subdir, "*.tif"))
        imgs.sort()


    ## input video
    fps=3
    z1 = cv.LoadImageM( imgs[0] ) 
    #vidfourcc = cv.CV_FOURCC('M','J','P','G') 
    vidfourcc = cv.CV_FOURCC('X','V','I','D') 
    frame_size=cv.GetSize( z1 )
    cwriter = cv.CreateVideoWriter ("cpout.avi", int(vidfourcc), fps, frame_size, True)
    d=cv.CreateImage( cv.GetSize(z1), 8,3)
    cv.Copy(z1,d)
    cv.WriteFrame(iwriter, d)    

    for i in range(1,len(imgs)):
        z2 = cv.LoadImageM( imgs[i] )
        z1 = z2
        cv.Copy(z1,d)
        cv.WriteFrame(cwriter, d)    



def getzcorrcalc():
    #zcorr=[(0, 0), (4, 1), (-3, 1), (-1, 4), (3, 3), (3, 0), (-4, 3), (-7, 4), (-2, 2), (0, 0), (5, 0), (-6, 2), (-5, 1), (-3, 1), (2, 1), (-2, 2), (2, 4), (-2, 3), (-6, 6), (-2, 0), (8, 3), (4, 3), (-4, 4), (-2, 0), (0, -1), (3, 0), (7, 0), (-2, 1), (-2, 0), (-1, 0), (1, 0), (5, -1), (2, 2), (-4, 1), (-3, 0), (1, 0), (2, 0), (5, -2), (0, 1), (-3, 0), (-1, 0), (3, -1), (5, -1), (9, -1), (2, 0), (3, -3), (0, -3), (2, -2), (-1, -2), (1, -4), (4, -3), (5, -3), (5, -3), (6, -2), (-3, -3), (-2, -2), (3, -3), (5, -2), (4, -2), (3, 0)]
    pass



# ---------------MAIN ----------------------------------
if __name__ == '__main__':

    """
    if (sys.argv[1] != "" and sys.argv[2]!=""):
        z1 = cv.LoadImageM( sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE) 
        z2 = cv.LoadImageM( sys.argv[2], cv.CV_LOAD_IMAGE_GRAYSCALE) 
    """

    #print "------------------"            
    #pathD="./vidstblz/"
    #resdir="./vidstblz/"
    print "------------------"            
    #pathD="./stblze2/"
    #resdir="./stblze2/"    

    pathD="./stblze/"
    resdir="./resstblze/"    
    #pathD="./plnData/"
    #resdir="./resplnData/"    

    #pathD="./gcenter1/"
    #resdir="./resgcenter1/"    

    #pathD="./gcenter2/"
    #resdir="./resgcenter2/"    


    for subdir in dF.DFS(pathD):
        imgs = glob.glob(os.path.join(subdir, "*.jpg"))
        imgs.sort()


    zcorr=[]

    ## input video
    fps=3


    z1 = cv.LoadImageM( imgs[0] , cv.CV_LOAD_IMAGE_GRAYSCALE) 
    #z1 = cv.LoadImageM( imgs[0] ) 


    #vidfourcc = cv.CV_FOURCC('M','J','P','G') 
    vidfourcc = cv.CV_FOURCC('X','V','I','D') 
    frame_size=cv.GetSize( z1 )
    iwriter = cv.CreateVideoWriter ("cpin.avi", int(vidfourcc), fps, frame_size, True)

    d=cv.CreateImage( cv.GetSize(z1), 8,3)
    #cv.Copy(z1,d)
    #cv.WriteFrame(iwriter, d)    

    for i in range(1,len(imgs)):
        z2 = cv.LoadImageM( imgs[i] , cv.CV_LOAD_IMAGE_GRAYSCALE) 
        #z2 = cv.LoadImageM( imgs[i] )
        print imgs[i-1], imgs[i]
        zcorr.append(getMaxLoc(z1,z2))
        z1 = z2
        #cv.Copy(z1,d)
        #cv.WriteFrame(iwriter, d)    

        

    print zcorr



    y=np.asarray(zcorr)
    print np.cumsum(y[:,0])
    print np.cumsum(y[:,1])
    

    xmin = min(np.cumsum(y[:,0]))
    xmax = max(np.cumsum(y[:,0]))
    ymin = min(np.cumsum(y[:,1]))
    ymax = max(np.cumsum(y[:,1]))


    if xmin>0:
        xmin=0
    if ymin>0:
        ymin=0
    if xmax<0:
        xmax=0
    if ymax<0:
        ymax=0

    print xmin, ymin, xmax, ymax
    
    T = (abs(ymin) + z1.rows + abs(ymax), abs(xmin) + z1.cols + abs(xmax), 3)
    print "T=",T

    M = np.zeros( T,  dtype=np.float32)
    #z2 = cv.LoadImageM( imgs[0]  , cv.CV_LOAD_IMAGE_GRAYSCALE)      
    z2 = cv.LoadImageM( imgs[0] )


    xprm =  xmax
    yprm =  ymax
    print "xprm, yprm =", xprm, yprm


    rTop = yprm
    rBot = yprm + z1.rows
    cLeft =xprm 
    cRight=xprm  + z1.cols

    print "rTop, rBot, cLeft, cRight=", rTop, rBot, cLeft, cRight

    M[ rTop:rBot, cLeft: cRight ] = z2
    q = cv.fromarray( np.ascontiguousarray(M) )
    cv.ConvertScale(q, q,1/255.0) 
    cv.ShowImage("q", q)
    print "type=", type(q), q.cols, q.rows, q.type

    i=0
    WriteImFile(resdir, i, q)


    """
    # output video
    fps=3
    frame_size=cv.GetSize(q)
    #frame_size= (q.cols, q.rows)
    #frame_size= (q.rows, q.cols)
    print "frame_size=", frame_size
    vidfourcc = cv.CV_FOURCC('M','J','P','G') 
    #vidfourcc = cv.CV_FOURCC('X','v','i','D') 
    #vidfourcc = CV_FOURCC('D', 'I', 'V', '3')
    cwriter = cv.CreateVideoWriter ("cpout.avi", vidfourcc, fps, frame_size, True)
    ## convert cvMat to IplImage
    zx=cv.CreateMat( q.rows, q.cols, cv.CV_8UC3   )
    cv.ConvertScale( q,zx, 255)
    s = cv.CreateImage( (q.cols, q.rows), 8, 3 )       
    sbar = cv.GetSubRect(zx, (0,0,q.cols,q.rows) )
    cv.SetImageROI(s,(0,0,q.cols,q.rows))
    cv.Copy(sbar, s)
    cv.ShowImage("s",s)
    """


    #cv.WriteFrame(cwriter, s)    


    for i in range(1,len(imgs)):
        print "---", imgs[i], "----"
        M = np.zeros( T )
        print "image=",i, xprm, yprm, zcorr[i-1][0], zcorr[i-1][1]
        xprm = xprm - zcorr[i-1][0] 
        yprm = yprm - zcorr[i-1][1]
        
        rTop = yprm
        rBot = yprm + z1.rows
        cLeft =xprm 
        cRight=xprm  + z1.cols

        #z2 = cv.LoadImageM( imgs[i] , cv.CV_LOAD_IMAGE_GRAYSCALE)      
        z2 = cv.LoadImageM( imgs[i] )
        z1prm = np.asarray(z1, dtype=np.float32)

        print "rTop, rBot, cLeft, cRight=", rTop, rBot, cLeft, cRight
        print M[ rTop:rBot, cLeft: cRight ].shape

        M[ rTop:rBot, cLeft: cRight ] = z2
        q = cv.fromarray( np.ascontiguousarray(M) )
        cv.ConvertScale(q, q,1/255.0) 

        #cv.Circle(q, (200,200), 2, (255,0,0),2)
        cv.ShowImage("q", q)


        WriteImFile(resdir, i, q)

        """
        ## convert cvMat to IplImage
        zx=cv.CreateMat( q.rows, q.cols, cv.CV_8UC3   )
        cv.ConvertScale( q,zx, 255)
        s = cv.CreateImage( (q.cols, q.rows), 8, 3 )       
        sbar = cv.GetSubRect(zx, (0,0,q.cols,q.rows) )
        cv.SetImageROI(s,(0,0,q.cols,q.rows))
        cv.Copy(sbar, s)
        cv.ShowImage("s",s)
        cv.WriteFrame(cwriter, s)    
        """

        cv.WaitKey()






