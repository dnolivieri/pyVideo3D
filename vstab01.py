#!/usr/bin/env python
"""
    dnolivieri:
     -- stabilization with OpenCV
"""
import sys
import cv2
import cv
import numpy as np
import copy   # for deepcopy


lk_params = dict( winSize  = (21, 21), 
                  maxLevel = 5, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                  derivLambda = 0.0 )   

feature_params = dict( maxCorners = 200, 
                       qualityLevel = 0.01,
                       minDistance = 30,
                       blockSize = 30 )

class TransformParam: 
    def __init__(self, dx, dy, da):
        self.dx = dx
        self.dy = dy
        self.da = da

class Trajectory:
    def __init__(self, x,y,a):
        self.x = x
        self.y = y
        self.a = a





# ------------------------------------------
if __name__ == '__main__':

    #cap = cv2.VideoCapture('test1800.mp4')
    cap = cv2.VideoCapture('M2U00302.MPG')

    print type

    k=1
    max_frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)


    flag, frame = cap.read() 
    width = np.size(frame, 1) 
    height = np.size(frame, 0)
    writer = cv2.VideoWriter(filename="your_writing_file.avi", 
                             fourcc=cv2.cv.CV_FOURCC('X','v','i','D'), 
                             fps=30, 
                             frameSize=(width, height))



    prev = frame
    prev_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    last_T = np.array([])


    prev_to_cur_transform = []


    k=1
    while(cap.isOpened()):
        if k==90:
            break
        ret, frame = cap.read()
        cur = frame
        cur_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        

        #cv2.imshow('frame',cur_grey)
        #cv2.waitKey()

        
        """
        goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, 
                corners[, mask[, blockSize[, useHarrisDetector[, k]]]]]) -> corners
        """
        #prev_corner = cv2.goodFeaturesToTrack(prev_grey, 200, 0.01, 30)=[]
        prev_corner = cv2.goodFeaturesToTrack(prev_grey, **feature_params)
        print "prev_corner=", prev_corner[0:5]
        #calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);

        # calculate optical flow
        cur_corner, st, err = cv2.calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, None,**lk_params)
        print cur_corner[0:5]



        print "type(cur_corner)=", type(cur_corner)
        print "cur_corner.shape=", cur_corner.shape

        p=[]
        q=[]
        for i in range(len(st)):
            if st[i]: 
                p.append(prev_corner[i])
                q.append(cur_corner[i])

        prev_corner_corr= np.array(p)
        cur_corner_corr = np.array(q)
        print "type(cur_corner_corr)=", type(cur_corner_corr) 
        print "cur_corner_corr.shape=", cur_corner_corr.shape       

        T = cv2.estimateRigidTransform(prev_corner_corr, cur_corner_corr, False)

        if T.data==None:
            T = last_T 

        last_T = copy.deepcopy(T)

        print "T.shape=", T.shape
        print T

        dx = T[0,2]
        dy = T[1,2]
        da = np.arctan2( T[1,0], T[0,0] )
        prev_to_cur_transform.append( TransformParam(dx, dy, da)  )
        
        print "Frame:", k, dx, dy, da
    

        prev = copy.deepcopy(cur)
        prev_grey = copy.deepcopy(cur_grey)
        k+=1

        #writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




    # step 2:  accum the transform to get the image trajectory
    print "--step 2. -----------------"
    a=0
    x=0
    y=0
    trajectory = []
    for i in range(len(prev_to_cur_transform)): 
        x += prev_to_cur_transform[i].dx
        y += prev_to_cur_transform[i].dy
        a += prev_to_cur_transform[i].da

        trajectory.append(Trajectory(x,y,a) )
        print i, x,y,a




    print "--step 3. -----------------"
    # smoothing step:
    smoothed_trajectory = []
    RADIUS = 30

    for i in range(len(trajectory)):
        sum_x=0
        sum_y=0
        sum_a=0
        cnt=0
        for j in range(-RADIUS, RADIUS+1, 1):
            if i+j>= 0 and i+j < len(trajectory):
                sum_x += trajectory[i+j].x
                sum_y += trajectory[i+j].y
                sum_a += trajectory[i+j].a
                cnt+=1


        avg_a = sum_a/cnt
        avg_x = sum_a/cnt
        avg_y = sum_y/cnt
        
        smoothed_trajectory.append(Trajectory(avg_x, avg_y, avg_a))
        print i, avg_x,avg_y,avg_a



    #step 4
    print "--step 4. -----------------"
    new_prev_to_cur_transform = []

    a=0
    x=0
    y=0

    for i in range(len(prev_to_cur_transform)): 
        x += prev_to_cur_transform[i].dx
        y += prev_to_cur_transform[i].dy
        a += prev_to_cur_transform[i].da


        diff_x = smoothed_trajectory[i].x - x 
        diff_y = smoothed_trajectory[i].y - y
        diff_a = smoothed_trajectory[i].a - a


        dx = prev_to_cur_transform[i].dx + diff_x
        dy = prev_to_cur_transform[i].dy + diff_y
        da = prev_to_cur_transform[i].da + diff_a
        

        new_prev_to_cur_transform.append(  TransformParam(dx, dy, da)  )

        print i, dx, dy, da

                     


    # step5:  app
    print "--step 5: Apply the transform -----------------"
    
    HORIZONTAL_BORDER_CROP = 20

    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
    flag, frame = cap.read()




    #vert_border= HORIZONTAL_BORDER_CROP * prev.rows / prev.cols

    k=0
    while (k< max_frames-1): 
        if k==90:
            break
        ret, frame = cap.read()
        cur = frame

        #cur2 = cv2.cv.CreateMat(width,height,cv2.CV_64F)
        T = np.zeros([2,3])
        T[0,0] = np.cos(new_prev_to_cur_transform[k].da)
        T[0,1] = -np.sin(new_prev_to_cur_transform[k].da)
        T[1,0] = np.sin(new_prev_to_cur_transform[k].da)
        T[1,1] = np.cos(new_prev_to_cur_transform[k].da)

        T[0,2] = new_prev_to_cur_transform[k].dx
        T[1,2] = new_prev_to_cur_transform[k].dy


        print "type(T)=", type(T)
        print "type(cur)", type(cur)
        #print "(cur.shape[1],cur.shape[0])=" (cur.shape[1],cur.shape[0]) 
        print "cur.shape=", cur.shape,( cur.shape[1], cur.shape[0] )

        cur2 = cv2.warpAffine(cur, T, (cur.shape[1], cur.shape[0])  )
        # here do some cropping:
      

        # Puts the two side by side:


        rows,cols =  cur.shape[0], cur.shape[1] 
        canvas = np.zeros( [rows, cols*2+10, 3] )
        canvas[:,0:cols] = cur
        canvas[:,cols+10:cols*2+10] = cur2


        canvasR  = cv2.resize(canvas, (canvas.shape[1]/2, canvas.shape[0]/2)  ) 


        q = cv2.cv.fromarray( np.ascontiguousarray(canvasR) )
        cv2.cv.ConvertScale(q, q,1/255.0) 



        cv2.cv.ShowImage("q", q)
        #cv2.imshow('frame',q)
        cv2.waitKey()
        #cv2.imshow('frame',cur)

        k+=1


    cap.release()
    cv2.destroyAllWindows()
