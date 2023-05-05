#!/usr/bin/env python
# coding: utf-8

# ### Dense optical flow based sampling
# [Reference tutorial](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html)

# In[13]:


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
from cap_from_youtube import cap_from_youtube


# In[9]:


def get_max_flow_frame(url):    
    
    #cap = cv.VideoCapture(cv.samples.findFile("../data/archery.mp4"))
    
    cap = cap_from_youtube(url)
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    highest_magnitude_flow_frame = None
    highest_flow = 0

    flow_plot = []

    while(1):
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break

        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Find the frame with the highest magnitude of flow
        highest_magnitude_flow_frame = frame2 if highest_magnitude_flow_frame is None else highest_magnitude_flow_frame
        highest_flow = np.linalg.norm(flow) if highest_flow == 0 else highest_flow
        
        highest_flow = np.linalg.norm(flow) if np.linalg.norm(flow) > highest_flow else highest_flow
        highest_magnitude_flow_frame = frame2 if np.linalg.norm(flow) > highest_flow else highest_magnitude_flow_frame
        
        # Plot the flow
        flow_plot.append(np.linalg.norm(flow))

        
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        #cv.imshow('frame2', bgr)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # elif k == ord('s'):
        #     cv.imwrite('opticalfb.png', frame2)
        #     cv.imwrite('opticalhsv.png', bgr)
        prvs = next

    cv.destroyAllWindows()

    cap.release()
    
    return highest_magnitude_flow_frame, highest_flow, flow_plot
    
    


# In[14]:


# url = 'https://www.youtube.com/watch?v=--07WQ2iBlw&ab_channel=RyanWatters'
# start_time = time.time()
# highest_magnitude_flow_frame, highest_flow, flow_plot = get_max_flow_frame(url)

# runtime = time.time() - start_time # 1m 16.0s


# In[15]:


# display the frame with the highest magnitude of flow
# plt.imshow(highest_magnitude_flow_frame)
# plt.title('Frame with the highest magnitude of flow')
# plt.show()
# print('Highest magnitude of flow: ', highest_flow)


# In[16]:


# plt.plot(flow_plot)
# plt.title('Flow plot')
# plt.xlabel('Frame')
# plt.ylabel('Flow')
# plt.show()

