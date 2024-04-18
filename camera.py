# 用于图像展示用，局部放大镜效果
import cv2
import os
import numpy as np
 
img_path= r'G:\\OneDrive\\Desktop\\A\\5.png' # 请输入自己需要放大图像的路径
    
img_name = os.path.basename(img_path)
img = cv2.imread(img_path)

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global img
    if event == cv2.EVENT_LBUTTONDOWN: # 按下鼠标左键则放大所点的区域
        xy = "%d,%d" % (x, y)
        print (xy)
        length = 30 # 局部区域的边长的一半
        big_length = 150 # 放大后图像的边长
        part_left = x - length
        part_right = x + length
        part_top = y - length
        part_bottom = y + length
        height, width, _ = np.shape(img)
        if (x < width / 2) & (y < height / 2): 
            loc_left = 10
            loc_top = 10
            loc_right = loc_left + big_length
            loc_bottom = loc_top + big_length
            cv2.line(img,(part_right,part_top),(loc_right,loc_top),(0,0,0),2)
            cv2.line(img,(part_left,part_bottom),(loc_left,loc_bottom),(0,0,0),2)
        elif (x >= width / 2) & (y < height / 2):
            loc_right = width - 10
            loc_left = loc_right - big_length
            loc_top = 10
            loc_bottom = loc_top + big_length
            cv2.line(img,(part_left,part_top),(loc_left,loc_top),(0,0,0),2)
            cv2.line(img,(part_right,part_bottom),(loc_right,loc_bottom),(0,0,0),2)
        elif (x < width / 2) & (y >= height / 2):
            loc_left = 10
            loc_right = loc_left + big_length
            loc_bottom = height - 10
            loc_top = loc_bottom - big_length
            cv2.line(img,(part_left,part_top),(loc_left,loc_top),(0,0,0),2)
            cv2.line(img,(part_right,part_bottom),(loc_right,loc_bottom),(0,0,0),2)
        elif (x >= width / 2) & (y >= height / 2):
            loc_bottom = height - 10
            loc_top = loc_bottom - big_length
            loc_right = width - 10
            loc_left = loc_right - big_length
            cv2.line(img,(part_right,part_top),(loc_right,loc_top),(0,0,0),2)
            cv2.line(img,(part_left,part_bottom),(loc_left,loc_bottom),(0,0,0),2)
        
        part = img[part_top:part_bottom,part_left:part_right]
        mask = cv2.resize(part, (big_length, big_length), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        img[loc_top:loc_bottom,loc_left:loc_right]=mask
        cv2.rectangle(img,(part_left,part_top),(part_right,part_bottom),(0,0,0),2)
        cv2.rectangle(img,(loc_left,loc_top),(loc_right,loc_bottom),(0,0,0),2)
        cv2.imshow("image", img)
        
    if event == cv2.EVENT_RBUTTONDOWN: # 按下鼠标右键恢复原图
        img = cv2.imread(img_path)
        cv2.imshow("image", img)

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)
         
cv2.waitKey(0)   
cv2.imwrite("image1.jpg", img)
