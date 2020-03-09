# -*- coding: utf-8 -*-

#Main function is at the bottom of the file.

import numpy as np
import cv2
import math


def get_crreunt_video_time(captured_video):
    return captured_video.get(cv2.CAP_PROP_POS_MSEC)

#Switches the video after each MILLISECOND_TO_SWITCH between grey and color.
#seconds: Indicates for how many seconds, video will be switched betwee grey and color.
#captured_video: Instance of VideoCapture.
#Return: returns the current frame,instance of videowriter.
def switch_color_for_N_seconds(captured_video,seconds,MILLISECOND_TO_SWITCH,fps):
      frame = None
      grey_turn = True
      changed_frame = None
      current_video_time = 0
      previous_time = 0
      _,frame = captured_video.read()
      isColor = True
      fourcc = cv2.VideoWriter_fourcc(*'XVID')
      #Lower dimensions do not save the video.
      height,width,_ = frame.shape
      text = "Flashing greyscale and colored frames"
      out = cv2.VideoWriter("output.MP4", fourcc, fps, (1280,720),isColor)
      while(get_crreunt_video_time(captured_video)  < seconds):
          if(grey_turn):
             changed_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
             changed_frame = cv2.cvtColor(changed_frame,cv2.COLOR_GRAY2BGR)
          else:
             changed_frame = frame
          current_video_time =get_crreunt_video_time(captured_video)
          cv2.putText(changed_frame,text,(10,height-300),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),5)
          out.write(changed_frame)
          if(current_video_time > previous_time + MILLISECOND_TO_SWITCH):
              if(grey_turn):
                  grey_turn = False
              else:
                  grey_turn = True
              previous_time = current_video_time
          _,frame = captured_video.read()
      return frame,out
  
    
#Applies gaussian filter.
#seconds: Indicates for how many seconds, video will be switched betwee grey and color.
#captured_video: Instance of VideoCapture.
def apply_gaussian_filter(captured_video,out,current_frame,seconds):
     kernel_size = 31
     increment_kernel_size = 10
     total_frames_processed = 0
     start_index = 10
     font = cv2.FONT_HERSHEY_PLAIN
     text = "Gaussian filter with window's width and height equal"
     height,width,_ = current_frame.shape
     blur_text = "Increasing the kernel size will make image more blurry"
     while(get_crreunt_video_time(captured_video)  < seconds):
           if(total_frames_processed >= 15):
               total_frames_processed = 0
               kernel_size = kernel_size + increment_kernel_size
           final_text =  "Kernel size  = "  + str(kernel_size)
           total_frames_processed = total_frames_processed + 1
           blur = cv2.GaussianBlur(current_frame,(kernel_size,kernel_size),0)
           cv2.putText(blur,text,(start_index,height-500),font,2,(0,0,255),2)
           cv2.putText(blur,blur_text,(start_index,height-300),font,2,(0,0,255),2)
           cv2.putText(blur,final_text,(start_index,height-100),font,3,(255,0,150),2)
           out.write(blur)
           _,current_frame = captured_video.read()
     return current_frame,out
 
    
#Applies bilateral filter.
#seconds: Indicates for how many seconds, video will be switched betwee grey and color.
#captured_video: Instance of VideoCapture.
def apply_bilateral_filter(captured_video,out,current_frame,seconds):
        diameter_for_each_pixel = 1
        increment_step = 5
        sigmaColor = 75
        start_index = 10
        sigmaSpace = 75
        total_frames_processed = 0
        font = cv2.FONT_HERSHEY_PLAIN
        height,width,_ = current_frame.shape
        text = "Bilateral filter: Edge preserving filter"
        second_text = "Takes into account the value of each pixel"
        while(get_crreunt_video_time(captured_video) < seconds):
              if(total_frames_processed >= 15):
                  total_frames_processed = 0
                  diameter_for_each_pixel = diameter_for_each_pixel + increment_step
              final_text = "Kernel Size = "  + str(diameter_for_each_pixel)
              total_frames_processed = total_frames_processed + 1
              blur = cv2.bilateralFilter(current_frame, diameter_for_each_pixel, sigmaColor, sigmaSpace) 
              cv2.putText(blur,text,(start_index,height-300),font,3,(255,0,0),3)
              cv2.putText(blur,second_text,(start_index,height-200),font,3,(0,255,0),3)
              cv2.putText(blur,final_text,(start_index,height-100),font,3,(0,0,255),3)
              out.write(blur)
              _,current_frame = captured_video.read()
        return current_frame,out
              
             

    
#Applies two filter.
def apply_filters(captured_video,out,current_frame):
     current_frame,out = apply_gaussian_filter(captured_video,out,current_frame,8000)
     current_frame,out = apply_bilateral_filter(captured_video,out,current_frame,12000)#8000+4000
     return current_frame,out
 


#Tracks a white object in HSV space.
#seconds: Indicates for how many seconds, video will be switched betwee grey and color.
#captured_video: Instance of VideoCapture.
def show_object_in_HSV_space(captured_video,out,current_frame,seconds):
     lower_white = (0,0,130)
     upper_white = (179,25,255)
     text = "Football detection in HSV space"
     final_text = "Erode and Dilation are used for smooth shape"
     third_text = "Image without morphological operations"
     fin_text = "Image with morphological operations"
     rows,cols,_ = current_frame.shape
     while(get_crreunt_video_time(captured_video) < seconds):
            blurred = cv2.GaussianBlur(current_frame,(11,11),0)
            hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv,lower_white,upper_white)
            copied = mask.copy()
            mask = cv2.erode(mask,None,iterations=8)
            mask = cv2.dilate(mask,None,iterations=8)
            mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            copied_bgr = cv2.cvtColor(copied,cv2.COLOR_GRAY2BGR)
            resized_mask = cv2.resize(mask, (rows, int(cols/2)), interpolation = cv2.INTER_LINEAR) 
            resized_orig = cv2.resize(copied_bgr, (rows, int(cols/2)), interpolation = cv2.INTER_LINEAR) 
            cv2.putText(resized_mask,text,(10,rows-600),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
            cv2.putText(resized_mask,final_text,(10,rows-400),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
            cv2.putText(resized_mask,fin_text,(10,rows-200),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)
            cv2.putText(resized_orig,third_text,(10,rows-200),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255),2)
            result = np.concatenate((resized_mask, resized_orig), axis=1)
            result = cv2.resize(result, (1280, 720), interpolation = cv2.INTER_LINEAR) 
            out.write(result)
            _,current_frame= captured_video.read()
     return current_frame,out
 
    
#Tracks a white object in RGB space.
#seconds: Indicates for how many seconds, video will be switched betwee grey and color.
#captured_video: Instance of VideoCapture.
def show_object_in_RGB_space(captured_video,out,current_frame,seconds):
     lower_white = (150,130,130)
     upper_white = (255,255,255)
     rows,cols,_ = current_frame.shape
     text = "Football detection in RGB space"
     final_text = "Erode and Dilation are used for smooth shape"
     third_text = "Image without morphological operations"
     fin_text = "Image with morphological operations"
     while(get_crreunt_video_time(captured_video) < seconds):
            blurred = cv2.GaussianBlur(current_frame,(11,11),0)
            rgb = cv2.cvtColor(blurred,cv2.COLOR_BGR2RGB)
            mask = cv2.inRange(rgb,lower_white,upper_white)
            copied = mask.copy()
            mask = cv2.erode(mask,None,iterations=8)
            mask = cv2.dilate(mask,None,iterations=8)
            mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            copied_bgr = cv2.cvtColor(copied,cv2.COLOR_GRAY2BGR)
            resized_mask = cv2.resize(mask, (rows, int(cols/2)), interpolation = cv2.INTER_LINEAR) 
            resized_orig = cv2.resize(copied_bgr, (rows, int(cols/2)), interpolation = cv2.INTER_LINEAR) 
            cv2.putText(resized_mask,text,(10,rows-600),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
            cv2.putText(resized_mask,final_text,(10,rows-400),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
            cv2.putText(resized_mask,fin_text,(10,rows-200),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
            cv2.putText(resized_orig,third_text,(10,rows-200),cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)
            result = np.concatenate((resized_mask, resized_orig), axis=1)
            result = cv2.resize(result, (1280, 720), interpolation = cv2.INTER_LINEAR) 
            out.write(result)
            _,current_frame= captured_video.read()
     return current_frame,out




#Applies soble filter to an image.
#seconds: Indicates for how many seconds, video will be switched betwee grey and color.
#captured_video: Instance of VideoCapture.
def apply_sobel_gradient(captured_video,out,current_frame,seconds):
     kernel_size = -1
     kernel_increment = 2
     sigmaColor = 75
     sigmaIncrement = 20
     text = "Sobel edge detector: Detected edges are shown in Green"
     height,width,_ = current_frame.shape
     frames_processed = 0
     while(get_crreunt_video_time(captured_video) < seconds):
           if(frames_processed >= 30):
               frames_processed = 0
               sigmaColor = sigmaColor + sigmaIncrement
               kernel_size = kernel_size + kernel_increment
           if (kernel_size  > 9):
               kernel_size = 9
           frames_processed  = frames_processed + 1
           blurred = cv2.bilateralFilter(current_frame, 25, sigmaColor, 70)
           converted_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
           sobel_x = cv2.Sobel(converted_img,cv2.CV_64F,1,0,kernel_size)
           sobel_x = np.absolute(sobel_x)
           sobel_x = np.uint8(sobel_x)
           sobel_y = cv2.Sobel(converted_img,cv2.CV_64F,0,1,kernel_size)
           sobel_y = np.absolute(sobel_y)
           sobel_y = np.uint8(sobel_y)
           sobel = sobel_x+sobel_y
           sobel = cv2.cvtColor(sobel,cv2.COLOR_GRAY2BGR)
           final_text = "Kernel size  = " + str(kernel_size) + " &  sigmaColor = " + str(sigmaColor)
           sobel[np.where((sobel > [200,200,200]).all(axis = 2))] = [0,255,0]
           cv2.putText(sobel,text,(10,height-600),cv2.FONT_HERSHEY_PLAIN,2,(150,230,255),3)
           cv2.putText(sobel,final_text,(10,height-300),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),5)
           out.write(sobel)
           _,current_frame= captured_video.read()
     return current_frame,out
         
 
    
#Returns the cricle.
#seconds: Indicates for how many seconds, video will be switched betwee grey and color.
#captured_video: Instance of VideoCapture.
def get_circles(current_frame):
         img = current_frame
         param_1 = 400
         param_2 = 1
         min_Radius = 0
         max_Radius = 70
         d_p = 1
         min_Distance = 10
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         gray = cv2.erode(gray,None,iterations=15)
         gray = cv2.dilate(gray,None,iterations=15)
         img_blur = cv2.bilateralFilter(gray, 5, 175, 175)
         circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=d_p, minDist=min_Distance, param1=param_1, param2=param_2, minRadius=min_Radius, maxRadius=max_Radius)
         return circles
     
        

 
#Helper function which returns a sharped image.
#seconds: Indicates for how many seconds, video will be switched betwee grey and color.
#captured_video: Instance of VideoCapture.
def get_Sharpened_image(current_frame):
         kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) /8.0
         return cv2.filter2D(current_frame, -1, kernel_sharpen) #Edge sharpening
    
    
    
#It detects circle with hough transform.
#seconds: Indicates for how many seconds, video will be switched betwee grey and color.
#captured_video: Instance of VideoCapture.
def detect_circle_with_hough_transform(captured_video,out,current_frame,seconds):
         frames_processed = 0
         text = "Intial maxRadius = 70 and minRadius = 0"
         second_text = "Radius has impact on the circle detection."
         third_text = "minDist is 10"
         fourth_text = ""
         checked = False
         height,width,_ = current_frame.shape
         param_1 = 100
         param_2 = 1
         min_Radius = 0
         max_Radius = 70
         d_p = 1
         min_Distance = 10
         while(get_crreunt_video_time(captured_video) < seconds):
               if(frames_processed >= 60 and checked == False):
                   max_Radius = 30
                   text = " Max Radius  = 40 => Less circles are detected"
                   third_text =  "MinDist = 10 & larger minDist => some circles will never be detected"
                   fourth_text = "minDist = 10 & smaller minDist => few false circles will be detected"
                   checked = True
                   frames_processed = 0
               if(frames_processed >= 50 and checked):
                   text =  " Max Radius  = 70 => Good enough to detect ball at all scales in this video"
                   max_Radius = 70
               frames_processed = frames_processed+1
               sharpened = get_Sharpened_image(current_frame)
               gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
               gray = cv2.erode(gray,None,iterations=15)
               gray = cv2.dilate(gray,None,iterations=15)
               img_blur = cv2.bilateralFilter(gray, 5, 175, 175)
               circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=d_p, minDist=min_Distance, param1=param_1, param2=param_2, minRadius=min_Radius, maxRadius=max_Radius)
               if circles is not None:
                  circles = np.uint16(np.around(circles))
                  for i in circles[0, :]:
                    cv2.circle(current_frame, (i[0], i[1]), i[2], (255, 0, 0), 2)
                    cv2.circle(current_frame, (i[0], i[1]), 2, (255, 0, 0),20)
                    break
               cv2.putText(current_frame,text,(10,height-600),cv2.FONT_HERSHEY_PLAIN,2,(30,230,255),3)
               cv2.putText(current_frame,second_text,(10,height-400),cv2.FONT_HERSHEY_PLAIN,2,(30,30,255),3)
               cv2.putText(current_frame,third_text,(10,height-200),cv2.FONT_HERSHEY_PLAIN,2,(80,130,255),3)
               cv2.putText(current_frame,fourth_text,(10,height-100),cv2.FONT_HERSHEY_PLAIN,2,(90,100,255),3)
               out.write(current_frame)
               _,current_frame= captured_video.read()
         return current_frame,out
                   
         
#It draws a rectangle arround the ball.
#seconds: Indicates for how many seconds, video will be switched betwee grey and color.
#captured_video: Instance of VideoCapture.
def draw_rectangle_arround_ball(captured_video,out,current_frame,seconds):
        while(get_crreunt_video_time(captured_video) < seconds):
           circles = get_circles(current_frame)
           if circles is not None:
                  circles = np.uint16(np.around(circles))
                  for i in circles[0, :]:
                    cv2.rectangle(current_frame,(i[0]-int(i[2]), i[1]-int(i[2])),(i[0]+int(i[2]), i[1]+int(i[2])),(0,0,255),2)
                    break
           out.write(current_frame)
           _,current_frame= captured_video.read()
        return current_frame,out 


#Converts images to grayscale.
#seconds: Indicates for how many seconds, video will be switched betwee grey and color.
#captured_video: Instance of VideoCapture.
def convert_image_to_grey_scale(captured_video,out,current_frame,seconds):
       rows,cols,_ = current_frame.shape
       text = "Gray scale image with pixel's intensity proportional to football's pixel's"
       while(get_crreunt_video_time(captured_video) < seconds):
             gray = cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
             sharpened = get_Sharpened_image(current_frame)
             circles = get_circles(sharpened)
             x_co = 0
             y_co = 0
             if circles is not None:
                  circles = np.uint16(np.around(circles))
                  for i in circles[0, :]:
                    x_co = i[0]
                    y_co = i[1]
                    break
             for i in range(rows-1):
                 for j in range(cols-1):
                     value = min(int(((gray[i][j])/(gray[x_co][y_co]))*gray[i][j]),255)
                     gray[i][j] = value
             gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
             cv2.putText(gray,text,(10,rows-400),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
             out.write(gray)
             _,current_frame= captured_video.read()
       return current_frame,out
   


#Makes the object invisible by copying pixels from neighbourhood.
#seconds: Indicates for how many seconds, video will be switched betwee grey and color.
#captured_video: Instance of VideoCapture.
def make_object_invisible(captured_video,out,current_frame,seconds):
       height,width,_ = current_frame.shape
       text = "Trying to make the object invisible by copying pixel values from neighbourhood"
       second_text = "Hough-Transfrom failure => object not invisible"
       while(get_crreunt_video_time(captured_video) < seconds):
             sharpened = get_Sharpened_image(current_frame)
             circles = get_circles(sharpened)
             x_co = 0
             y_co = 0
             radius = 0
             if circles is not None:
                  circles = np.uint16(np.around(circles))
                  for i in circles[0, :]:
                    x_co = i[0]
                    y_co = i[1]
                    radius = i[2]+20
                    break
             start_point_x = x_co-radius-10
             end_point_x = x_co+radius+10
             start_point_y = y_co-radius-10
             end_point_y = y_co+radius+0
             for k in range(start_point_x-10,end_point_x+20):
                for g in range(start_point_y-10,end_point_y+20):
                    first = (x_co-k)**2
                    second = (y_co-g)**2
                    summed = first+second
                    index_y = k+radius+20
                    index_x = g+radius+20
                    distance = math.sqrt(summed)
                    if(distance < radius):
                       if(index_x < height and index_y < width):
                           current_frame[g][k] = current_frame[index_x][index_y]
                       elif(((index_x-5) < height and (index_y-5) < width)):
                           current_frame[g][k] = current_frame[index_x-5][index_y-5]
                       elif((index_x-8 < height and index_y-8 < width)):
                           current_frame[g][k] = current_frame[index_x-8][index_y-8]
                       elif((index_x-9 < height and index_y-9 < width)):
                           current_frame[g][k] = current_frame[index_x-9][index_y-9]
                       else:
                           current_frame[g][k] = current_frame[index_x-10-radius][index_y-10-radius]
             cv2.putText(current_frame,text,(10,height-400),cv2.FONT_HERSHEY_PLAIN,2,(30,30,255),2)
             cv2.putText(current_frame,second_text,(10,height-200),cv2.FONT_HERSHEY_PLAIN,2,(30,30,255),2)
             out.write(current_frame)
             _,current_frame= captured_video.read()
       return current_frame,out



#Sharp the objects
#seconds: Indicates for how many seconds, video will be switched betwee grey and color.
#captured_video: Instance of VideoCapture.
def shapren_objects_with_edges(captured_video,out,current_frame,seconds):
      rows,cols,_ = current_frame.shape
      text = "Original frame"
      second_text = "Sharped images"
      while(get_crreunt_video_time(captured_video) < seconds):
            sharp = get_Sharpened_image(current_frame)
            original_frame = cv2.resize(current_frame, (rows, int(cols/2)), interpolation = cv2.INTER_LINEAR) 
            cv2.putText(original_frame,text,(10,rows-400),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
            sharp_image = cv2.resize(sharp, (rows, int(cols/2)), interpolation = cv2.INTER_LINEAR) 
            cv2.putText(sharp_image,second_text,(10,rows-200),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
            result = np.concatenate((original_frame, sharp_image), axis=1)
            result = cv2.resize(result, (1280, 720), interpolation = cv2.INTER_LINEAR) 
            out.write(result)
            _,current_frame= captured_video.read()
      return current_frame,out





""""------------------------------MAIN-------------------------------------------------"""
captured_video = cv2.VideoCapture('input.MP4')
fps = round(captured_video.get(cv2.CAP_PROP_FPS))
print("Started switching colors")
current_frame,out = switch_color_for_N_seconds(captured_video,4000,500,fps)
print("Applying filters")
current_frame,out = apply_filters(captured_video,out,current_frame) 
print("Tracking objects in HSV space")
current_frame,out = show_object_in_HSV_space(captured_video,out,current_frame,16000) #HSV Space for 4 seconds
print("Tracking objects in RGB space")
current_frame,out = show_object_in_RGB_space(captured_video,out,current_frame,20000) #RGB Spae
print("Applying sobel gradient")
current_frame,out = apply_sobel_gradient(captured_video,out,current_frame,25000)
print("Detecting circles with hough-transform")
current_frame,out = detect_circle_with_hough_transform(captured_video,out,current_frame,35000)
print("Drawing a rectangle arround the ball")
current_frame,out = draw_rectangle_arround_ball(captured_video,out,current_frame,37000)
print("Converting image to grayscale with intensity proportional to the ball's pixel's")
current_frame,out = convert_image_to_grey_scale(captured_video,out,current_frame,40000)
print("Making the object invisible")
current_frame,out = make_object_invisible(captured_video,out,current_frame,50000)
print("Sharpend the football section and edges")
current_frame,out = shapren_objects_with_edges(captured_video,out,current_frame,60000)
out.release()
captured_video.release()














