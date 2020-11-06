import numpy as np
import cv2 as cv
import time
import sys, os

class StarMap:
  def __init__(self, image_scene_path, image_cropped_path):
    self.img1 = image_scene_path
    self.img2 = image_cropped_path
    
  def find_area(self):
    #Read image from given path
    img_scene = cv.imread(self.img1, cv.IMREAD_GRAYSCALE) # starmap 
    img_object = cv.imread(self.img2, cv.IMREAD_GRAYSCALE) #cropped image
    
    #If there is a mistake in the given path show an error message
    if img_object is None or img_scene is None:
      print('Could not open or find the images, please change your path!')
      exit(0)
   
    #keypoint and descriptor initilaization
    # I prefer to use ORB method in here, because it is an efficient algorithm especially for rotation and scale invariation. Moreover, it is a low cost method that combines FAST keypoint detector and BRIEF descriptor.  
    #Parameter fine tuning: 
    # Used parameters have tried for given images (normal images and rotated). 
    # nfeatures: It is essential to set the value to a high value. This beacuse there are many features on the starmap.
    # nlevels: 	It is related to the number of pyramid levels. Hence, this parameter is used mainly for the partial scale invariance
    # scaleFactor: It is defined as Pyramid decimation ratio. Higher scale factor can lead to the degreadation of feature matching performance. I changed this value , but I decided to use default value.  
    # WTA_K: The number of random points to compare their brightnesses. I tried 3 and 4, and the best result is taken with 2 for starmap (for 2 Hamming distance may be used)
    #edgeThreshold: the default value is too high, especially for cropped images. For this reason I set it to 5 to capture the information that is near to the borders.
    #patchSize: It is used to change patch size for descriptor. The default value can bring about the misinformation on the images. I set it to 25.   
    detector = cv.ORB_create(nfeatures=100000,nlevels=8, scaleFactor=1.2, WTA_K=2, edgeThreshold = 5, patchSize = 25)

    #The calculation of key points and descriptors. 
    keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None) #cropped image
    keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None) #starmap

    #This part is about descriptor matcher. The matcher find the distance between the features for the descriptors.
    #Due to the necessary for using hamming distance, I think brute force search is more suitable than flann(ORB includes a binary descriptor)
   
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    knn_matches = bf.knnMatch(descriptors_obj,descriptors_scene,k=2)	

	
    # find reliable matches using a threshold
    # I set the threshold ratio to '0.85'. With '0.7' there are few points are selected. With '1' there are many points can pass the threshold whether they are necessary or unnecessary.   

    ratio_thresh = 0.85 # The threshold for the selection of good matches.
    good_matches = []
    for m,n in knn_matches:
      if m.distance < ratio_thresh * n.distance:
        good_matches.append(m) #take filtered results


    #show the matching results on the image.
    img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8) # matcher image initilization.

    cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) #draw matched keypoints. 

    obj = np.empty((len(good_matches),2), dtype=np.float32) #good points on the cropped area
    scene = np.empty((len(good_matches),2), dtype=np.float32) # good points on the starmap

    #Take the good points on the cropped area and obj.(This process is necessary for finding homography)
    for i in range(len(good_matches)):
    #-- Get the keypoints from the good matches
      obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
      obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
      scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
      scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
    

    # It is vital to find a 'good' transformation(or projection) between the cropped area and starmap.
    #Homography function is eligible to handle this issue. 
    #There can be some outliers that cannot be eliminated with threshold. (Without RANSAC)
    #RANSAC is used to find a suitable area without any outliers. (eliminate the outliers and match the features)

    H, _ =  cv.findHomography(obj, scene, cv.RANSAC) # estimated homography
    #The cropped object must be transformed onto original image(starmap)

    #It is important to take the cropped image's property because we have to find this area on the surface properly.
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = img_object.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = img_object.shape[1]
    obj_corners[2,0,1] = img_object.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = img_object.shape[0]

    #Cropped area is transformed by using H matrix(3X3)
    scene_corners = cv.perspectiveTransform(obj_corners, H)

    show_image = cv.cvtColor(img_scene,cv.COLOR_GRAY2RGB) #To show the corner points in the scene, the show image is created. 

    #Draw the lines between calculated corner points on the original image. (red)
    cv.line(show_image, (int(scene_corners[0,0,0] ), int(scene_corners[0,0,1])),\
    (int(scene_corners[1,0,0] ), int(scene_corners[1,0,1])), (0,0,255), 2) 
    cv.line(show_image, (int(scene_corners[1,0,0] ), int(scene_corners[1,0,1])),\
    (int(scene_corners[2,0,0] ), int(scene_corners[2,0,1])), (0,0,255), 2)
    cv.line(show_image, (int(scene_corners[2,0,0] ), int(scene_corners[2,0,1])),\
    (int(scene_corners[3,0,0] ), int(scene_corners[3,0,1])), (0,0,255), 2)
    cv.line(show_image, (int(scene_corners[3,0,0] ), int(scene_corners[3,0,1])),\
    (int(scene_corners[0,0,0] ), int(scene_corners[0,0,1])), (0,0,255), 2)

    #Draw the lines between calculated corner points on the matcher image. (green)
    cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
    (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 3)
    cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
    (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 3)
    cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
    (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 3)
    cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
    (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 3)

    #Path for saving
    images_path_save = os.path.abspath(os.path.dirname(sys.argv[0])) 
    images_path_save = images_path_save[:-6]
    images_path_save = images_path_save + "result_images/" # take path for images
  
    #images_path_save = images_path_save + "normal.png"
    images_path_save = images_path_save + "rotated.png"
    #Show the results
    cv.imshow('STARMAP FIND CROPPED AREA WITH MATCHES',cv.resize(img_matches, (960, 540))) # show matches
    cv.imshow('STARMAP FIND CROPPED AREA',cv.resize(show_image, (960, 540))) # show cropped area
    cv.imwrite(images_path_save,cv.resize(img_matches, (960, 540)) ) #write path
    cv.waitKey()
    #return with corners 
    return scene_corners
