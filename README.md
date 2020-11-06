# starmap
This repository is about finding cropped area on the starmap image. The algorithm for the problem is implemented via Python programming language. 

The codes are tested on Ubuntu 18.04, Python 3.6.9, Opencv 4.4.0 

# Please follow the instructions to run the repo: 

Open a terminal 
Then, write: 
```
$ wget https://github.com/mericdurukan/starmap.git
$ cd starmap/python
$ chmod +x main.py
$ python3 main.py
```
If you want to give different area to test the code please change the path for the file: 

In main.py choose among them: 

```
9 path_area = images_path + "normal.png"
10 path_area = images_path + "rotated.png"
```

The main script print out the corner points on the given starmap image. 

Note: Please to see the cropped area on the original image, replace the image resizing. My monitor is not capable to see whole starmap image, so I resized the starmap image. However, the corner points are on the orginal image.

# Some Detail Information: 


The given StarMap image: 

![Alt text](https://github.com/mericdurukan/starmap/blob/main/images/starmap.png)

The given normal cropped image: 

![Alt text](https://github.com/mericdurukan/starmap/blob/main/images/normal.png)

The given cropped and rotated image: 

![Alt text](https://github.com/mericdurukan/starmap/blob/main/images/rotated.png)

# RESULTS: 

# Normal cropped image: 

![Alt text](https://github.com/mericdurukan/starmap/blob/main/result_images/starmap_normal.jpg)

# Normal cropped image with feature match:  

![Alt text](https://github.com/mericdurukan/starmap/blob/main/result_images/starmap_normal_match.jpg)
 
 # Normal cropped and rotated image: 
 
 ![Alt text](https://github.com/mericdurukan/starmap/blob/main/result_images/starmap_rotated.jpg)
 
 
 
  # Normal cropped and rotated image with feature map: 
 
 ![Alt text](https://github.com/mericdurukan/starmap/blob/main/result_images/starmap_rotated_match.jpg)


 contact info: mericdurukan@gmail.com
 
