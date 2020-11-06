from StarMap import StarMap
import sys, os

if __name__ == '__main__':
    images_path = os.path.abspath(os.path.dirname(sys.argv[0])) 
    images_path = images_path[:-6]
    images_path = images_path + "images/" # take path for images
    path_scene = images_path + "starmap.png"
    #path_area = images_path + "normal.png"
    path_area = images_path + "rotated.png"


    star = StarMap(path_scene, path_area)
    scene_corns= star.find_area()
    print(scene_corns)
