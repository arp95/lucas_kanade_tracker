"""
 *  MIT License
 *
 *  Copyright (c) 2019 Arpit Aggarwal Shantam Bajpai
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without
 *  limitation the rights to use, copy, modify, merge, publish, distribute,
 *  sublicense, and/or sell copies of the Software, and to permit persons to
 *  whom the Software is furnished to do so, subject to the following
 *  conditions:
 *
 *  The above copyright notice and this permission notice shall be included
 *  in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
"""

# header files
from utils import *
import glob
import sys

# set data path
args = sys.argv
path_data = ""
if(len(args) > 1):
    path_data = args[1]

# define constants
path_of_images = path_data + "/*"
files  = glob.glob(path_of_images)
files = sorted(files)
x_range = np.array([269, 303])
y_range = np.array([75, 139])
p = np.array([[0, 0, 0, 0, 0, 0]]).T
template_image = cv2.imread(files[0])
gray_template_image = get_grayscale_image(template_image)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output_car4.avi", fourcc, 20.0, (480, 270))

# read frames
count = 0
for file in files:
    # read image
    image = cv2.imread(file)
    image_copy = image.copy()
    
    # get grayscale image
    gray = get_grayscale_image(image_copy)
    gray = update_grayscale_image(gray_template_image, gray)
    
    # run lucas-kanade algo
    (new_rectangle_coordinates, p) = lucas_kanade_algorithm(gray_template_image, gray, x_range, y_range, p)
    count = count + 1
    
    centroid_x = int((new_rectangle_coordinates[0, 0] + new_rectangle_coordinates[0, 2]) / 2.0)
    centroid_y = int((new_rectangle_coordinates[1, 0] + new_rectangle_coordinates[1, 2]) / 2.0)
    image = cv2.rectangle(image, (centroid_x - 17, centroid_y - 32), (centroid_x + 17, centroid_y + 32), (0, 0, 255), 2)
    
    # write frame
    out.write(image)

out.release()
cv2.destroyAllWindows()
