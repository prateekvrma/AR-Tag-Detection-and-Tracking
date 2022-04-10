'''
ENPM673 - Project 2: Problem 2(A)
Code by Prateek Verma (118435039)
    '''
import argparse
import csv
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy

# Intrinsic matrix to transform 3D camera coordinates to 2D homogeneous image coordinates.
k_matrix_val = np.array([[1346.10059534175, 0, 0], [0, 1355.93313621175, 0], [932.163397529403, 654.898679624155, 1]])
k_matrix_val = np.transpose(k_matrix_val)
c_detect = None

class MovingAverage:
    def __init__(self, window_size, weight):

        self.window_size_ = window_size
        self.quadrilaterals_ = []
        self.average_ = 0
        self.weight_ = weight

    def addQuadrilateral(self, points):

        if len(self.quadrilaterals_) < self.window_size_:
            self.quadrilaterals_.append(points)

        else:
            self.quadrilaterals_.pop(0)
            self.quadrilaterals_.append(points)

    def getAverage(self):
        
        quadrilaterals = np.array(self.quadrilaterals_)
        # print(quadrilaterals.shape)
        weights = np.ones((1, self.window_size_))
        weights[0, self.window_size_-1] = self.weight_

        sum = 0
        for i in range(self.window_size_):
            sum = sum + weights[0,i] * quadrilaterals[i]

        self.average_ = sum / np.sum(weights)
        # self.average_ = np.mean(quadrilaterals, axis = 0)
        return self.average_

    def getListLength(self):
        l = len(self.quadrilaterals_)
        return l

def drawCube(image, bottom_points, top_points):
    cv2.drawContours(image, [bottom_points], 0, (255, 0 ,255),3)
    cv2.drawContours(image, [top_points], 0, (255, 0, 255),3)

    for i in range(0, bottom_points.shape[0]):
        color = (int(255/(i+1)), 0, int(255/(i+1)))
        cv2.line(image, (bottom_points[i,0], bottom_points[i,1]), (top_points[i,0], top_points[i,1]), color, 3)

    return image

def sortCorners(points):

    x_sorted = points[np.argsort(points[:,0])]

    points_left = x_sorted[0:2, :]
    points_right = x_sorted[2:4, :]

    left_sorted_y = points_left[np.argsort(points_left[:,1])]
    tl, bl = left_sorted_y

    right_sorted_y = points_right[np.argsort(points_right[:,1])]
    tr, br = right_sorted_y
    points_sorted = np.array([tl, bl, br, tr])
    return points_sorted

def getTagCorners(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    # bilateral_filter = cv2.bilateralFilter(gray, 5, 75, 75)
    blur = cv2.medianBlur(gray,3)
    dst = cv2.cornerHarris(blur,6,7,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    corners = corners[1:,:]
    count = 0
    count1 = 0
    count2 = 0
    for c in corners:
        if c[0] > int(1.075*min(corners[:,0])) and c[0] < int(0.925*max(corners[:,0])) and c[1] > int(1.075*min(corners[:,1])) and c[1] < int(0.925*max(corners[:,1])):
            # cv2.circle(image, (int(c[0]), int(c[1])), 3, (255,0,0), -1)
            if count==0:
                c_detect = np.array([(c[0], c[1])])
            else:
                c_detect = np.append(c_detect, [(c[0], c[1])], axis=0)
            count +=1
    
    for c in c_detect:
        if c[0] > min(c_detect[:,0]) and c[0] < max(c_detect[:,0]) and c[1] > min(c_detect[:,1]) and c[1] < max(c_detect[:,1]):
            # cv2.circle(image, (int(c[0]), int(c[1])), 3, (0,0,0), -1)
            if count1==0:
                c_in = np.array([(c[0], c[1])])
            else:
                c_in = np.append(c_detect, [(c[0], c[1])], axis=0)
            count1 += 1
        # if c[0] > int(1.075*min(corners[:,0])) and c[0] < int(0.925*max(corners[:,0])) and c[1] > int(1.075*min(corners[:,1])) and c[1] < int(0.925*max(corners[:,1])):
        else:
            cv2.circle(image, (int(c[0]), int(c[1])), 3, (255,0,0), -1)
            if count2==0:
                c_out = np.array([(c[0], c[1])])
            else:
                c_out = np.append(c_detect, [(c[0], c[1])], axis=0)
            count2 +=1
    
    TagPoints = c_out
    TagPoints = order_points_old(TagPoints)

    return TagPoints



def order_points_old(pts):
	# initialize a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect
def computeHomography(corners1, corners2):

    if (len(corners1) < 4) or (len(corners2) < 4):
        print("Atleast four points needed to compute SVD.")
        return 0

    x1 = corners1[:,0]
    y1 = corners1[:,1]
    x2 = corners2[:,0]
    y2 = corners2[:,1]

    nrows = 8
    # ncols = 9
    
    A = []
    for i in range(int(nrows/2)):
        row1 = np.array([-x1[i], -y1[i], -1, 0, 0, 0, x1[i]*x2[i], y1[i]*x2[i], x2[i]])
        A.append(row1)
        row2 = np.array([0, 0, 0, -x1[i], -y1[i], -1, x1[i]*y2[i], y1[i]*y2[i], y2[i]])
        A.append(row2)

    A = np.array(A)
    U, E, VT = np.linalg.svd(A)
    V = VT.transpose()
    H_vertical = V[:, V.shape[1] - 1]
    H = H_vertical.reshape([3,3])
    H = H / H[2,2]
    # print("Homography Matrix:")
    # print(H)
    return H


def applyHomography2ImageUsingInverseWarping(image, H, size):

    Yt, Xt = np.indices((size[0], size[1]))
    lin_homg_pts_trans = np.stack((Xt.ravel(), Yt.ravel(), np.ones(Xt.size)))

    H_inv = np.linalg.inv(H)
    lin_homg_pts = H_inv.dot(lin_homg_pts_trans)
    lin_homg_pts /= lin_homg_pts[2,:]

    Xi, Yi = lin_homg_pts[:2,:].astype(int)

    Xi[Xi >=  image.shape[1]] = image.shape[1]-1
    Xi[Xi < 0] = 0
    Yi[Yi >=  image.shape[0]] = image.shape[0]-1
    Yi[Yi < 0] = 0

    image_transformed = np.zeros((size[0], size[1], 3))
    image_transformed[Yt.ravel(), Xt.ravel(), :] = image[Yi, Xi, :]
    
    return image_transformed

def extractInfoFromTag(tag):
    tag_size = tag.shape[0]
    grid_size = 8
    pixels_in_one_grid =  int(tag_size/8)

    info_with_padding = np.zeros((8,8))

    for i in range(0, grid_size, 1):
        for j in range(0, grid_size, 1):
            grid = tag[i*pixels_in_one_grid:(i+1)*pixels_in_one_grid, j*pixels_in_one_grid:(j+1)*pixels_in_one_grid]
            
            if np.sum(grid) > 100000*0.7 and np.median(grid) == 255:
                # print(np.sum(grid))
                info_with_padding[i,j] = 255
    # print(info_with_padding)
    info = info_with_padding[2:6, 2:6]
    return info


def applyHomography2Points(points, H):
    Xi = points[:, 0]
    Yi = points[:, 1]
    lin_homg_pts = np.stack((Xi, Yi, np.ones(Xi.size)))

    lin_homg_pts_trans = H.dot(lin_homg_pts)
    lin_homg_pts_trans /= lin_homg_pts_trans[2,:]

    Xt, Yt = lin_homg_pts_trans[:2,:].astype(int)
    points_trans = np.dstack([Xt, Yt])
    return points_trans

def applyHomography2ImageUsingForwardWarping(image, H, size, background_image = None):
    cols, rows = size
    h, w = image.shape[:2] 
    Yi, Xi = np.indices((h, w)) 
    lin_homg_pts = np.stack((Xi.ravel(), Yi.ravel(), np.ones(Xi.size)))
    trans_lin_homg_pts = H.dot(lin_homg_pts)
    trans_lin_homg_pts /= (trans_lin_homg_pts[2,:] + 1e-7)
    trans_lin_homg_pts = np.round(trans_lin_homg_pts).astype(int)


    if background_image is None:
        image_transformed = np.zeros((rows, cols, 3)) 
    else:
        image_transformed = background_image
    x1 = trans_lin_homg_pts[0,:]
    y1 = trans_lin_homg_pts[1,:]
    

    x1[x1 >= cols] = cols - 1
    y1[y1 >= rows] = rows - 1
    x1[x1 < 0] = 0
    y1[y1 < 0] = 0

    image_transformed[y1, x1] = image[Yi.ravel(), Xi.ravel()]
    image_transformed = np.uint8(image_transformed)
    return image_transformed


def computeProjectionMatrix(H, k_matrix_val):
    K_inv = np.linalg.inv(k_matrix_val)

    B_tilda = np.dot(K_inv, H)
    B_tilda_mod = np.linalg.norm(B_tilda)
    if B_tilda_mod < 0:
        B = -1  * B_tilda
    else:
        B =  B_tilda

    b1 = B[:,0]
    b2 = B[:,1]
    b3 = B[:,2]

    lambda_ = (np.linalg.norm(b1) + np.linalg.norm(b2))/2
    lambda_ = 1 / lambda_

    r1 = lambda_ * b1
    r2 = lambda_ * b2
    r3 = np.cross(r1, r2)
    t = lambda_ * b3

    P = np.array([r1,r2, r3, t]).T
    P = np.dot(k_matrix_val, P)
    P = P / P[2,3]
    return P

def applyProjectionMatrix2Points(points, P):
    Xi = points[:, 0]
    Yi = points[:, 1]
    Zi = points[:, 2]

    lin_homg_pts = np.stack((Xi, Yi, Zi, np.ones(Xi.size)))
    lin_homg_pts_trans = P.dot(lin_homg_pts)
    lin_homg_pts_trans /= lin_homg_pts_trans[2,:]
    x1 = lin_homg_pts_trans[0,:].astype(int)
    y1 = lin_homg_pts_trans[1,:].astype(int)

    projected_points = np.dstack((x1,y1)).reshape(4,2)
    return projected_points

def rotatePoints(points):
    point_list = list(points.copy())
    top = point_list.pop(-1)
    point_list.insert(0, top)
    return np.array(point_list)


def decodTag_Orient(tag):
    x = np.arange(0, 128, 16)
    y = np.arange(0, 128, 16)

    orientation = -1
    # Decode orientation based on location of white block in the space between 2x2 and 4x4 grid
    # Detect white block by checking whether mean >= 256/2 since 255 is highest value for all white pixels 
    if np.mean(np.reshape(tag[y[2]:y[2]+16, x[2]:x[2]+16],(256, 1))) >=128:
        orientation = np.pi
    elif np.mean(np.reshape(tag[y[2]:y[2]+16, x[5]:x[5]+16],(256, 1))) >=128:
        orientation = np.pi/2
    elif np.mean(np.reshape(tag[y[5]:y[5]+16, x[2]:x[2]+16],(256, 1))) >=128:
        orientation = 3*np.pi/2
    elif np.mean(np.reshape(tag[y[5]:y[5]+16, x[5]:x[5]+16],(256, 1))) >=128:
        orientation = 0
    else:
        print('Orientation Not found')
    
    # Choose inner 2x2 Grid
    x_bin = x[3:5]
    y_bin = y[3:5]

    # Based on Orientation, decode ID using binary representation (powers of 2)
    if orientation == 0:
        ID = 0

        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >=128:
            ID += 1
        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 2
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 4
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >=128:
            ID += 8
    elif orientation == np.pi/2:
        ID = 0

        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >=128:
            ID += 2
        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 4
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 8
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >=128:
            ID += 1
    elif orientation == np.pi:
        ID = 0

        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >= 128:
            ID += 4
        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 8
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 1
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >=128:
            ID += 2
    elif orientation == 3*np.pi/2:
        ID = 0

        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >=128:
            ID += 8
        if np.mean(np.reshape(tag[y_bin[0]:y_bin[0]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 1
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[1]:x_bin[1]+16],(256, 1))) >=128:
            ID += 2
        if np.mean(np.reshape(tag[y_bin[1]:y_bin[1]+16, x_bin[0]:x_bin[0]+16],(256, 1))) >=128:
            ID += 4
    else:
        ID = 0
        print("Orientation not identifiable and hence no ID found")
    return ID, orientation

def getOrientation_2(AR_block, margin = 10, decode = True):

    AR_block = AR_block[margin:-margin,margin:-margin]
    # _, AR_block = cv2.threshold(cv2.cvtColor(AR_block, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY) # only threshold
    # cropped_AR_block = crop_AR(AR_block)
    # cropped_AR_block  = cv2.resize(cropped_AR_block, (64,64))
    cropped_AR_block = AR_block

    # lowerright = cropped_AR_block[48:64,48:64]
    # lowerleft = cropped_AR_block[48:64,0:16]

    # upperright = cropped_AR_block[0:16,48:64]
    # upperleft = cropped_AR_block[0:16,0:16]

    lowerright = cropped_AR_block[48:64,48:64]
    lowerleft = cropped_AR_block[48:64,0:16]

    upperright = cropped_AR_block[0:16,48:64]
    upperleft = cropped_AR_block[0:16,0:16]

    UL,UR,LL,LR = np.int(np.median(upperleft)), np.int(np.median(upperright)), np.int(np.median(lowerleft)), np.int(np.median(lowerright))

    AR_orientationPattern = [UL,UR,LL,LR]
    orientations = [180,-90,90,0]

    index = np.argmax(AR_orientationPattern)

    orientation = orientations[index]

    if decode ==  True:
        rotated_AR_block = RotatebyOrientation(cropped_AR_block, orientation)
        
        block1 = rotated_AR_block[16:32,16:32]
        block2 = rotated_AR_block[16:32,32:48]
        block3 = rotated_AR_block[32:48,32:48]
        block4 = rotated_AR_block[32:48, 16:32]

        bit1 = np.median(block1)/255
        bit2 = np.median(block2)/255
        bit3 = np.median(block3)/255
        bit4 = np.median(block4)/255

#         print("Bit Value: ",bit1,bit2,bit3,bit4 )
        decodedValue = bit1*1 + bit2*2 + bit3*4 + bit4*8

    else:
        decodedValue = None
    return orientation, decodedValue, rotated_AR_block


def crop_AR(AR_block):

    """
    For a given AR tag, crop the black region
    
    """
    global prev_AR_block
    Xdistribution = np.sum(AR_block,axis=0)
    Ydistribution = np.sum(AR_block,axis=1)
    
    mdpt = len(Xdistribution)//2
    left_Xdistribution = Xdistribution[:mdpt]
    right_Xdistribution = Xdistribution[mdpt:]
    
    leftx,rightx,topx,topy = -1,-1,-1,-1
    for i in range(len(left_Xdistribution)):
        if left_Xdistribution[i] > 2000:
            leftx = i
            break

    for i in range(len(right_Xdistribution)):
        if right_Xdistribution[i] < 2000:
            rightx = i
            rightx+=mdpt
            break
    

    top_Ydistribution = Ydistribution[:mdpt]
    bottom_Ydistribution = Ydistribution[mdpt:]

    for i in range(len(top_Ydistribution)):
        if top_Ydistribution[i] > 2000:
            topy = i
            break

    for i in range(len(bottom_Ydistribution)):
        if bottom_Ydistribution[i] < 2000:
            bottomy = i
            bottomy+=mdpt
            break

    cropped_AR_block = AR_block[topy:bottomy,leftx:rightx]
    
    if (leftx < 0 )or(rightx < 0)or(topy < 0 )or(bottomy < 0):
        cropped_AR_block  = prev_AR_block
        print('bad tag found')
    else:
        prev_AR_block = cropped_AR_block        
        
    return cropped_AR_block

def RotatebyOrientation(Block, orientation):
    
    # rotateBlock by orientation degree
    if orientation == 90:
#         print("Rotated anticlckwise 90")
        Block = cv2.rotate(Block, cv2.cv2.ROTATE_90_CLOCKWISE) 

    elif orientation == -90:
#         print("Rotated clckwise 90")
        Block = cv2.rotate(Block, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    elif orientation == 180:
#         print("Rotated 180")
        Block = cv2.rotate(Block, cv2.cv2.ROTATE_180) 
    return Block


def drawGrids(block, step = 8):
    """
    ref: http://study.marearts.com/2018/11/python-opencv-draw-grid-example-source.html
    """
    
    block  = cv2.resize(block, (512,512))
    h,w = block.shape[:2]
    
    x = np.linspace(0, w, step).astype(np.int32)
    y = np.linspace(0, h, step).astype(np.int32)

    v_lines = []
    h_lines = []
    for i in range(step):
        v_lines.append( [x[i], 0, x[i], w-1] )
        h_lines.append( [0, int(y[i]), h-1, int(y[i])] )


    for i in range(step):
        [vx1, vy1, vx2, vy2] = v_lines[i]
        [hx1, hy1, hx2, hy2] = h_lines[i]

        block = cv2.line(block, (vx1,vy1), (vx2, vy2), (0,255,255),1 )
        block = cv2.line(block, (hx1,hy1), (hx2, hy2), (0,255,255),1 )
        
    return block


def main(): 

    Video_Path = "/home/prateek/ENPM673/Project1/1tagvideo.mp4"
    SaveVid_Testudo = "/home/prateek/ENPM673/Project1/Prob2A_Output/Prob2_Testudo_harris.avi"
    ProjectTestudo = 1 #0
    UseFilter = 0
    print("ProjectTestudo = ", ProjectTestudo)
    print("UseFilter = ", UseFilter)

    project_testudo = ProjectTestudo
    use_filter = UseFilter

    cap = cv2.VideoCapture(Video_Path)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    result = cv2.VideoWriter(SaveVid_Testudo,  
                            cv2.VideoWriter_fourcc(*'MJPG'), 
                            10, (frame_width, frame_height)) 

    testudoFileName = "testudo.png"
    testudo_image = cv2.imread(testudoFileName)
    if testudo_image is None:
        print("testudo image no found!")


    testudo_x = testudo_image.shape[1]
    testudo_y = testudo_image.shape[0]
    testudo_corners = np.array([[0,0], [0, testudo_y-1], [testudo_x-1, testudo_y-1], [testudo_x-1, 0]])
    testudo_corners = sortCorners(testudo_corners)

    tag_size = 160
    desired_tag_corner = sortCorners(np.array([ [0, tag_size-1], [tag_size-1, tag_size-1], [tag_size-1, 0], [0, 0]]))

    first_time = True
    window_size_base = 4
    window_size_top = 5
    count = 0
    rotation = 0 

    if use_filter:
        print("Apply filter to the points in cube")
        fps = cap.get(cv2.CAP_PROP_FPS)
        dt = 1/fps

    while(True):
        ret, frame = cap.read()
        if not ret:
            print("Video Stream Finished")
            break
        
        image_rgb = frame
        rows,cols,ch = image_rgb.shape        
        detected_april_tag = np.uint8(image_rgb) 
        image_show = image_rgb.copy()

        if first_time:
            old_corners = getTagCorners(detected_april_tag)
            number_of_tags = len(old_corners)
            maBase = MovingAverage(window_size_base, 10)
            maTop = MovingAverage(window_size_top, 5)

        corners = getTagCorners(detected_april_tag)
        print(corners)

        pos_tag = np.array([[0,0], [127,0], [127,127], [0,127]]) # Corners 

        HomoFind = computeHomography(corners, pos_tag)

        tag = applyHomography2ImageUsingInverseWarping(image_rgb, HomoFind, (tag_size, tag_size))
        tag = cv2.cvtColor(np.uint8(tag), cv2.COLOR_BGR2GRAY)
        ret,tag = cv2.threshold(np.uint8(tag), 230 ,255,cv2.THRESH_BINARY)

        tag_info = extractInfoFromTag(tag)
        ARcorners = np.array([tag_info[0,0], tag_info[0,3], tag_info[3,0], tag_info[3,3]])

        tag_id, tag_orient = decodTag_Orient(tag)

        if project_testudo:
            H12 = computeHomography(testudo_corners, corners)
            testudo_corners_trans = applyHomography2Points(testudo_corners, H12)
            cv2.drawContours(detected_april_tag, [testudo_corners_trans], 0, (0,255,255),3)
            testudo_transormed = applyHomography2ImageUsingForwardWarping(testudo_image, H12, (cols, rows), background_image = image_show)

        else: #projecting cube
            cube_height = np.array([-(tag_size-1), -(tag_size-1), -(tag_size-1), -(tag_size-1)]).reshape(-1,1)
            cube_corners = np.concatenate((desired_tag_corner, cube_height), axis = 1)
            Hdt = computeHomography(np.float32(desired_tag_corner), np.float32(corners))
            P = computeProjectionMatrix(Hdt, k_matrix_val)
            set2_top = applyProjectionMatrix2Points(cube_corners, P)

            x1, y1, z1 = np.matmul(P, np.array([0,0,0,1]))
            x2, y2, z2 = np.matmul(P, np.array([160,0,0,1]))
            x3, y3, z3 = np.matmul(P, np.array([160,160,0,1]))
            x4, y4, z4 = np.matmul(P, np.array([0,160,0,1]))
            x5, y5, z5 = np.matmul(P, np.array([0,160,-160,1]))
            x6, y6, z6 = np.matmul(P, np.array([0,0,-160,1]))
            x7, y7, z7 = np.matmul(P, np.array([160,0,-160,1]))
            x8, y8, z8 = np.matmul(P, np.array([160,160,-160,1]))

            # Draw Cube edges
            cv2.line(image_show, (int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (200,0,200),2)
            cv2.line(image_show, (int(x2/z2),int(y2/z2)),(int(x3/z3),int(y3/z3)), (200,0,200),2)
            cv2.line(image_show, (int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (200,0,200),2)
            cv2.line(image_show, (int(x1/z1),int(y1/z1)),(int(x4/z4),int(y4/z4)), (200,0,200),2)
            cv2.line(image_show, (int(x1/z1),int(y1/z1)),(int(x6/z6),int(y6/z6)), (200,0,200),2)
            cv2.line(image_show, (int(x2/z2),int(y2/z2)),(int(x7/z7),int(y7/z7)), (200,0,200),2)
            cv2.line(image_show, (int(x3/z3),int(y3/z3)),(int(x8/z8),int(y8/z8)), (200,0,200),2)
            cv2.line(image_show, (int(x4/z4),int(y4/z4)),(int(x5/z5),int(y5/z5)), (200,0,200),2)
            cv2.line(image_show, (int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (200,0,200),2)
            cv2.line(image_show, (int(x6/z6),int(y6/z6)),(int(x7/z7),int(y7/z7)), (200,0,200),2)
            cv2.line(image_show, (int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (200,0,200),2)
            cv2.line(image_show, (int(x8/z8),int(y8/z8)),(int(x5/z5),int(y5/z5)), (200,0,200),2)

        count = count + 1
        first_time = False

        cv2.imshow('frame', np.uint8(image_show))
        result.write(np.uint8(image_show)) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    result.release() 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


