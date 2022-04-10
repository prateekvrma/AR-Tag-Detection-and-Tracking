'''
ENPM673 - Project 2: Problem 2(A) 
- using cv2.findContour() function.
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

def getCube(image, bottom_points, top_points):
    cv2.drawContours(image, [bottom_points], 0, (255, 0 ,255),3)
    cv2.drawContours(image, [top_points], 0, (255, 0, 255),3)

    for i in range(0, bottom_points.shape[0]):
        color = (int(255/(i+1)), 0, int(255/(i+1)))
        cv2.line(image, (bottom_points[i,0], bottom_points[i,1]), (top_points[i,0], top_points[i,1]), color, 3)
    return image

def CornerSort(points):

    x_sorted = points[np.argsort(points[:,0])]
    points_left = x_sorted[0:2, :]
    points_right = x_sorted[2:4, :]

    left_sorted_y = points_left[np.argsort(points_left[:,1])]
    tl, bl = left_sorted_y

    right_sorted_y = points_right[np.argsort(points_right[:,1])]
    tr, br = right_sorted_y
    points_sorted = np.array([tl, bl, br, tr])
    return points_sorted

def getTagMask(image):
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (21, 21), 0)

    chosen_contours = findContour(image_gray)
    out_mask = np.zeros_like(image_gray)
    corners = []
    for chosen_contour in chosen_contours:
        corner = cv2.approxPolyDP(chosen_contour, 0.009 * cv2.arcLength(chosen_contour, True), True)
        corners.append(corner.reshape(-1,2))
        cv2.drawContours(out_mask, [chosen_contour], -1, 1, cv2.FILLED)  

    out_mask_mul = np.dstack((out_mask, out_mask, out_mask))
    detected_april_tag = image * out_mask_mul
    return detected_april_tag

def findContour(image):
    
    ret,thresh = cv2.threshold(np.uint8(image), 200 ,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    chosen_contours = []

    previous_area = cv2.contourArea(contours[0])
    for j in range(len(contours)):
        if hierarchy[0, j, 3] == -1:#no parent
            if hierarchy[0, j, 2] !=-1: #child
                #print("no parent, child present")
                area = cv2.contourArea(contours[j])
                if True: #np.abs(area - previous_area) < 1000:
                    chosen_contours.append(contours[j])
                    previous_area = area
    return chosen_contours

def getTagCorners(image):

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.medianBlur(image_gray, 3)

    (T, thresh) = cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, hierarchy=cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ctr=[]
    if hierarchy is not None:
        for j, cnt in zip(hierarchy[0], contours):
            cnt_len = cv2.arcLength(cnt,True)
            cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len,True)

            if cv2.isContourConvex(cnt) and len(cnt) == 4 and cv2.contourArea(cnt) > 500 :
                cnt=cnt.reshape(-1,2)
                #if j[0] == -1 and j[1] == -1 and j[3] != -1:
                if j[2] != -1 and j[3] != -1:
                    ctr.append(cnt)
    return ctr

def computeHomography(corners1, corners2):

    if (len(corners1) < 4) or (len(corners2) < 4):
        print("Atleast four points needed to compute SVD.")
        return 0

    x1 = corners1[:,0]
    y1 = corners1[:,1]
    x2 = corners2[:,0]
    y2 = corners2[:,1]

    nrows = 8
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
    return H


def applyHomography2ImageUsingInverseWarping(image, H, size):

    Yt, Xt = np.indices((size[0], size[1]))
    lin_homg_pts_trans = np.stack((Xt.ravel(), Yt.ravel(), np.ones(Xt.size)))

    H_inv = np.linalg.inv(H)
    lin_homg_pts = H_inv.dot(lin_homg_pts_trans)
    lin_homg_pts /= lin_homg_pts[2,:]

    Xi, Yi = lin_homg_pts[:2,:].astype(int)
    Xi[Xi >=  image.shape[1]] = image.shape[1]
    Xi[Xi < 0] = 0
    Yi[Yi >=  image.shape[0]] = image.shape[0]
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


def main(): 

    Video_Path = "/home/prateek/ENPM673/Project1/1tagvideo.mp4"
    SaveVid_Testudo = "/home/prateek/ENPM673/Project1/Prob2A_Output/Prob2_Testudo.avi"
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
    testudo_corners = CornerSort(testudo_corners)
    tag_size = 160
    desired_tag_corner = CornerSort(np.array([ [0, tag_size-1], [tag_size-1, tag_size-1], [tag_size-1, 0], [0, 0]]))

    first_time = True
    window_size_base = 4
    window_size_top = 5
    count = 0
    rotation = 0 

    if use_filter:
        print("Aplly filter to cube")
        fps = cap.get(cv2.CAP_PROP_FPS)
        dt = 1/fps

    while(True):
        
        ret, frame = cap.read()
        if not ret:
            print("Stream ended..")
            break
        
        image_rgb = frame
        rows,cols,ch = image_rgb.shape        
        detected_april_tag = np.uint8(getTagMask(image_rgb))

        if first_time:
            old_corners = getTagCorners(detected_april_tag)
            number_of_tags = len(old_corners)
            maBase = MovingAverage(window_size_base, 10)
            maTop = MovingAverage(window_size_top, 5)

        corners = getTagCorners(detected_april_tag)

        if(len(corners) < 1):
            corners = old_corners
        else:
            old_corners = corners

        image_show = image_rgb.copy()
        for corner in corners:
            set1 = testudo_corners
            set2 = CornerSort(corner)#from video

            if use_filter:
                if maBase.getListLength() <  window_size_base:
                    maBase.addQuadrilateral(set2)
                else:
                    maBase.addQuadrilateral(set2)
                    set2 = maBase.getAverage().astype(int)

            Htd = computeHomography(np.float32(set2), np.float32(desired_tag_corner))

            tag = applyHomography2ImageUsingInverseWarping(image_rgb, Htd, (tag_size, tag_size))
            tag = cv2.cvtColor(np.uint8(tag), cv2.COLOR_BGR2GRAY)
            ret,tag = cv2.threshold(np.uint8(tag), 230 ,255,cv2.THRESH_BINARY)
            tag_info = extractInfoFromTag(tag)
            ARcorners = np.array([tag_info[0,0], tag_info[0,3], tag_info[3,0], tag_info[3,3]])

            rotation = 0 
            if np.sum(ARcorners) == 255:
                while not tag_info[3,3]:
                    tag_info = np.rot90(tag_info, 1)
                    rotation = rotation + 90
                if first_time:
                    old_rotation = rotation                
                f_rotate = np.abs(old_rotation - rotation)
                if f_rotate == 270:
                    f_rotate = 90

                if (f_rotate > 100): #basically, greater than 90#REVIEW
                    print("High Rotation", f_rotate)
                    # rotation = old_rotation

                old_rotation = rotation
            else:
                print("No filter used!!")
                rotation = old_rotation

            num_rotations = int(rotation/90)
            for n in range(num_rotations):
                set2 = rotatePoints(set2)

            if project_testudo:
                f_Homo = computeHomography(set1, set2)
                set1_trans = applyHomography2Points(set1, f_Homo)
                cv2.drawContours(detected_april_tag, [set1_trans], 0, (0,255,255),3)
                testudo_transormed = applyHomography2ImageUsingForwardWarping(testudo_image, f_Homo, (cols, rows), background_image = image_show)

            else: #projecting cube
                cube_height = np.array([-(tag_size-1), -(tag_size-1), -(tag_size-1), -(tag_size-1)]).reshape(-1,1)
                cube_corners = np.concatenate((desired_tag_corner, cube_height), axis = 1)
                Hdt = computeHomography(np.float32(desired_tag_corner), np.float32(set2))
                P = computeProjectionMatrix(Hdt, k_matrix_val)
                set2_top = applyProjectionMatrix2Points(cube_corners, P)

                if use_filter:
                    if maTop.getListLength() <  window_size_top:
                        maTop.addQuadrilateral(set2_top)
                    
                    else:
                        maTop.addQuadrilateral(set2_top)
                        set2_top = maTop.getAverage().astype(int)

                image_show = getCube(image_rgb, set2, set2_top)

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


