
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import argparse
import os

def drawGrids(block, step = 9):
    
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


def findOrientation(AR_block, margin = 10, decode = True):
    
    AR_block = AR_block[margin:-margin,margin:-margin]
    _, AR_block = cv2.threshold(cv2.cvtColor(AR_block, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY) # only threshold
    cropped_AR_block = crop_AR(AR_block)
    cropped_AR_block  = cv2.resize(cropped_AR_block, (64,64))

    lowerright = cropped_AR_block[48:64,48:64]
    lowerleft = cropped_AR_block[48:64,0:16]

    upperright = cropped_AR_block[0:16,48:64]
    upperleft = cropped_AR_block[0:16,0:16]

    UL,UR,LL,LR = np.int32(np.median(upperleft)), np.int32(np.median(upperright)), np.int32(np.median(lowerleft)), np.int32(np.median(lowerright))

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


def main():
    
    
    VideoPath = "/home/prateek/ENPM673/Project1/1tagvideo.mp4"
    SavePath = "/home/prateek/ENPM673/Project1/Prob1B_Output/"
    ReferencePath = "ref_marker.png"
    if(not (os.path.isdir(SavePath))):
        os.makedirs(SavePath)
    

    ref_AR_block = cv2.imread(ReferencePath)
    ref_AR_block  = cv2.resize(ref_AR_block, (64,64))
    grid_AR_block = drawGrids(ref_AR_block,9)
    orientation,decodedValue,rotated_AR_block = findOrientation(ref_AR_block,decode=True)  
    

    fig,plts = plt.subplots(1,3,figsize = (15,5))
    plts[0].imshow(ref_AR_block)    
    plts[0].axis('off')
    plts[0].title.set_text('a) Input Reference AR Tag')
    
    plts[1].imshow(grid_AR_block)
    plts[1].axis('off')
    plts[1].title.set_text('b) AR tag with a grid')
    
    plts[2].imshow(rotated_AR_block, cmap='gray')
    plts[2].axis('off')
    plts[2].text(24,54,"Decoded Value: "+str(decodedValue))
    plts[2].title.set_text('c) Rotated by '+str(orientation)+" degrees")
    
    fig.savefig(SavePath+'AR_Decoded.png')
    print("Check ", SavePath," for Problem1b Results")
if __name__ == '__main__':
    main()
  