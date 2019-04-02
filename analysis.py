import cv2
import numpy as np
import time
import os
from affine_ransac import Ransac
from affine_transform import Affine

def split_to_frames(video, dst_path):
    cap = cv2.VideoCapture(video)
    print (cap.get(cv2.CAP_PROP_FPS))
    print (cap.get(3),cap.get(4))
    i = 0
    while(1):
        ret, frame = cap.read()
        if frame is None:
            break
        if i < start or i > end:
            i += 1
            continue
        temp = cv2.transpose(frame)
        frame = cv2.flip(temp, 1)
        cv2.imwrite(dst_path+video+'_frame_{}.jpg'.format(i),frame)
        i +=1
        print (i)
        #if i == 100:
            #break
    cap.release()
    return i

def add_frame_info(video_name):
    frame_path = './frames/'
    label_frame_path = './labeled_frames/'
    frame_count = split_to_frames(video_name, frame_path)
    print ('frame count is ', frame_count)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter( './labeled_'+video_name[2:-3]+'avi', fourcc, 30.0, (1080,1920))
    #video = cv2.VideoWriter( './labeled_'+video_name[2:-3]+'avi', fourcc, 30.0, (1080,1920))
    i = 0
    while(1):
        if i < start:
            i += 1
            continue
        if i > end:
            break
        img = cv2.imread(frame_path+video_name[2:]+'_frame_{}.jpg'.format(i))
        print ((frame_path+video_name[2:]+'_frame_{}.jpg'.format(i)))
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'frame_'+ str(i), (0,40),font,1, (0,0,255), 3)
        video.write(img)
        i += 1
    video.release()


def gen_label_feature(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(img_gray, None)
    kp = np.array([p.pt for p in kp]).T

    return kp, desc


def read_mask(mask_name, label_name):
    valid_pos = []
    mask = cv2.imread(mask_name)
    for i,x in enumerate(mask):
        for j,y in enumerate(x):
            if mask[i][j][0] == 0 and mask[i][j][1] == 0 and mask[i][j][2] == 0:
                valid_pos.append([i,j])

    print ('valid pos is ', len(valid_pos))
    label = cv2.imread(label_name)
    mask_p, mask_d = gen_label_feature(label)
    print ('origianl mask point count is ', len(mask_p[0]))
    temp1 = [[],[]]
    temp2 = []
    for i,p in enumerate(mask_p[0]):
        if i % 100 == 0:
            print (i)
        pos = [int(mask_p[0][i]), int(mask_p[1][i])]
        if pos in valid_pos:
            temp1[0].append(mask_p[0][i])
            temp1[1].append(mask_p[1][i])
            temp2.append(mask_d[i])
    mask_p = np.array(temp1)
    mask_d = np.array(temp2)
    print ('valid mask point count is ', len(mask_p[0]))

    return valid_pos, mask_p, mask_d

def match_SIFT(desc_s, desc_t):
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(desc_s,desc_t)
    fit_pos = np.array([], dtype=np.int32).reshape((0, 2))
    matches_num = len(matches)
    for i in range(matches_num):
            temp = np.array([matches[i].queryIdx,
                             matches[i].trainIdx])
            fit_pos = np.vstack((fit_pos, temp))

    return fit_pos

def match_with_mask(img_path, valid_pos, mask_p, mask_d, result_path):
    img = cv2.imread(img_path)
    print('gen label feature')
    img_p, img_d = gen_label_feature(img)
    temp1 = [[],[]]
    temp2 = []
    print ('original img pos count is ', len(img_p[0]))
    for i,x in enumerate(img_p[0]):
        if i % 100 ==0:
            print (i)
        pos = [int(img_p[0][i]), int(img_p[1][i])]
        if pos in valid_pos:
            temp1[0].append(img_p[0][i])
            temp1[1].append(img_p[1][i])
            temp2.append(img_d[i])
    img_p = np.array(temp1)
    img_d = np.array(temp2)
    print ('valie img pos count is ', len(img_p[0]))

    fit_pos = match_SIFT(img_d, mask_d)
    print ('fit count is ',len(fit_pos))
    img_p = img_p[:,fit_pos[:,0]]
    mask_p = mask_p[:,fit_pos[:,1]]
    start = time.time()
    _, _, inliers = Ransac(3, 1).ransac_fit(img_p, mask_p)
    print ('inliners count is ', len(inliers[0]))
    img_p = img_p[:, inliers[0]]
    mask_p = mask_p[:, inliers[0]]
    A, t = Affine().estimate_affine(img_p, mask_p)
    M = np.hstack((A, t))
    end = time.time()

    rows, cols, _ = img.shape
    warp = cv2.warpAffine(img, M, (cols, rows))
    print (result_path+img_path[9:])
    cv2.imwrite(result_path+img_path[9:],warp)

start = 640
end = 2760
video_name = './IMG_3063.MOV'
#split_to_frames('./IMG_3063.MOV', './frames/')
#add_frame_info('./IMG_3063.MOV')
valid_pos, mask_p, mask_d = read_mask('./label/mask_2.png', './label/label.jpg')
#match_with_mask( './frames/IMG_3063.MOV_frame_999.jpg', valid_pos, mask_p, mask_d, './result/')

for i in range(start,end):
    img_name  = './frames/'+video_name[2:]+'_frame_{}.jpg'.format(i)
    print (img_name)
    match_with_mask( img_name, valid_pos, mask_p, mask_d, './result/')
    print ('total is ', end-start, 'current is ', i)
    

