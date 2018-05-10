# -*- coding:utf-8 -*-
'''
SRC: Any annotated roi(e.g. i write this script to generate bankcard samples with random background)
SRC_BG: Any accpectable background for the annotated samples(e.g. i use the aygmentated indoorCVPR_09 datasets for previous project)
SRC_TXT: Corresponding annotation file(ICDAR format) for all samples
DST: output samples as well as corresponding new annotation file.
'''
import os
import csv
import numpy as np
import cv2
import shutil
import random
SRC = "./src_img/"
SRC_BG = "./sc_bg_img/"
DST = "./dst/"
SRC_TXT = "./ori_annotation/"
skip_list = []
skipbg_list = []


# used to storage skip image list
# used to judge where key is "b"
state = 0
#global bbox_height
#global bbox_width
global label
global bg_cnt
def rad(x):
    return x*np.pi/180
	
def check_dir(path):
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)

def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            loaded_label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            #text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if loaded_label != '#' and int(loaded_label) != 1:
                text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            else:
                text_polys.append([[x1, y1], [x2, y2], [x4, y4], [x3, y3]])
            if loaded_label != '':
                text_tags.append(True)
            else:
                text_tags.append(False)
        print(len(text_polys))
        return np.array(text_polys, dtype=np.float32)

def writeFile(filename):
    global label

    f = open(filename, 'w')
    iter = 0
    label_cnt = 0
    for e in label:
        if iter==8:
            f.write(str(e)+"\n")
            iter=0
            label_cnt = label_cnt + 1
        else:
            f.write(str(e)+",")
            iter=iter+1
    label = []
    card_num = []
    f.close()

def synth_img(fg_image_path, bg_image_path, combine_title, bboxes):
    global label, bg_cnt
    fg_img_ori = cv2.imread(fg_image_path)
    bg_img_ori = cv2.imread(bg_image_path)

    fg_scale = 1.1#random.uniform(0.5, 1.0)
    fg_img = fg_img_ori.copy()
    fg_img = cv2.resize(fg_img_ori, dsize=(int(fg_scale*fg_img_ori.shape[1]), int(fg_scale*fg_img_ori.shape[0])),interpolation=cv2.INTER_AREA)
    #fg_img = cv2.copyMakeBorder(fg_img,dy_up,dy_down,dx_left,dx_right,cv2.BORDER_CONSTANT,0)
    fg_height,fg_width=fg_img.shape[0:2]
    bg_scale = 1.3
    #bg_img = cv2.resize(bg_img_ori, dsize=(int(bg_scale*fg_width), int(bg_scale*fg_height)))

    '''dx_left = 0
    dx_right = 0
    dy_up = 0
    dy_down = 0
    if bg_width>=fg_width and bg_height>=fg_height:
        left_ratio = 0.5#random.uniform(0.0, 1.0)
        up_ratio = 0.5#random.uniform(0.0, 1.0)
        dx_left = left_ratio*(bg_width-fg_width)
        dx_eight = (1-left_ratio)*(bg_width-fg_width)
        dy_up = up_ratio*(bg_height-fg_height)
        dy_down = (1-up_ratio)*(bg_height-fg_height)
    else:
        print('ERROR: bg image too small!...should not happen')
        return False'''
    z_times = random.randint(0,4)
    anglex = random.uniform(-10, 10)
    angley = random.uniform(-10, 10)
    anglez = 90*z_times
    anglez = anglez + random.uniform(-15,15)
    fov = 50
    z_value = random.randint(-0.5*max(fg_height,fg_width), 0.5*max(fg_height,fg_width))

    if (z_times%2) == 1:
        bg_img = cv2.resize(bg_img_ori, dsize=(int(bg_scale * fg_height), int(bg_scale * fg_width)))
    else:
        bg_img = cv2.resize(bg_img_ori, dsize=(int(bg_scale * fg_width), int(bg_scale * fg_height)))
    bg_height, bg_width = bg_img.shape[0:2]
    #about fov, please to cg tutorial
    z=np.sqrt(bg_width**2 + bg_height**2)/2/np.tan(rad(fov/2))
    #rotation angle of x/y/z dim
    rx = np.array([[1,                  0,                          0,                          0],
                   [0,                  np.cos(rad(anglex)),        -np.sin(rad(anglex)),       0],
                   [0,                 -np.sin(rad(anglex)),        np.cos(rad(anglex)),        0,],
                   [0,                  0,                          0,                          1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0,                         np.sin(rad(angley)),       0],
                   [0,                   1,                         0,                          0],
                   [-np.sin(rad(angley)),0,                         np.cos(rad(angley)),        0,],
                   [0,                   0,                         0,                          1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)),      0,                          0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)),      0,                          0],
                   [0,                  0,                          1,                          0],
                   [0,                  0,                          0,                          1]], np.float32)

    r = rx.dot(ry).dot(rz)

    #4 pairs of point to handle src/dst roi location
    pcenter = np.array([int(bg_width/2), int(bg_height/2), 0, 0], np.float32)
    
    p1 = np.array([(bg_width-fg_width)/2,(bg_height-fg_height)/2,  z_value,0], np.float32) - pcenter
    p2 = np.array([(bg_width+fg_width)/2,(bg_height-fg_height)/2,  z_value,0], np.float32) - pcenter
    p3 = np.array([(bg_width-fg_width)/2,(bg_height+fg_height)/2,  z_value,0], np.float32) - pcenter
    p4 = np.array([(bg_width+fg_width)/2,(bg_height+fg_height)/2,z_value,0], np.float32) - pcenter
    
    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0,0],
                    [fg_width,0],
                    [0,fg_height],
                    [fg_width,fg_height]], np.float32)
    
    dst = np.zeros((4,2), np.float32)

    #main transform process fro ROI
    for i in range(4):
        dst[i,0] = list_dst[i][0]*z/(z-list_dst[i][2]) + pcenter[0]
        dst[i,1] = list_dst[i][1]*z/(z-list_dst[i][2]) + pcenter[1]
    min_x = min(min(dst[0,0], dst[1, 0]), min(dst[2, 0], dst[3, 0]))
    max_x = max(max(dst[0,0], dst[1, 0]), max(dst[2, 0], dst[3, 0]))
    min_y = min(min(dst[0,1], dst[1, 1]), min(dst[2, 1], dst[3, 1]))
    max_y = max(max(dst[0,1], dst[1, 1]), max(dst[2, 1], dst[3, 1]))
    minx_res = min_x
    maxx_res = bg_width - max_x
    miny_res = min_y
    maxy_res = bg_height - max_y
    if minx_res>-10 and maxx_res>-10 and miny_res >-10 and maxy_res > -10:
        warpR = cv2.getPerspectiveTransform(org, dst)
        im_out = cv2.warpPerspective(fg_img, warpR, (bg_img.shape[1],bg_img.shape[0]), flags = cv2.INTER_AREA,borderMode=cv2.BORDER_TRANSPARENT)
        for i in range(bg_img.shape[1]):
            for j in range(bg_img.shape[0]):
                pixel = im_out[j, i]
                if not np.all(pixel == [0, 0, 0]):
                    bg_img[j, i] = im_out[j, i]
        #bg_img.save(DST + combine_title +".jpg")
        cv2.imwrite(DST + combine_title +".jpg", bg_img)
        label = []
        #print(len(bboxes))

        lt_coord = np.array([int((bg_width-fg_width)/2), int((bg_height-fg_height)/2), 0, 0], np.float32)
        for bbox in bboxes:
            #print(len(bbox))
            bbox_p1 = np.array([int(bbox[0][0]*fg_scale), int(bbox[0][1]*fg_scale), z_value, 0], np.float32) + lt_coord- pcenter
            bbox_p2 = np.array([int(bbox[1][0]*fg_scale), int(bbox[1][1]*fg_scale), z_value, 0], np.float32) + lt_coord- pcenter
            bbox_p3 = np.array([int(bbox[2][0]*fg_scale), int(bbox[2][1]*fg_scale), z_value, 0], np.float32) + lt_coord- pcenter
            bbox_p4 = np.array([int(bbox[3][0]*fg_scale), int(bbox[3][1]*fg_scale), z_value, 0], np.float32) + lt_coord- pcenter

            bbox_dst1 = r.dot(bbox_p1)
            bbox_dst2 = r.dot(bbox_p2)
            bbox_dst3 = r.dot(bbox_p3)
            bbox_dst4 = r.dot(bbox_p4)

            list_dst = [bbox_dst1, bbox_dst2, bbox_dst3, bbox_dst4]

            bbox_dst = np.zeros((4,2), np.float32)

            #main transform process fro control points
            for i in range(4):
                bbox_dst[i,0] = int(list_dst[i][0]*z/(z-list_dst[i][2]) + pcenter[0])
                bbox_dst[i,1] = int(list_dst[i][1]*z/(z-list_dst[i][2]) + pcenter[1])
            list_info = bbox_dst[0,0], bbox_dst[0,1], bbox_dst[1,0], bbox_dst[1,1],bbox_dst[2,0], bbox_dst[2,1],bbox_dst[3,0], bbox_dst[3,1]
            #print(len(list_info))
            for info in list_info:
                #print('label.append LINE 188')
                label.append(int(info))
            label.append('#')
            cv2.polylines(bg_img, [bbox_dst.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
        cv2.imwrite(DST + combine_title +"_label.jpg", bg_img)
        print('writeFile...')
        writeFile(DST + combine_title +".txt")
        if len(bboxes) == 1:
            bg_cnt = bg_cnt + 1
        else:
            bg_cnt = bg_cnt + 2
        #transform_annotation(bboxes, combine_title)
    else:
        print('bad warpPerspective...drop...')
    return True

def process(src_path, current_img, src_bg_path, current_bgimg, DST):
    #fg_image = cv2.imread(os.path.join(src_path, current_img))
    #bg_image = cv2.imread(os.path.join(src_bg_path, current_bgimg))
    print('processing:')
    print(current_img)
    print(current_bgimg)
    fg_image_path = os.path.join(src_path, current_img)
    bg_image_path = os.path.join(src_bg_path, current_bgimg)
    (fgtitle, ext) = os.path.splitext(current_img)
    (bgtitle, ext) = os.path.splitext(current_bgimg)
    fg_label_file = fgtitle + '.txt'
    combine_title = fgtitle + '_' + bgtitle
    fg_label_path = os.path.join(SRC_TXT, fg_label_file)
    load_annoatation_flag = True
    text_polys = load_annoataion(fg_label_path)
    if not load_annoatation_flag:
        print('annotation file not found:')
        print(fg_label_path)
        return False
    for text_poly in text_polys:
        # print('text_poly:')
        # print(text_poly)
        # print('text_poly[2][0]:')
        # print(text_poly[2][0])
        # print('text_poly[3][0]:')
        # print(text_poly[3][0])
        if text_poly[2][0]<text_poly[3][0]:
            # print('text_poly[2][0]<text_poly[3][0]!')
            temp_poly = text_poly[2][0]
            text_poly[2][0] = text_poly[3][0]
            text_poly[3][0] = temp_poly
            temp_poly = text_poly[2][1]
            text_poly[2][1] = text_poly[3][1]
            text_poly[3][1] = temp_poly
        # print('text_poly[2][0]:')
        # print(text_poly[2][0])
        # print('text_poly[3][0]:')
        # print(text_poly[3][0])
    synth_flag = True#synth_img(fg_image_path, bg_image_path, combine_title, text_polys)
    if not synth_flag:
        print('synth_img error while processing:')
        print(fg_image_path)
        print(bg_image_path)
        return False
    return True

if __name__ == '__main__':

    global current_img_name, label
    check_dir(SRC)
    check_dir(SRC_BG)

    label=[]
    image_file_list = os.listdir(SRC)
    bgimage_file_list = os.listdir(SRC_BG)

    x1, y1, x2, y2 = 0, 0, 0, 0  # bbox coordinate
    # for image_file in os.listdir(src_path):
    scale_index = 0
    status = True
    fg_cnt = 0
    bg_cnt = 0
    for fg_cnt in range(0, len(image_file_list)):
        image_file = image_file_list[fg_cnt]
        while bg_cnt<1:#bg count per samples...
            #print(fg_cnt)
            #print(bg_cnt)
            bgimage_file = bgimage_file_list[random.randint(0, len(bgimage_file_list)-1)]
            current_img = image_file
            current_bgimg = bgimage_file
            #main process
            status = process(SRC, current_img, SRC_BG, current_bgimg, DST)
            if status == True:
                bg_cnt = bg_cnt+1
                continue
            else:
                print('ERROR occurs while processing:')
                print(current_img)
                print(current_bgimg)
                exit()
        else:
            bg_cnt = 0