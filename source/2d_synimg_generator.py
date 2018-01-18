# -*- coding:utf-8 -*-

import os
import numpy as np
from PIL import Image, ImageTk
from tkinter import Tk, Label, Frame, Entry, StringVar
import cv2
import shutil
import random
SRC = "src_img"
SRC_BG = "sc_bg_img"
DST = "dst"
skip_list = []
skipbg_list = []


# used to storage skip image list
# used to judge where key is "b"
state = 0
#global bbox_height
#global bbox_width
global label
enable_debug = False

def writeFile(filename):
    global label
    f = open(filename, 'w')
    iter = 0
    label_cnt = 0
    if enable_debug:
        print('label length %d' % len(label))
    for e in label:
        f.write(str(e)+",")
    f.write("roi"+"\n")
    label = []
    f.close()
	
def check_dir(path):
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)


def process(src_path, current_img, src_bg_path, current_bgimg, DST):
    global label
    #fg_image = cv2.imread(os.path.join(src_path, current_img))
    #bg_image = cv2.imread(os.path.join(src_bg_path, current_bgimg))
    fg_image = Image.open(os.path.join(src_path, current_img))
    bg_image = Image.open(os.path.join(src_bg_path, current_bgimg))
    (fgtitle, ext) = os.path.splitext(current_img)
    (bgtitle, ext) = os.path.splitext(current_bgimg)
    cardnum_title = fgtitle.split('_')[1]
    bgnum_title = bgtitle#.split('_')[1]
    dst_name = cardnum_title + '_'+ bgnum_title
    translate_x_scale = random.uniform(-0.5, 0.2)
    translate_y_scale = random.uniform(-0.5, 0.2)
    #roi_ratio = random.uniform(0.25, 0.9)
    (bgimg_width, bgimg_height) = bg_image.size
    (fgimg_width, fgimg_height) = fg_image.size
    '''bgimg_width = bg_image.shape[1]
    bgimg_height = bg_image.shape[0]
    fgimg_width = fg_image.shape[1]
    fgimg_height = fg_image.shape[0]'''

    roi_minx = int(bgimg_width*(translate_x_scale + 0.5))
    roi_miny = int(bgimg_height*(translate_y_scale + 0.5))
    res_width = bgimg_width - roi_minx - 1
    res_height = bgimg_height - roi_miny -1
    max_width = min(bgimg_width*0.95,  res_width)
    max_height = min(bgimg_height*0.95,  res_height)
    max_ratio = min(max_width/fgimg_width, max_height/fgimg_height)
    min_ratio = 0.4
    roi_ratio = random.uniform(min_ratio, max_ratio)
    roi_maxx = int(min(bgimg_width*0.95, roi_minx + roi_ratio*fgimg_width))
    roi_maxy = int(min(bgimg_height*0.95, roi_miny + roi_ratio*fgimg_height))
    box =(roi_minx, roi_miny, roi_maxx, roi_maxy)
    print('roi:')
    print(box)
    region = fg_image.resize((box[2] - box[0], box[3] - box[1]))
    bg_image.paste(region, box)
    bg_image.save(DST + "/"+ dst_name +".jpg")

    list_info = roi_minx, roi_miny, roi_maxx, roi_miny, roi_maxx, roi_maxy, roi_minx, roi_maxy
    for info in list_info:
        label.append(info)
    writeFile(DST + "/"+ dst_name +".txt")
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
        for bg_cnt in range(0, 2):
            #print(fg_cnt)
            #print(bg_cnt)
            bgimage_file = bgimage_file_list[random.randint(0, len(bgimage_file_list)-1)]
            current_img = image_file
            current_bgimg = bgimage_file
            # do process
            status = process(SRC, current_img, SRC_BG, current_bgimg, DST)
            if status == True:
                continue
            else:
                print('ERROR occurs while processing:')
                print(current_img)
                print(current_bgimg)
                exit()
