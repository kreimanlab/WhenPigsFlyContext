#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:14:58 2020

@author: mengmi
"""

import IPython.display
# cd into virtualhome repo
import sys
sys.path.append('../simulation/')
from unity_simulator.comm_unity import UnityCommunication
import PIL
import numpy as np
from collections import defaultdict
import cv2
import os
import math
import pickle
import random

def display_grid_img(images_old, nrows=1):
    images = [x for x in images_old]
    h, w, _ = images[0].shape
    ncols = int((len(images)+nrows-1)/nrows)
    missing = ncols - (len(images)%ncols)
    for m in range(missing):
        images.append(np.zeros((h, w, 3)).astype(np.uint8))
    img_final = []
    for it_r in range(nrows):
        init_ind = it_r * ncols 
        end_ind = init_ind + ncols
        images_take = [images[it] for it in range(init_ind, end_ind)]
        img_final.append(np.concatenate(images_take, 1))
    img_final = np.concatenate(img_final, 0)
    img_final = PIL.Image.fromarray(img_final[:,:,::-1])
    return img_final

def display_scene_modalities(img_height, img_width, 
    comm, ids, modalities=['normal', 'seg_class', 'seg_inst', 'depth'], nrows=1):
    # Check the number of cameras
    _, ncameras = comm.camera_count()
    #print(ncameras)
    cameras_select = list(range(ncameras))
    cameras_select = [cameras_select[x] for x in ids]
    imgs_modality = []
    for mode_name in modalities:
        (ok_img, imgs) = comm.camera_image(cameras_select, mode=mode_name, image_width=img_height, image_height=img_width)
        #print(imgs)
        if mode_name == 'depth':
            #imgs = [((x/np.max(x))*255.).astype(np.uint8) for x in imgs]
            imgs = [(x*255.).astype(np.uint8) for x in imgs]
        imgs_modality += imgs
    img_final = display_grid_img(imgs_modality, nrows=nrows)
    return img_final

def find_nodes(graph, **kwargs):
    if len(kwargs) == 0:
        return None
    else:        
        k, v = next(iter(kwargs.items()))        
        return [n for n in graph['nodes'] if n[k] == v]

def find_nodes_byclassname(graph, classname):
    return [n for n in graph['nodes'] if n['class_name'] == classname]
    
def find_nodes_byid(graph, idnum):
    return [n for n in graph['nodes'] if n['id'] == idnum]
    
def find_edges(graph, **kwargs):
    if len(kwargs) == 0:
        return None
    else:
        k, v = next(iter(kwargs.items()))
        return [n for n in graph['edges'] if n[k] == v]
    
def find_allRooms(graph):
    return [n for n in graph['nodes'] if n['category'] == 'Rooms']
    
def find_rooms(graph, fromnode):   
    roomnodes = find_allRooms(graph)
    if fromnode['category'] != 'Rooms':
        for node in roomnodes:
            bboxroom = node['bounding_box']
            bboxobj = fromnode['bounding_box']
            status = checkTwo3DBboxOverlap(bboxobj, bboxroom)            
            if status:                
                return node['class_name']
            
    return fromnode['class_name']

def find_rooms_graphedges(graph, fromnode):    
    while fromnode['category'] != 'Rooms':
        objedge = find_edges(graph, from_id = fromnode['id'])[0]
        fromnode_id = objedge['to_id']
        fromnode = find_nodes_byid(graph, fromnode_id)[0]
    return fromnode

def displayAllBbox(img_height, img_width, JasonData, img):
    #convert to cv2 image and ready to draw
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for infor in JasonData.items():
        left = infor[1]['bbox'][2]
        top = infor[1]['bbox'][0]
        right = infor[1]['bbox'][3]
        bottom = infor[1]['bbox'][1]
        color = (0, 0, 255) 
        thick = 3
        label = infor[1]['class_name'] +', ' + infor[1]['roomtype'] 
        cv2.rectangle(img,(left, top), (right, bottom), color, thick)
        cv2.putText(img, label, (left, top - 12), 0, 1e-3 * img_width, color, thick//3)
    status = True
    return status, img    
    
def displayTargetBbox(img_height, img_width, JasonData, img, targetid, textflag, boxflag):
    #convert to cv2 image and ready to draw
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    status = False
    for infor in JasonData.items():
        if infor[1]['prefab_id'] == targetid:            
            left = infor[1]['bbox'][2]
            top = infor[1]['bbox'][0]
            right = infor[1]['bbox'][3]
            bottom = infor[1]['bbox'][1]
            targetbbox = [left, top, right, bottom]
            color = (0, 0, 255) 
            thick = 3
            label = infor[1]['class_name'] +', ' + infor[1]['roomtype'] 
            if boxflag:
                cv2.rectangle(img,(left, top), (right, bottom), color, thick)
            if textflag:
                cv2.putText(img, label, (left, top - 12), 0, 1e-3 * img_width, color, thick//3)
            status = True    
            targetarea = infor[1]['area']#(bottom - top)*(right - left)
            return status, targetarea, targetbbox, img
    return status, 0, 0, img

def extractColorInstanceTable(graph, message_color):
    ColorInstLookUpTab = {}
    for prefab_id in message_color:
        prefab_id = int(prefab_id)
        #print(type(prefab_id))
        objcolor_sm = message_color.get(str(prefab_id)) #color range from [0,1]
        #print(objcolor_sm)
        objcolor = np.round(np.array(objcolor_sm['Item1'], dtype=np.float32)*255.0).astype(np.uint8) #color range from [0,255]
        objcolor = tuple(objcolor)
        objnode = find_nodes_byid(graph, prefab_id)[0]
        infor = {}
        infor['prefab_id'] = prefab_id
        infor['prefab_name'] = objnode['prefab_name']
        infor['class_name'] = objnode['class_name']
        infor['category'] = objnode['category']
        roomname = find_rooms(graph, objnode)
        infor['roomtype'] = roomname
        ColorInstLookUpTab[objcolor] = infor
        
    return ColorInstLookUpTab

def extractJasonInstanceTable(img_inst_pil, img_inst_np, ColorInstLookUpTab):
    img_inst_color_tab = defaultdict(int)
    for pixel in img_inst_pil.getdata():    
        img_inst_color_tab[pixel] +=1
    
    [imgw, imgh, imgc] = img_inst_np.shape
    #consolidate all objects infor on image and output jasondata for this image
    JasonData = {}
    for pixel in img_inst_color_tab:
        if pixel in ColorInstLookUpTab.keys():
            X,Y = np.where(np.all(img_inst_np==np.asarray(pixel),axis=2))
            bbox = [min(X), max(X), min(Y), max(Y)]
            instinfor = ColorInstLookUpTab.get(pixel)
            infor = {}
            infor['prefab_id'] = instinfor['prefab_id']
            infor['prefab_name'] = instinfor['prefab_name']
            infor['class_name'] = instinfor['class_name']
            infor['roomtype'] = instinfor['roomtype']
            infor['category'] = instinfor['category']
            infor['bbox'] = bbox        
            infor['color'] = pixel
            infor['area'] = img_inst_color_tab.get(pixel)*1.0/(imgw*imgh) #ratio of isntance area on the entire image
            JasonData[pixel] = infor
            
    return JasonData
    
def convertPILImageToNumpyImage(img_all_pil, img_height, img_width):
    #img contains modalities=['normal', 'seg_class', 'seg_inst'], nrows=3
    #split into three images (normal, seg_class, seg_instance)    
    img_ori_pil = img_all_pil.crop((0, img_width*0, img_height, img_width*1))
    img_class_pil = img_all_pil.crop((0, img_width*1, img_height, img_width*2))
    img_inst_pil = img_all_pil.crop((0, img_width*2, img_height, img_width*3)) 
    #convert to numpy array
    img_ori_np = np.array(img_ori_pil)
    img_class_np = np.array(img_class_pil)
    img_inst_np = np.array(img_inst_pil)
    
    return img_ori_pil, img_class_pil, img_inst_pil, img_ori_np, img_class_np, img_inst_np

def IsHighContrast(img_height, img_width, ThresContrast, RatioCroppedContrast, JasonData, img, targetid):
    
    #convert to cv2 image and ready to draw
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    
    imgY = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
    
    status = False
    for infor in JasonData.items():
        if infor[1]['prefab_id'] == targetid:            
            left = infor[1]['bbox'][2]
            top = infor[1]['bbox'][0]
            right = infor[1]['bbox'][3]
            bottom = infor[1]['bbox'][1]
            #print(infor[1]['bbox'])
            width = bottom - top
            height = right - left          
            if int(left-RatioCroppedContrast*height) <0:
                left = 0
            else:
                left = int(left-RatioCroppedContrast*height)
            
            if int(right+RatioCroppedContrast*height) > (img_height-1):
                right = img_height - 1
            else:
                right = int(right+RatioCroppedContrast*height)
            
            if int(top-RatioCroppedContrast*width) <0:
                top = 0
            else:
                top = int(top-RatioCroppedContrast*width)
                
            if int(bottom+RatioCroppedContrast*width) > (img_width-1):
                bottom = img_width-1
            else:
                bottom = int(bottom+RatioCroppedContrast*width)            

            cropped_imgY = imgY[top:bottom, left:right]
            # compute min and max of Y
            #print(cropped_imgY.shape)
            if cropped_imgY.shape[0] == 0 or cropped_imgY.shape[1] == 0:
                return False
            Ymin = np.min(cropped_imgY)
            Ymax = np.max(cropped_imgY)
            #print(Ymin)
            #print(Ymax)
            # compute contrast
            contrast = (Ymax-Ymin)/(Ymax+Ymin)
            #print(contrast)
            
            if contrast > ThresContrast:
                status = True    
    return status

def checkCameraImageFitness(JasonData, targetprefabid, ThresRoomArea):
    #two criterias for a good pic:
    #1. the target object is on the pic
    #2. the camera is mostly looking at one room (not crossing two rooms); ThresRoomArea
    statusTarget = False #cond1 flag
    statusRoom = False #cond2 flag
    #keep track of total areas for each room type
    roomarea = defaultdict(float)
    for infor in JasonData.items():
        roomarea[infor[1]['roomtype']] += infor[1]['area']
        if infor[1]['prefab_id'] == targetprefabid:            
            statusTarget = True
            targetroom = infor[1]['roomtype']
            
    if not statusTarget:
        #print('Target not in pic')
        return False
    else:
        return True
#        otherarea = 0.0
#        for roomtype in roomarea:
#            if roomtype != targetroom:
#                otherarea += roomarea.get(roomtype)         
#        if otherarea <= ThresRoomArea:
#            statusRoom = True
#        
#        if not statusRoom:
#            print('contain too many rooms!')
#        return statusTarget & statusRoom
 
def checkCameraImageBlackSky(img_ori_np, ThresBlackSkyArea):
    [imgw, imgh, imgc] = img_ori_np.shape
    X,Y = np.where(np.all(img_ori_np==np.asarray([0,0,0]),axis=2))
    area = len(X)*1.0/(imgw*imgh)
    if area >= ThresBlackSkyArea: 
        return False
    else:
        return True
    
def IsTargetCollision(JasonData, graph, target_id):
    targetnode = find_nodes_byid(graph, target_id)[0] 
    targetbbox = targetnode['bounding_box']  
                
    for infor in JasonData.items():
        if infor[1]['prefab_id'] == target_id:
            continue
        elif infor[1]['category'] == 'Rooms':
            continue
        else:
            objbbox = find_nodes_byid(graph, infor[1]['prefab_id'])[0]['bounding_box']            
            status = checkTwo3DBboxOverlap(targetbbox, objbbox) or checkTwo3DBboxOverlap(objbbox, targetbbox)
            if status:
                print("collided with: " + infor[1]['prefab_name'] + "; from: " + infor[1]['category'])
                return True #collision is happening       
    return False    

def checkTwo3DBboxOverlap(bbox1, bbox2):    
    
    #get 8 vertex of bbox1
    vertexlist = []
    for i in [-1,1]:
        for j in [-1,1]:
            for k in [-1,1]:
                point = np.array([bbox1['center'][0]+i*bbox1['size'][0]/2, bbox1['center'][1]+j*bbox1['size'][1]/2, bbox1['center'][2]+k*bbox1['size'][2]/2])
                vertexlist.append(point) 

    #check wehtehr each point is within bbox2
    for i in range(8):
        status = isPointInsideBox(vertexlist[i], bbox2)        
        if status:            
            return True        
    return False

def checkCamCollision(cam_pos, graph):
    status = False
    
    for node in graph['nodes']:
        if node['category'] == 'Rooms' or  node['category'] == 'Walls':
            continue
        else:
            bbox = node['bounding_box']
            statusInside = isPointInsideBox(cam_pos,bbox)
            if statusInside:
                status = True
                print(node['prefab_name'])
                return status
    
    return status

def isPointInsideBox(point, bbox): 
    #get bbox2 boundaries
    minX = bbox['center'][0] - bbox['size'][0]/2 
    maxX = bbox['center'][0] + bbox['size'][0]/2 
    minY = bbox['center'][1] - bbox['size'][1]/2 
    maxY = bbox['center'][1] + bbox['size'][1]/2 
    minZ = bbox['center'][2] - bbox['size'][2]/2 
    maxZ = bbox['center'][2] + bbox['size'][2]/2
    
    return (point[0] >= minX and point[0] <= maxX) and (point[1] >= minY and point[1] <= maxY) and (point[2] >= minZ and point[2] <= maxZ)

def FindOptimalCamTargetConfig_original(targetSz, targetYpos, NumRes):
    if targetSz <0.5:
        Radius = np.sqrt(2)        
    elif targetSz <1:
        Radius = 1.5*np.sqrt(2)
    elif targetSz <2:
        Radius = 2.5*np.sqrt(2)
    else:
        Radius = 4*np.sqrt(2)
        
    if targetYpos > 1.4:
        camYStepSz = 0
        targetYStepSz = -0.25
    elif targetYpos>0.7:
        targetYStepSz = 0.25
        camYStepSz = 0.5
    else:
        targetYStepSz = 0.25
        camYStepSz = 1
    
    circ = CircleTrajectory(Radius, NumRes)
    return circ, camYStepSz, targetYStepSz

def FindOptimalCamTargetConfig_size(targetSz, sizeMult, targetYpos, NumRes):
            
    if targetSz <0.5:
        Radius = 2*np.sqrt(2)        
    elif targetSz <1:
        Radius = 1.5*2*np.sqrt(2)
    elif targetSz <2:
        Radius = 2.5*1.5*np.sqrt(2)
    else:
        Radius = 4*np.sqrt(2)
        
    if targetYpos > 1.4:
        camYStepSz = 0
        targetYStepSz = -0.25-0.2
    elif targetYpos>0.7:
        targetYStepSz = 0.25
        camYStepSz = 0.5+0.3
    else:
        targetYStepSz = 0.25
        camYStepSz = 1+0.5   
    
    circ = CircleTrajectory(Radius, NumRes)
    return circ, camYStepSz, targetYStepSz

#objects in their original place
def FindOptimalCamTargetConfig_gravity(targetSz, targetYpos, NumRes):
    if targetSz <0.5:
        Radius = np.sqrt(2)        
    elif targetSz <1:
        Radius = 1.5*np.sqrt(2)
    elif targetSz <2:
        Radius = 2.5*np.sqrt(2)
    else:
        Radius = 4*np.sqrt(2)
        
    if targetYpos > 1.4:
        camYStepSz = 0
        targetYStepSz = -0.25
    elif targetYpos>0.7:
        targetYStepSz = 0.25
        camYStepSz = 0.5
    else:
        targetYStepSz = 0.25
        camYStepSz = 1
    
    circ = CircleTrajectory(Radius, NumRes)
    return circ, camYStepSz, targetYStepSz

def FindOptimalCamTargetConfig_trained(targetSz, targetYpos, NumRes):
    if targetSz <0.5:
        Radius = 1*np.sqrt(2)        
    elif targetSz <1:
        Radius = 1.5*np.sqrt(2)
    elif targetSz <2:
        Radius = 2*np.sqrt(2)
    else:
        Radius = 2.5*np.sqrt(2)
        
    if targetYpos > 1.4:
        pitch = [np.pi/2 + np.pi/9, 7*np.pi/18]   #pitch angle in radians [-20, 20]     
    elif targetYpos>0.7:
        pitch = [7*np.pi/18, np.pi/6]  #pitch angle in radians  [20, 60]         
    else:
        pitch = [np.pi/3, np.pi/9] #pitch angle in radians   [30, 70]   
    
    circ = SphereTrajectory(Radius, pitch, NumRes)
    return circ

def SphereTrajectory(radius, pitch, Res):
    #takes in radius and how many uniformly sampled points on the circle
    #generate list of tuple (x,y) coordinates on the circle equally spaced
    circ = list()  
    for p in pitch:
        for j in range(Res):    
            circ.append( (  radius* np.sin(p) * np.cos(j* 2 * np.pi / Res), radius*np.cos(p), radius* np.sin(p) * np.sin(j* 2 * np.pi / Res) ))        
    return circ

def FindOptimalCamTargetConfig_trained2(targetSz, targetYpos, NumRes):
    
    Resolution = 2.0 # 1 deg angle resolution    
    radius = []
    pitch = []
    yaw = []
    for i in range(NumRes):
        
        RandSzTimes = random.randrange(2,7) #random int from [2,10] inclusive
        radius.append(1.0*RandSzTimes*targetSz)
        
        yaw.append( random.randrange(0, int(360/Resolution), Resolution)*Resolution/360 * math.pi*2)
        if targetYpos > 1.4:
            pitch.append( random.randrange(-int(35/Resolution), int(55/Resolution), Resolution)*Resolution/90 * math.pi/2)
        else:
            pitch.append( random.randrange(int(10/Resolution), int(90/Resolution), Resolution)*Resolution/90 * math.pi/2)
    
#    print(radius)
#    print(pitch)
#    print(yaw)
    circ = SphereTrajectory2(radius, pitch, yaw)
    return circ, radius, pitch, yaw

def FindOptimalCamTargetConfig_trained3(targetSz, targetYpos, NumRes):
    
    Resolution = 2.0 # 1 deg angle resolution    
    radius = []
    pitch = []
    yaw = []
    for i in range(NumRes):
        
        RandSzTimes = random.randrange(1,10) #random int from [2,10] inclusive
        radius.append(1.0*RandSzTimes*0.5)
        
        yaw.append( random.randrange(0, int(360/Resolution), Resolution)*Resolution/360 * math.pi*2)
        if targetYpos > 1.4:
            pitch.append( random.randrange(-int(35/Resolution), int(55/Resolution), Resolution)*Resolution/90 * math.pi/2)
        else:
            pitch.append( random.randrange(int(10/Resolution), int(90/Resolution), Resolution)*Resolution/90 * math.pi/2)
    
#    print(radius)
#    print(pitch)
#    print(yaw)
    circ = SphereTrajectory2(radius, pitch, yaw)
    return circ, radius, pitch, yaw

def SphereTrajectory2(radius, pitch, yaw):
    #takes in radius and how many uniformly sampled points on the circle
    #generate list of tuple (x,y) coordinates on the circle equally spaced
    circ = list()  
    for i, R in enumerate(radius):
        p = pitch[i]
        y = yaw[i]        
        circ.append( (  R* np.sin(p) * np.cos(y), R*np.cos(p), R* np.sin(p) * np.sin(y) ))        
    return circ


def CircleTrajectory(radius, Res):
    #takes in radius and how many uniformly sampled points on the circle
    #generate list of tuple (x,y) coordinates on the circle equally spaced
    circ = list()    
    for j in range(Res):    
        circ.append( (  radius* np.cos(j* 2 * np.pi / Res), radius* np.sin(j* 2 * np.pi / Res) ))        
    return circ    
        
def saveImgList(writedir, writedirjason, imageprefix, imgformat, sort_index, CamMImg, CamMID, TargetInfor, propFirstN, saveJasonflag):
    N = int(propFirstN * len(sort_index))
    for index in sort_index[:N]:
        count_camview = CamMID[index] 
        img_inst_target_cv2 = CamMImg[index]        
        print(writedir + imageprefix + str(count_camview) + imgformat)         
        cv2.imwrite(writedir + imageprefix + str(count_camview) + imgformat, img_inst_target_cv2)
        
        if saveJasonflag:
            storeinfor = TargetInfor[index]
            #storeinfor_json = json.dumps(storeinfor)
            f = open(writedirjason + imageprefix + str(count_camview) + ".pkl","wb")
            pickle.dump(storeinfor,f)
            f.close()
            
def saveImgList_train(writedir, writedirjason, imageprefix, imgformat, sort_index, CamMImg, CamMID, TargetInfor, propFirstN, saveJasonflag):
    N = int(propFirstN * len(sort_index))
    for index in sort_index[:N]:
        count_camview = CamMID[index] 
        img_inst_target_cv2 = CamMImg[index]        
        print(writedir + imageprefix + str(count_camview) + imgformat)
        img_inst_target_cv2 = cv2.resize(img_inst_target_cv2, (640, 512)) 
        cv2.imwrite(writedir + imageprefix + str(count_camview) + imgformat, img_inst_target_cv2)
        
        if saveJasonflag:
            storeinfor = TargetInfor[index]
            #storeinfor_json = json.dumps(storeinfor)
            f = open(writedirjason + imageprefix + str(count_camview) + ".pkl","wb")
            pickle.dump(storeinfor,f)
            f.close()

def findAllPossibleDestNodes(targetclass, wantedClass, ItemToRoom, SurfaceToRoom, RoomList, SurfaceList, graph):
    destnodesIDs = []
    destPrefabs = []
    destTargetRooms = []
    destSurfaceList=[]
    
    destRooms = []
    for i  in np.where(ItemToRoom[wantedClass.index(targetclass)] == 1)[0]:
        destRooms.append(RoomList[i])

    destSurface = []
    for dstR in destRooms:
        for i in np.where( SurfaceToRoom[:, RoomList.index(dstR)] == 1)[0]:
            destSurface.append(SurfaceList[i])
    
    destSurface = set(destSurface)
    destSurface = list(destSurface)
    
    for node in graph['nodes']:
        
        if node['class_name'] not in destSurface:
            continue
        
        roomIn = find_rooms(graph, node)
        if roomIn not in destRooms:
            #print("warning! " + roomIn + " doesnt belong to any rooms!")
            continue
        
        destnodesIDs.append(node['id'])
        destPrefabs.append(node['prefab_name'])
        destTargetRooms.append(roomIn)
        destSurfaceList.append(node['class_name'])
        
        
    return destnodesIDs, destPrefabs, destTargetRooms, destSurfaceList

def findAllPossibleDestNodes_anomaly(targetclass, wantedClass, ItemToRoom, RoomList, SurfaceList, graph):
    destnodesIDs = []
    destPrefabs = []
    destTargetRooms = []
    destSurfaceList=[]
    destRoomNode = []
    destWallNode = []
    
    destSurface = []
    for i  in np.where(ItemToRoom[wantedClass.index(targetclass)] == 1)[0]:
        surfacename = SurfaceList[i]
        if 'floor_' in surfacename:
            surfacename = surfacename[6:]
            destSurface.append(surfacename)
        else:
            destSurface.append(surfacename)

    destSurface = set(destSurface)
    destSurface = list(destSurface)
    
    #find all wall surfaces and their corresponding room
#    wallnodes=[]
#    wallroom = []
#    for node in graph['nodes']:
#        if node['class_name'] == 'wall':
#            sz = node['bounding_box']['size']
#            if all(x > 2 for x in sz):
#                continue;
#            else:
#                roomIn = find_rooms_graphedges(graph, node)
#                wallroom.append(roomIn)
#                wallnodes.append(node)
    
    for node in graph['nodes']:
        
        if node['class_name'] != 'wall':
            if node['class_name'] not in destSurface:
                continue
            
            if node['class_name'] in RoomList:
                roomIn = node['class_name']
            else:
                roomIn = find_rooms(graph, node)
                    
            destnodesIDs.append(node['id'])
            destPrefabs.append(node['prefab_name'])
            destTargetRooms.append(roomIn)
            destSurfaceList.append(node['class_name'])
            destRoomNode.append(float("nan"))
            destWallNode.append(float("nan"))
        else:
            sz = node['bounding_box']['size']
            if all(x > 2 for x in sz):
                continue;
            else:
                roomNode = find_rooms_graphedges(graph, node)
                roomIn = roomNode['class_name']
                destsurf = 'wall_' + roomIn
                if destsurf in destSurface:
                    destnodesIDs.append(node['id'])
                    destPrefabs.append(node['prefab_name'])
                    destTargetRooms.append(roomIn)
                    destSurfaceList.append(node['class_name'])
                    destRoomNode.append(roomNode)
                    destWallNode.append(node)
    return destnodesIDs, destPrefabs, destTargetRooms, destSurfaceList, destRoomNode, destWallNode
        
def add_node(graph, n):
    graph['nodes'].append(n)

def add_edge(graph, fr_id, rel, to_id):
    graph['edges'].append({'from_id': fr_id, 'relation_type': rel, 'to_id': to_id})

def deleteGraphByClassname(graph, target_classname):
    #print(graph)
    ToDeleteList = find_nodes_byclassname(graph, target_classname)
    #print(ToDeleteList)
    ToDeleteIDList = []
    for i, mc in enumerate(ToDeleteList):
        ToDeleteIDList.append(mc['id'])
            #del mc['obj_transform']
            #del mc['bounding_box']
    flagAll = True
    while flagAll:
        for i, node in enumerate(graph['nodes']):
            if node['class_name'] == target_classname:
                del graph['nodes'][i]
                flagAll = True
                break
            else:
                flagAll = False
    
    #print(ToDeleteIDList)
    #for idDelete in ToDeleteIDList:             
    graph['edges'] = [edge for edge in graph['edges'] if (edge['from_id'] not in ToDeleteIDList) and (edge['to_id'] not in ToDeleteIDList)]
            
    return graph   

def computeMoveNodeOffset_anomaly(destwallnode, destroomnode, targetnode):
    wallcenter = destwallnode['bounding_box']['center']
    roomcenter = destroomnode['bounding_box']['center']
    if destwallnode['bounding_box']['size'][0]<2:
        alongaxis = 0
    else:
        alongaxis = 2   
    if wallcenter[alongaxis] - roomcenter[alongaxis] > 0:
        axisorient = -1
    else:
        axisorient = 1
        
    desiredpos = wallcenter.copy()
    desiredpos[alongaxis] = wallcenter[alongaxis] + axisorient*targetnode['bounding_box']['size'][alongaxis]/2
    movenode_offset = desiredpos.copy()
    for dim in range(3):
        movenode_offset[dim] = desiredpos[dim] - targetnode['bounding_box']['center'][dim]
    return movenode_offset

def find_destsurfnode_byclassname(graph, targetnode, destsurf):
    targetid = targetnode['id']
    destsurflist = find_nodes_byclassname(graph, destsurf)
    destsurfidlist = [node['id'] for node in destsurflist]
    targetsurfidlist = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == targetid and edge['relation_type'] == 'ON']
    surfnode = []
    if len(destsurfidlist)>0 and len(targetsurfidlist)>0 :
        counter = 0
        for did in destsurfidlist:
            if did in targetsurfidlist:
                surfnode.append(destsurflist[counter])
                break
            counter = counter + 1
            
    return surfnode    
        
def computePossibleLocationsOnSurf(targetnode, surfnode, scaleStepSz):
       
    targetSzX = targetnode['bounding_box']['size'][0]
    targetSzZ = targetnode['bounding_box']['size'][2]
    
    leftBoundSurfX =  surfnode['bounding_box']['center'][0] - surfnode['bounding_box']['size'][0]/2 + targetSzX/2
    rightBoundSurfX =  surfnode['bounding_box']['center'][0] + surfnode['bounding_box']['size'][0]/2 - targetSzX/2
    
    leftBoundSurfZ =  surfnode['bounding_box']['center'][2] - surfnode['bounding_box']['size'][2]/2 + targetSzZ/2
    rightBoundSurfZ =  surfnode['bounding_box']['center'][2] + surfnode['bounding_box']['size'][2]/2 - targetSzZ/2
    
    x = np.arange(leftBoundSurfX,rightBoundSurfX,scaleStepSz*targetSzX)
    z = np.arange(leftBoundSurfZ,rightBoundSurfZ,scaleStepSz*targetSzZ)
#    x = np.arange(leftBoundSurfX,rightBoundSurfX,0.1)
#    z = np.arange(leftBoundSurfZ,rightBoundSurfZ,0.1)
    
    xpos, zpos = np.meshgrid(x,z)
    
    xpos = xpos.flatten() 
    zpos = zpos.flatten() 
    
    xoffset = xpos - targetnode['bounding_box']['center'][0]
    zoffset = zpos - targetnode['bounding_box']['center'][2] 
    
    return xoffset, zoffset
        
def segmentTargetBbox(img_height, img_width, JasonData, img, targetid):
    #convert to cv2 image and ready to draw
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    seg = np.zeros((img_width, img_height)).astype('uint8')
    status = False
    for infor in JasonData.items():
        if infor[1]['prefab_id'] == targetid: 
            pixel = infor[1]['color']
            X,Y = np.where(np.all(img==np.asarray(pixel),axis=2))
            
            left = infor[1]['bbox'][2]
            top = infor[1]['bbox'][0]
            right = infor[1]['bbox'][3]
            bottom = infor[1]['bbox'][1]
            targetbbox = [left, top, right, bottom]
            
            seg[X,Y] = 255                        
            status = True    
            targetarea = infor[1]['area']#(bottom - top)*(right - left)
            seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
            return status, targetarea, targetbbox, seg
    return status, 0,0, img 
