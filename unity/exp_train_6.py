#matplotlib notebook
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
from MMutils import *
import pickle
import random

#parameter initialization
img_height = 1280
img_width = 1024
ThresTargetObjectSize = 0.
ThresRoomArea = 0.2 #20% area of the picture is in the other rooms other than the target prefab
ThresBlackSkyArea = 0.2 #20% area of the picture is black skybox
ThresNumInst = 10 #at least 15 instances on the picture
ThresContrast = 0.5 #(Ymax - Ymin) / (ymax + ymin)
NumResCam = 25 #number of cameras generated on a circle centered at target locations
RatioCroppedContrast = 0.1 #cropped surrounding regions based on ratio of the target bbbox size
propFirstN = 1 #only save first 50% of images to avoid occlusions
scaleStepSz = 5 #move every 3 times target object size and place the target
NumTotalMaterials = 1000
ProbMaterialChange = 0.8

#the list of categories we want to study prefab; 
#wanted = pickle.load( open( "wanted_anomaly.pkl", "rb" ) ) 
#wantedClass = wanted['wantedClass']
#print(wantedClass)
wantedClass = ['pillow', 'keyboard' , 'microwave', 'dishbowl', 'toothbrush' , 'poundcake','cupcake','pie']
#wantedClass = ['pillow', 'keyboard' , 'microwave', 'dishbowl', 'toothbrush' , 'poundcake','cupcake']
#categories like walls, floors are not interesting to study context at all; ignore those categories
#PrefabCategoryList = ['Furniture']
PrefabCategoryList = ['Furniture','Decor','Electronics','Props','Appliances','Foods'] 
#BadClassName = ['wallpictureframe', 'lightswitch', 'wallphone', 'powersocket', 'toiletpaper', 'knifeblock']

rootdir = '/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/unity/'
stimulusdirname = rootdir + 'stimulus_train_6'
jasondirname = rootdir + 'jason_train_6'
graphdirname = rootdir + 'GraphHuman_2'

if os.path.exists(stimulusdirname):
    print('folder already exists')
else:
    os.mkdir(stimulusdirname)
        
if os.path.exists(jasondirname):
    print('folder already exists')
else:
    os.mkdir(jasondirname)

#ResumeFlag = False
for f, filename in enumerate(os.listdir(graphdirname)):
    
#    if f == 1383:
#        ResumeFlag = True
#        
#    if not ResumeFlag:
#        continue
        
    
    print('processing ... : ' + filename)
    PICKLE = pickle.load( open( graphdirname + '/' + filename , "rb" ) ) 
    #PICKLE = pickle.load( open( graphdirname + '/pie_apartment_6_room_kitchen.pkl' , "rb" ) )
    
    wantedgraph = PICKLE['graph']
    node = PICKLE['targetnode']
    
    if node['class_name'] not in wantedClass:
            continue
    
    env_id = PICKLE['apartment_id']
    destsurf = PICKLE['dest_surf']    
    destsurfnode = find_destsurfnode_byclassname(wantedgraph, node, destsurf)
    
    if len(destsurfnode) < 1:
        continue
    
    destsurfnode = destsurfnode[0]   
    Target_xoffset, Target_zoffset = computePossibleLocationsOnSurf(node, destsurfnode, scaleStepSz)
    
##############################################################
    writedir = stimulusdirname + '/apartment_' + str(env_id) + '/'
    
    if os.path.exists(writedir):
        print('folder already exists')
    else:
        os.mkdir(writedir)
        
    writedirjason = jasondirname + '/apartment_' + str(env_id) + '/'
    if os.path.exists(writedirjason):
        print('folder already exists')
    else:
        os.mkdir(writedirjason)
    
    
    for i, target_xoffset in enumerate(Target_xoffset):
    
    
        # start simulation (only once for each apartment)
        comm = UnityCommunication()
        comm.reset(env_id)
        #comm.setBathroomTexture(env_id)
        #wantedgraph = pickle.load( open( graphdirname + '/apartment_' + str(env_id) + ".pkl", "rb" ) ) 
        #print(wantedgraph)
        _, message = comm.expand_scene(wantedgraph)       
        
        #============================ START PYTHON SCRIPT ===========================
        res, origraph = comm.environment_graph()
        node = PICKLE['targetnode']
        target_id = node['id'] #target object to recognize    
        targetclassname = node['class_name']
        targetprefabname = node['prefab_name']    
        
        print('Processing prefab id: ' + str(target_id)) 
        targetroomname = find_rooms(origraph, node)
        targetroombbox = find_nodes_byclassname(origraph, targetroomname)[0]['bounding_box'] 
        avgtargetsize =  np.mean(node['bounding_box']['size']) 
        targetYpos = node['bounding_box']['center'][1]
        #print('avgtargetsize: ', avgtargetsize)
        #print('y aixs pos: ', targetYpos)
        circ, radius, pitch, yaw = FindOptimalCamTargetConfig_trained3(avgtargetsize, targetYpos, NumResCam)
        #print(circ)
        #print(radius)
        #targetYStepSz = 0.5 #force to be in original position
                       
        count_camview = 0
    
        CamMImg = list() #modified target image
        CamMTArea = list() #modified target area
        CamMID = list()
        TargetInfor = list()
        
        for c, circ_coor in enumerate(circ): #camera revolving along a circle cnetered at target
            
            try:
                print("Processing G_" + str(f) + "_Oset_" + str(i) + "_Cset_" + str(c) + "; prefab id: " + str(target_id) + "; " + targetprefabname + '; cam=' + str(count_camview)) 
                
                storeinfor = {}
                
                storeinfor['cam_Radius'] = radius[c]
                storeinfor['cam_Pitch'] = pitch[c]
                storeinfor['cam_Yaw'] = yaw[c]
                
                xmov = circ_coor[0]
                camYStepSz = circ_coor[1]
                zmov = circ_coor[2]
                count_camview += 1
                
                res, graph = comm.environment_graph()
                #very IMPORTANT! target is the ONLY classname
                #must update target id after expanding graphs; node sequence changed!
                objnode = find_nodes_byclassname(graph, targetclassname)[0]
                target_id = objnode['id']
                
                #print(graph['nodes'])
                objnode = find_nodes_byid(graph, target_id)[0]
                
                cam_pos = [objnode['obj_transform']['position'][0] +  target_xoffset, objnode['obj_transform']['position'][1], objnode['obj_transform']['position'][2] +  Target_zoffset[i]]  
                cam_pos[0] = cam_pos[0] + xmov
                cam_pos[1] = cam_pos[1] + camYStepSz #move up 1 meter
                cam_pos[2] = cam_pos[2] - zmov
                storeinfor['cam_pos'] = cam_pos
                storeinfor['cam_offset'] = [xmov, camYStepSz, zmov]
                
                #check camera is inside the room; TRUE: inside the room
                status_cam_inRoom = isPointInsideBox(cam_pos,targetroombbox)
                if not status_cam_inRoom:
                    print('warning: camera is not in room')
                    continue
                
                #check camera is not colliding with other objects; TRUE: collided
                status_cam_collision = checkCamCollision(cam_pos, graph)
                if status_cam_collision:
                    print('warning: camera collides with objects in room')
                    continue
                
                #always set camera rotation to [0, 0, 0]
                #C# in unity code will rotate camera automatically to focus on the target object
                comm.add_camera(position=cam_pos, rotation=[0, 0, 0])
                s, cam_count = comm.camera_count()
                camera_indices = [cam_count-1]
                
                # specify the position and rotation of the object you want to modify
                # C# code in unity below for reference
                #tf_obj.gameObject.transform.position = PrefabPosList[indexprefab] + MMobj_config.position;
                #tf_obj.gameObject.transform.localScale = PrefabScaList[indexprefab] + MMobj_config.scaling;
                #PrefabRotList[indexprefab] = new Vector3(-roll_x, -yaw_y, -pitch_z);
                pos = [target_xoffset, 0, Target_zoffset[i]] #move object 1 meter up
                sca = [0, 0, 0] #let the scale remain to be [1,1,1]
                rot = [0, 0, 0] #(roll, yaw, pitch) in degrees; 180 deg; etc 
                #success, message = comm.lookat_MMobjectTransform(target_id, position=pos, rotation=rot, scaling=sca)
                
                material_id = float('nan')
                if random.uniform(0, 1) <= ProbMaterialChange:
                    material_id = random.randint(0,NumTotalMaterials-1)    
                    success, message = comm.changeMaterial_MMobjectTransform(target_id, material_id, position=pos, rotation=rot, scaling=sca)
                success, message = comm.modify_MMobjectTransform(target_id, position=pos, rotation=rot, scaling=sca)
                storeinfor['target_tf_pos'] = pos
                storeinfor['target_tf_sca'] = sca
                storeinfor['target_tf_rot'] = rot   
                storeinfor['material_id'] = material_id
                
                #we update graph after moving objects
                res, graph = comm.environment_graph()  
                objnode = find_nodes_byclassname(graph, targetclassname)[0]
                target_id = objnode['id']
                
                success, message_color = comm.getObjInstanceColorAndPrefab()            
                ColorInstLookUpTab = extractColorInstanceTable(graph, message_color)              
        
                            
                # View the newly created camera from different modes
                img_all_pil = display_scene_modalities(img_height, img_width, comm, camera_indices, modalities=['normal', 'seg_class', 'seg_inst'], nrows=3) #hard-coded; do NOT change modalities
                img_ori_pil, img_class_pil, img_inst_pil, img_ori_np, img_class_np, img_inst_np = convertPILImageToNumpyImage(img_all_pil, img_height, img_width)
                JasonData = extractJasonInstanceTable(img_inst_pil, img_inst_np, ColorInstLookUpTab)
                storeinfor['JasonData'] = JasonData
                        
                status_numInst = True
                if len(JasonData) < ThresNumInst:
                    status_numInst = False
                
                status_camerafit = checkCameraImageFitness(JasonData, target_id, ThresRoomArea) #true: target in pic
                status_blacksky = checkCameraImageBlackSky(img_ori_np, ThresBlackSkyArea) #true: little sky
                #status_collision = IsTargetCollision(JasonData, graph, target_id) #true: collided
                status_collision = False #objects in original position; cannot collide
                status_contrast = IsHighContrast(img_height, img_width, ThresContrast, RatioCroppedContrast, JasonData, img_ori_np, target_id) #true: high contrast
                print('Validaity of num insts:', status_numInst)
                print('validity of camera view:', status_camerafit)
                print('validity of black skybox:', status_blacksky)
                print('validity of collision:', (not status_collision))
                print('validity of contrast:', status_contrast)
                status = status_contrast and status_numInst and status_camerafit and status_blacksky and (not status_collision)
                
                if not status:
                    #this camera view does not satisfy two criterias; abandon image generation
                    #put object back to original pose 
                    comm.reset_MMobjectTransform()
                    #reset everything; and start another prefab modification
                    comm.reset(env_id)  
                    _, message = comm.expand_scene(wantedgraph)
                    #comm.setBathroomTexture(env_id)
                    continue
                
                
                status, targetarea, targetbbox, img_inst_target_cv2 = displayTargetBbox(img_height, img_width, JasonData, img_ori_np, target_id, textflag=False, boxflag=False)
        
                CamMImg.append(img_inst_target_cv2)
                CamMTArea.append(-targetarea) #use neg val in order to sort in descending order
                CamMID.append(count_camview)
                storeinfor['target_node'] = objnode
                storeinfor['JasonData'] = JasonData
                storeinfor['targetroomname'] = targetroomname
                storeinfor['graphname']=filename
                storeinfor['dest_surf'] = destsurf
                storeinfor['target_bbox'] = targetbbox
                TargetInfor.append(storeinfor)            
        
                #reset everything; and start another prefab modification
                comm.reset_MMobjectTransform()
                comm.delete_new_camera()        
                comm.reset(env_id)
                
                _, message = comm.expand_scene(wantedgraph)            
                print('satisfied all conditions...')         
         
            except:
                continue
            
        #exhausively tried all camera options for the same target
        #sort the canonical views of the targets based on target bbox area
        if len(CamMTArea) > 0:
            sort_index = np.argsort(CamMTArea)#in descending order; save the top 50% images    
            if int(len(sort_index)*propFirstN) > 1:
                print('saving images...')
                imageprefix = "img_G_" + str(f) + "_Oset_" + str(i) + "_Cset_" + str(c) + "_id_" + str(target_id) + "_fab_" + targetprefabname + "_cam_"
                imgformat = ".png"
                #a = pickle.load( open("jason/apartment_0/img_54_prefab_PRE_PRO_Towel_02_all_modified_cam_2.pkl","rb"))
                saveImgList_train(writedir, writedirjason, imageprefix, imgformat, sort_index, CamMImg, CamMID, TargetInfor, propFirstN, saveJasonflag = True)            
            else:
                print('too few optimal camera views for the current target')
        
        CamMImg.clear()
        CamMTArea.clear()
        CamMID.clear()
        TargetInfor.clear()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    