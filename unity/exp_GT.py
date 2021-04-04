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

#parameter initialization
img_height = 1280
img_width = 1024
ThresTargetObjectSize = 0.
ThresRoomArea = 0.2 #20% area of the picture is in the other rooms other than the target prefab
ThresBlackSkyArea = 0.2 #20% area of the picture is black skybox
ThresNumInst = 10 #at least 15 instances on the picture
ThresContrast = 0.5 #(Ymax - Ymin) / (ymax + ymin)
NumResCam = 9 #number of cameras generated on a circle centered at target locations
RatioCroppedContrast = 0.1 #cropped surrounding regions based on ratio of the target bbbox size
propFirstN = 0.5 #only save first 50% of images to avoid occlusions

#the list of categories we want to study prefab; 
wanted = pickle.load( open( "wanted.pkl", "rb" ) ) 
wantedClass = wanted['wantedClass']

#categories like walls, floors are not interesting to study context at all; ignore those categories
#PrefabCategoryList = ['Furniture']
PrefabCategoryList = ['Furniture','Decor','Electronics','Props','Appliances','Foods'] 
#BadClassName = ['wallpictureframe', 'lightswitch', 'wallphone', 'powersocket', 'toiletpaper', 'knifeblock']

rootdir = '/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/unity/'
stimulusdirname = rootdir + 'stimulus_2'
jasondirname = rootdir + 'jason_2'
graphdirname = rootdir + 'GraphHuman_2'

if os.path.exists(stimulusdirname):
    print('folder already exists')
else:
    os.mkdir(stimulusdirname)
        
if os.path.exists(jasondirname):
    print('folder already exists')
else:
    os.mkdir(jasondirname)

for fileid, filename in enumerate(os.listdir(graphdirname)):
    print('processing ... graphid: ' + str(fileid) + '; ' + filename)
    PICKLE = pickle.load( open( graphdirname + '/' + filename , "rb" ) ) 
    #PICKLE = pickle.load( open( graphdirname + '/pie_apartment_6_room_kitchen.pkl' , "rb" ) )
    
    wantedgraph = PICKLE['graph']
    node = PICKLE['targetnode'] 
    env_id = PICKLE['apartment_id']
    destsurf = PICKLE['dest_surf']
    
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
    
    # start simulation (only once for each apartment)
    comm = UnityCommunication()
    comm.reset(env_id)
    #comm.setBathroomTexture(env_id)
    #wantedgraph = pickle.load( open( graphdirname + '/apartment_' + str(env_id) + ".pkl", "rb" ) ) 
    #print(wantedgraph)
    _, message = comm.expand_scene(wantedgraph)
    
    
    #============================ START PYTHON SCRIPT ===========================
    res, origraph = comm.environment_graph()
    totalnodes = len(origraph['nodes'])
    #print(origraph)
    
    #for node in origraph['nodes']:
        
    target_id = node['id'] #target object to recognize 
    #for debugging; comment out
#    if target_id !=350:
#        continue
    targetclassname = node['class_name']
    targetprefabname = node['prefab_name']    
    
    #remove those objects that we dont want
#        if targetclassname not in wantedClass:
#            continue
    
    #remove those objects not belong to these general cateogires
    #if node['category'] not in PrefabCategoryList:
    #    continue
    
    #remove those bad object classes
    #if node['class_name'] in BadClassName:
    #    continue
    
    #remove those large target objects; we are not interested    
#        if node['bounding_box']['size'][0] >1 or node['bounding_box']['size'][1] >1 or node['bounding_box']['size'][2] >1:
#            print('Object Size too large')
#            continue 
    
    print('Processing prefab id: ' + str(target_id)) 
    targetroomname = find_rooms(origraph, node)
    targetroombbox = find_nodes_byclassname(origraph, targetroomname)[0]['bounding_box'] 
    avgtargetsize =  np.mean(node['bounding_box']['size']) 
    targetYpos = node['bounding_box']['center'][1]
    #print('avgtargetsize: ', avgtargetsize)
    #print('y aixs pos: ', targetYpos)
    circ, camYStepSz, targetYStepSz = FindOptimalCamTargetConfig_original(avgtargetsize, targetYpos, NumResCam)
    targetYStepSz = 0 #force to be in original position
    count_camview = 0
    
    CamMImg = list() #modified target image
    CamMTArea = list() #modified target area
    CamMID = list()
    TargetInfor = list()        
        
    try:
        for circ_coor in circ: #camera revolving along a circle cnetered at target
                
            print('Processing prefab id: ' + str(target_id) + "; " + targetprefabname + '; cam id =' + str(count_camview)) 
            
            storeinfor = {}
            
            xmov = circ_coor[0]
            zmov = circ_coor[1]
            count_camview += 1
            
            res, graph = comm.environment_graph()
            #print(graph['nodes'])
            objnode = find_nodes_byid(graph, target_id)[0]
    #            print('ojbnode1:')
    #            print(objnode)
            cam_pos = objnode['obj_transform']['position']           
            cam_pos[0] = cam_pos[0] + xmov
            cam_pos[1] = cam_pos[1] + camYStepSz #move up 1 meter
            cam_pos[2] = cam_pos[2] - zmov
            storeinfor['cam_pos'] = cam_pos
            
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
            pos = [0, targetYStepSz, 0] #move object 1 meter up
            sca = [0, 0, 0] #let the scale remain to be [1,1,1]
            rot = [0, 0, 0] #(roll, yaw, pitch) in degrees; 180 deg; etc 
            success, message = comm.lookat_MMobjectTransform(target_id, position=pos, rotation=rot, scaling=sca)            
            storeinfor['target_tf_pos'] = pos
            storeinfor['target_tf_sca'] = sca
            storeinfor['target_tf_rot'] = rot
            
            #we update graph after moving objects
            #res, graph = comm.environment_graph()  
            objnode = find_nodes_byid(graph, target_id)[0]
    #            print('ojbnode1:')
    #            print(objnode)
    #            print("==========================================")
            #res, graph = comm.environment_graph()
    #            print(graph)
    #            print("---------------------------------------------")
            success, message_color = comm.getObjInstanceColorAndPrefab()
    #            print(message_color)            
            
    #            print("+++++++++++++++++++++++++++++++++++++++++++")
            ColorInstLookUpTab = extractColorInstanceTable(graph, message_color)              
    #            print(ColorInstLookUpTab)
                        
            # View the newly created camera from different modes
            img_all_pil = display_scene_modalities(img_height, img_width, comm, camera_indices, modalities=['normal', 'seg_class', 'seg_inst'], nrows=3) #hard-coded; do NOT change modalities
            img_ori_pil, img_class_pil, img_inst_pil, img_ori_np, img_class_np, img_inst_np = convertPILImageToNumpyImage(img_all_pil, img_height, img_width)
            JasonData = extractJasonInstanceTable(img_inst_pil, img_inst_np, ColorInstLookUpTab)
            storeinfor['JasonData'] = JasonData
            #print("==========================================")
            #print(JasonData)
            
        
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
                #comm.reset_MMobjectTransform()
                #reset everything; and start another prefab modification
                comm.reset(env_id)  
                _, message = comm.expand_scene(wantedgraph)
                #comm.setBathroomTexture(env_id)
                continue
            
            #display all bounding boxes and their labels
            #print('saving images...')
            #status, img_inst_all_cv2 = displayAllBbox(img_height, img_width, JasonData, img_ori_np)
    #            print(JasonData)
            #target_id = 354
            status, targetarea, targetbbox, img_inst_target_cv2 = displayTargetBbox(img_height, img_width, JasonData, img_ori_np, target_id, textflag=False, boxflag=False)
    #            print(target_id)
    #            cv2.imwrite("img1.png", img_inst_target_cv2)
    #            exit()
            
            #CamMImg.append(img_inst_np)
            CamMImg.append(img_inst_target_cv2)
            CamMTArea.append(-targetarea) #use neg val in order to sort in descending order
            CamMID.append(count_camview)
            storeinfor['target_node'] = node
            storeinfor['JasonData'] = JasonData
            storeinfor['targetroomname'] = targetroomname
            storeinfor['graphname']=filename
            storeinfor['dest_surf'] = destsurf
            storeinfor['target_bbox'] = targetbbox
            TargetInfor.append(storeinfor)
            #cv2.imwrite(writedir + "img_" + str(target_id) + "_prefab_" + targetprefabname + "_all_modified_cam_" + str(count_camview) + ".png", img_inst_all_cv2)
            #cv2.imwrite(writedir + "img_" + str(target_id) + "_prefab_" + targetprefabname + "_target_modified_cam_" + str(count_camview) + ".png", img_inst_target_cv2)
       
            #put object back to original pose 
            #comm.reset_MMobjectTransform()
            #comm.setBathroomTexture(env_id)
    #            # for initial position
    #            # View the SAME newly created camera from different modes
    #            img_back_pil = display_scene_modalities(img_height, img_width, comm, camera_indices, modalities=['normal', 'seg_class', 'seg_inst'], nrows=3)
    #            img_ori_pil, img_class_pil, img_inst_pil, img_ori_np, img_class_np, img_inst_np = convertPILImageToNumpyImage(img_back_pil, img_height, img_width)
    #            JasonData = extractJasonInstanceTable(img_inst_pil, img_inst_np, ColorInstLookUpTab)    
    #            #print('saving images...')
    #            #status, img_inst_all_cv2 = displayAllBbox(img_height, img_width, JasonData, img_ori_np)
    #            status, img_inst_target_cv2 = displayTargetBbox(img_height, img_width, JasonData, img_ori_np, target_id)
    #            #cv2.imwrite(writedir + "img_" + targetprefabname + "_all_initial_cam_" + str(count_camview) + ".png", img_inst_all_cv2)
    #            cv2.imwrite(writedir + "img_" + targetprefabname + "_target_initial_cam_" + str(count_camview) + ".png", img_inst_target_cv2)
    #            print('Snapshot saved in demo/'+ "img_" + str(target_id) + "_prefab_" + targetprefabname + "_initial.png")
    
            #reset everything; and start another prefab modification
            comm.delete_new_camera()
            comm.reset(env_id)
            
            _, message = comm.expand_scene(wantedgraph)
            #comm.setBathroomTexture(env_id)
            #check graph is back to its original form
            #res, graph = comm.environment_graph()
            #print(graph)
            print('satisfied all conditions...')
            
        #exhausively tried all camera options for the same target
        #sort the canonical views of the targets based on target bbox area        
        sort_index = np.argsort(CamMTArea)#in descending order; save the top 50% images    
        if int(len(sort_index)*propFirstN) > 1:
            print('saving images...')
            imageprefix = "img_" + str(target_id) + "_prefab_" + targetprefabname + "_all_modified_cam_"
            imgformat = ".png"
            #a = pickle.load( open("jason/apartment_0/img_54_prefab_PRE_PRO_Towel_02_all_modified_cam_2.pkl","rb"))
            saveImgList(writedir, writedirjason, imageprefix, imgformat, sort_index, CamMImg, CamMID, TargetInfor, propFirstN, saveJasonflag = True)
        else:
            print('too few optimal camera views for the current target')
     
    except:
        continue
    
    CamMImg.clear()
    CamMTArea.clear()
    CamMID.clear()
    TargetInfor.clear()
