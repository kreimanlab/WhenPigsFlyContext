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
propFirstN = 0.8 #only save first 50% of images to avoid occlusions
SizeChange = [4,3,2] #how to change the size of the items for every threshold

#the list of categories we want to study prefab; 
wanted = pickle.load( open( "wanted_anomaly.pkl", "rb" ) ) 
wantedClass = wanted['wantedClass']

#categories like walls, floors are not interesting to study context at all; ignore those categories
#PrefabCategoryList = ['Furniture']
PrefabCategoryList = ['Furniture','Decor','Electronics','Props','Appliances','Foods'] 
#BadClassName = ['wallpictureframe', 'lightswitch', 'wallphone', 'powersocket', 'toiletpaper', 'knifeblock']

rootdir = '/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/unity/'
stimulusdirname = rootdir + 'stimulus_size'
jasondirname = rootdir + 'jason_size'
graphdirname = rootdir + 'GraphHuman_2'

if os.path.exists(stimulusdirname):
    print('folder already exists')
else:
    os.mkdir(stimulusdirname)
        
if os.path.exists(jasondirname):
    print('folder already exists')
else:
    os.mkdir(jasondirname)

resumeflag = 0
for sz, sizeMult in enumerate(SizeChange):

    for g, filename in enumerate(os.listdir(graphdirname)):
        print('processing ... : ' + str(g) +'; sz = ' + str(sz) + '; graphname: ' + filename)
        PICKLE = pickle.load( open( graphdirname + '/' + filename , "rb" ) ) 
        #PICKLE = pickle.load( open( graphdirname + '/pie_apartment_6_room_kitchen.pkl' , "rb" ) )
        
        if g == 196 and sz == 0:
            resumeflag = 1
            
        if resumeflag == 0:
            continue
        
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
        target_id = node['id'] #target object to recognize    
        targetclassname = node['class_name']
        targetprefabname = node['prefab_name']    
        
        if targetclassname not in wantedClass:
            continue
        
        print('Processing prefab id: ' + str(target_id)) 
        targetroomname = find_rooms(origraph, node)
        targetroombbox = find_nodes_byclassname(origraph, targetroomname)[0]['bounding_box'] 
        avgtargetsize =  np.mean(node['bounding_box']['size']) 
        targetsize_y = node['bounding_box']['size'][1]
        targetYpos_offset = targetsize_y*sizeMult/2 - targetsize_y/2
        #print(targetYpos_offset)
        targetYpos = node['bounding_box']['center'][1] + targetYpos_offset
        #print('avgtargetsize: ', avgtargetsize)
        #print('y aixs pos: ', targetYpos)
               
        circ, camYStepSz, targetYStepSz = FindOptimalCamTargetConfig_size(avgtargetsize*sizeMult, sizeMult, targetYpos, NumResCam)
        count_camview = 0
        
        CamMImg = list() #modified target image
        CamMTArea = list() #modified target area
        CamMID = list()
        TargetInfor = list()        
            
        try:
            for circ_coor in circ: #camera revolving along a circle cnetered at target
                    
                print('Processing prefab id: ' + str(target_id) + "; " + targetprefabname + '; cam id =' + str(count_camview) + '; size multiplier = ' + str(sizeMult)) 
                
                storeinfor = {}
                
                xmov = circ_coor[0]
                zmov = circ_coor[1]
                count_camview += 1
                
                res, graph = comm.environment_graph()
                #very IMPORTANT! target is the ONLY classname
                #must update target id after expanding graphs; node sequence changed!
                objnode = find_nodes_byclassname(graph, targetclassname)[0]
                target_id = objnode['id']
                
                #print(graph['nodes'])
                objnode = find_nodes_byid(graph, target_id)[0]
                
                cam_pos = objnode['obj_transform']['position']            
                cam_pos[0] = cam_pos[0] + xmov
                cam_pos[1] = cam_pos[1] + targetYpos_offset + camYStepSz #move up 1 meter
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
                pos = [0, targetYpos_offset, 0] #don't move the object
                sca = list((sizeMult-1)*np.array([1.0,1.0,1.0]))
                rot = [0,0,0]
    			  #success, message = comm.lookat_MMobjectTransform(target_id, position=pos, rotation=rot, scaling=sca)    
                success, message = comm.modify_MMobjectTransform(target_id, position=pos, rotation=rot, scaling=sca)
                storeinfor['target_tf_pos'] = pos
                storeinfor['target_tf_sca'] = sca
                storeinfor['target_tf_rot'] = rot 
                storeinfor['sizeMult'] = sizeMult
                
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
                
                
                status, targetarea, targetbbox, img_inst_target_cv2 = displayTargetBbox(img_height, img_width, JasonData, img_ori_np, target_id, textflag=False, boxflag=True)
        
                CamMImg.append(img_inst_target_cv2)
                CamMTArea.append(-targetarea) #use neg val in order to sort in descending order
                CamMID.append(count_camview)
                storeinfor['target_node'] = objnode
                storeinfor['JasonData'] = JasonData
                storeinfor['targetroomname'] = targetroomname
                storeinfor['graphname']=filename
                storeinfor['dest_surf'] = destsurf
                storeinfor['targetbbox'] = targetbbox
                TargetInfor.append(storeinfor)            
        
                #reset everything; and start another prefab modification
                comm.reset_MMobjectTransform()
                comm.delete_new_camera()        
                comm.reset(env_id)
                
                _, message = comm.expand_scene(wantedgraph)            
                print('satisfied all conditions...')
            
            #exhausively tried all camera options for the same target
            #sort the canonical views of the targets based on target bbox area        
            sort_index = np.argsort(CamMTArea)#in descending order; save the top 50% images    
            if int(len(sort_index)*propFirstN) > 1:
                print('saving images...')
                imageprefix = "img_" + str(target_id) + "_prefab_" + targetprefabname + "_size_modified_" + str(sizeMult) + "_cam_"
                imgformat = ".png"
                #a = pickle.load( open("jason/apartment_0/img_54_prefab_PRE_PRO_Towel_02_all_modified_cam_2.pkl","rb"))
                saveImgList(writedir, writedirjason, imageprefix, imgformat, sort_index, CamMImg, CamMID, TargetInfor, propFirstN, saveJasonflag = True)            
            else:
                print('too few optimal camera views for the current target')
         
        except:
            print('we are here')
            continue
        
        CamMImg.clear()
        CamMTArea.clear()
        CamMID.clear()
        TargetInfor.clear()
