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

rootdir = '/media/mengmi/KLAB15/Mengmi/Proj_context3/VirtualHome/unity/'
graphdirname = rootdir + 'GraphHuman_anomaly'
if os.path.exists(graphdirname):
    print('folder already exists')
else:
    os.mkdir(graphdirname)
       
#pre-setup rules for context and a list of wanted classes    
wanted = pickle.load( open( "wanted_anomaly.pkl", "rb" ) ) 
ItemToRoom = np.array(wanted['ItemToRoom'])
RoomList = wanted['RoomList']
SurfaceList = wanted['SurfaceList']
wantedClass = wanted['wantedClass']

for target_classname in wantedClass:   

    for env_id in range(7): # env_id ranges from 0 to 6   
    
        # start simulation (only once for each apartment)
        comm = UnityCommunication()
        comm.reset(env_id)
        #comm.setBathroomTexture(env_id)
        
        #============================ START PYTHON SCRIPT ===========================
        response, origraph = comm.environment_graph()
        origraph = deleteGraphByClassname(origraph, target_classname) 
        success, message = comm.expand_scene(origraph)
        #print(origraph)
        #exit()
        #loop through all nodes and check whether we have wanted
        DestIds, DestPrefabs, destRooms, destSurface, destRoomNodes, destWallNodes = findAllPossibleDestNodes_anomaly(target_classname, wantedClass, ItemToRoom, RoomList, SurfaceList, origraph) 
         
        #loop through missed classes
        #randomly sample prefab from it and add to the graph        
        for i, destid in enumerate(DestIds):
            
            newIDs = 1000 #fixed; dont change; always starts from 1000 and onwards
            comm.reset(env_id)
            response, origraph = comm.environment_graph()    
            #print(origraph)
            
            #get ready to add new object belonging to current class
            destid = DestIds[i] 
            destprefab = DestPrefabs[i]
            destroom = destRooms[i]
            destsurf = destSurface[i]
            destroomnode = destRoomNodes[i]
            destwallnode = destWallNodes[i]
            print("processing class [" + target_classname + "]; apartment [" + str(env_id) + "]; room [" + destroom + "]; surf [" + destsurf + "]") 

            #delete all objects belonging to target class from the current graph
            origraph = deleteGraphByClassname(origraph, target_classname) 
#            mclist = find_nodes_byclassname(origraph, target_classname)
#            if len(mclist)>0:
#                error('we should not be here')            
            success, message = comm.expand_scene(origraph)
#            print(origraph)    
#            exit()
                        
            add_node(origraph, {'class_name': target_classname, 'id': newIDs, 'properties': [], 'states': []})
            if destroom == destsurf:
                add_edge(origraph, newIDs, 'INSIDE', destid)
            elif 'wall' in destsurf:
                add_edge(origraph, newIDs, 'INSIDE', destroomnode['id'])
            else:
                add_edge(origraph, newIDs, 'ON', destid)
            
            success, message = comm.expand_scene(origraph)
            #print(success)
            #print(message)
            #newIDs = newIDs + 1
            
            _, origraph = comm.environment_graph()
                
            mclist = find_nodes_byclassname(origraph, target_classname)
            #print(mclist)     
            #print(origraph['edges'])
            
            if len(mclist)>1:
                print('how come? we should not be here')
            elif len(mclist) == 1:
                for i, mc in enumerate(mclist):
                    #print()
                    #print(mc)
                    mc_room = find_rooms(origraph, mc)
                    #print(mc_room)
                    #mc_edge = [edge for edge in origraph['edges'] if edge['from_id'] == mc['id'] and edge['relation_type'] == 'ON'][0]
                    #print(mc_edge)
                    #mc_OnObj = find_nodes_byid(origraph, mc_edge['to_id'])[0]
                    #print(mc_OnObj['class_name'])
                    #print(origraph)
                    if mc_room == destroom: #and mc_OnObj['class_name'] == destsurf and mc_edge['relation_type'] == 'ON':
#                        print(mc_room)
#                        print(mc)
#                        print(origraph)
                        print(mc['class_name'] + " with id " + str(mc['id']) + " has been added to room: " + mc_room)
                        #print(origraph['nodes'])
                        #graphname = graphdirname + '/' + mc['class_name'] + '_apartment_' + str(env_id) + '_room_' + destroom + '_surf_' + destsurf + '.pkl'
                        graphname = graphdirname + '/' + mc['class_name'] + '_apartment_' + str(env_id) + '_room_' + destroom + '_Fur_' + destsurf + '.pkl'
                        dest_wall = float("nan")
                        dest_room = float("nan")
                        
                        if 'wall' in destsurf:
                            dest_wall = destroomnode
                            dest_room = destwallnode
                            
                        f = open(graphname,"wb")
                        TargetGraph = {'graph': origraph, 'targetnode':mc, 'apartment_id': env_id, 'dest_surf':destsurf, 'dest_room': destroom, 'destnode_wall':dest_wall, 'destnode_room':dest_room}
                        pickle.dump(TargetGraph,f)
                        f.close()
                            
                        #exit()
                        #break
            else:
                print("!!!!!!!!!!!!!!!!!!! Warning: " + target_classname + " failed!")
    
    
    
    