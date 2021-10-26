'''
    generate obj file from fitted 3D faces using 3DDFA
    Based on the code of Xiangyu Zhu:
    http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
'''

import os
import numpy as np
from get_meshNormal import *
import scipy.io as sio

class getObj_3DDFA():
    def __init__(self, triangle_path, UV_path, mtl_path):
        '''
            triangle_path contains the path to model_info.mat
            objPath contains path to 3DMM_normal.obj
        ''' 
        self.triangle_info = sio.loadmat(triangle_path)
        self.uv_path = UV_path
        self.mtl_path = mtl_path
        self.uv_list = None
        self.get_UV()

    def get_UV(self):
        '''
            load UV map
        '''
        uv_mat = sio.loadmat(self.uv_path)
        uv_mat = uv_mat['UV']
        # get UV map for each vertex in 3DDFA model
        self.uv_list = uv_mat[np.squeeze(self.triangle_info['trimIndex']-1)]
        return

    def load_obj_3DDFA(self, fileName):
        '''
            load obj file generated by 3DDFA
        '''
        vertex_list = []
        face_list = []
        vertexColor_list = []

        with open(fileName) as f:
            for line in f:
                tmp = line.strip().split()
                if len(tmp) > 0 and tmp[0] == 'v':
                    # dealing with vertex
                    vertex_list.append([float(item) for item in tmp[1:4]])
                    vertexColor_list.append([int(item) for item in tmp[4:]])
                if len(tmp) > 0 and tmp[0] == 'f':
                    # dealing with face
                    # NOTE: index of face is start from 0
                    face_list.append([int(item) for item in tmp[1:]])
        return np.array(vertex_list), np.array(vertexColor_list), np.array(face_list)
    
    def create_newObj(self, objFileName, saveFileName):
        '''
            create a obj file based on 3DDFA and 3DMM_normal
        '''
        vertex_list, vertexColor_list, face_list = self.load_obj_3DDFA(objFileName)
        # change x y coordinate so the normal computed is pointing outward
        tmp = vertex_list[:,1].copy()
        vertex_list[:,1] = vertex_list[:,0]
        vertex_list[:,0] = tmp

        if self.uv_list is None:
            self.get_UV()

        # get normal
        vertex_normal = get_normal(vertex_list, face_list)

        fid = open(saveFileName, 'w')
        print('mtllib {:}'.format(self.mtl_path), file=fid)

        # print normals and vertex
        for i in range(vertex_list.shape[0]):
            print('v %0.6f %0.6f %0.6f' % tuple(vertex_list[i]), file=fid)
            print('vn %0.6f %0.6f %0.6f' % tuple(vertex_normal[i]), file=fid)
        print('usemtl material_0', file=fid)

        # print vertex
        for item in self.uv_list:
            print('vt %0.6f %0.6f' % tuple(item), file=fid)

        # print face
        for face in face_list:
            print('f %d/%d/%d %d/%d/%d %d/%d/%d' % (face[0], face[0], face[0], face[1], face[1], face[1], face[2], face[2], face[2]), file=fid)
        fid.close()
