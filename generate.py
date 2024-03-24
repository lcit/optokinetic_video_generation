import os
import sys
import numpy as np
import cv2
import networkx as nx
from scipy import interpolate

def generate_trajectory(g, h, w, source, dest):
    img = np.zeros((h,w), np.float32)
    for i in range(200):
        cx = int(np.random.randint(w))
        cy = int(np.random.randint(h))
        s = np.random.randint(1,10)
        img = cv2.circle(img, (cx,cy), s, s, -1)

    g_mod = g.copy()
    for u,v,data in g_mod.edges(data=True):
        d = (u[0]-v[0])**2 + (u[1]-v[1])**2
        s = max(img[u[1], u[0]], img[v[1], v[0]])
        if img[u[1], u[0]] > 0 or img[v[1], v[0]] > 0:
            d *= s
        data["weight"] = d

    path = np.array(nx.shortest_path(g_mod, source, weight='weight')[dest])
    #img[path[:,1], path[:,0]] = 50
    return path, img

def interpolate_trajectory(path, oversampling=1, s=None, k=2):

    _indexes, _x = list(zip(*[(i,x) for i,x in zip(np.linspace(0, 1, len(path)), path)]))

    _x = np.array(_x)
    shape = _x.shape
    _x = _x.reshape(_x.shape[0], -1)   

    tck, _ = interpolate.splprep(_x.T, s=s, k=k, u=_indexes)

    _x_new = np.array(interpolate.splev(np.linspace(0, 1, len(path)*oversampling), tck)).T
    
    return _x_new

def generate_image(pos, h, w, background_image=None):
    img = np.zeros((h, w, 3), np.uint8)
    if background_image is not None:
        background_image = np.uint8(background_image)
        if np.ndim(background_image)==2:
            background_image = cv2.cvtColor(background_image, cv2.COLOR_GRAY2RGB)
        background_image = cv2.resize(background_image, (w, h))
        img = background_image
    
    cv2.putText(img, "x", (int(pos[0]), int(pos[1])), cv2.FONT_HERSHEY_PLAIN,
                3, (255,255,255), 2)
    return img

def main(n_trajectories=30,
         filename_output="output.avi",
         use_bubble_background=False):
    
    scale = 16
    h,w = 1024//scale, 1920//scale
    g = nx.grid_2d_graph(w,h)
    g.add_edges_from([
        ((x, y), (x+1, y+1))
        for x in range(w-1)
        for y in range(h-1)
    ] + [
        ((x+1, y), (x, y+1))
        for x in range(w-1)
        for y in range(h-1)
    ])
    
    print(f"Generate trajectory")
    border_nodes = [[(i,10) for i in range(w-10)],
                    [(i,h-11) for i in range(w-10)],
                    [(10,i) for i in range(h-10)],
                    [(w-11,i) for i in range(h-10)]]    
    
    j_end = np.random.choice(4)
    i_end = np.random.choice(range(len(border_nodes[j_end])))
    n_source = border_nodes[j_end][i_end]

    full_trajectory = []
    for k in range(n_trajectories):
        border_nodes_red = border_nodes[:]
        border_nodes_red.pop(j_end)
        d = 0
        while d<15:
            j_end = int(np.random.choice(3, size=1))
            i_end = np.random.choice(range(len(border_nodes_red[j_end])))
            n_end = border_nodes_red[j_end][i_end]
            # making sure that there is some distance between source and end nodes
            d = np.sqrt((n_end[0]-n_source[0])**2 + (n_end[1]-n_source[1])**2)

        path, _ = generate_trajectory(g, h, w, n_source, n_end)
        full_trajectory.append(path)
        
        n_source = n_end

    full_trajectory = np.vstack(full_trajectory)
    full_trajectory_smooth = interpolate_trajectory(full_trajectory, scale//8, s=10)
        
    # render
    fps = 30
    print(f"Generate video {filename_output}")
    video = cv2.VideoWriter(filename_output, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w*scale,h*scale))
    
    for i in range(len(full_trajectory_smooth)):
        if i%100==0:
            print(i)
        if use_bubble_background:
            if i%10==0:
                background_image = np.zeros((h,w,3), np.uint8)
                for j in range(200):
                    cx = int(np.random.randint(w))
                    cy = int(np.random.randint(h))
                    s = np.random.randint(1,10)
                    background_image = cv2.circle(background_image, (cx,cy), s, 
                                                  np.random.randint(40, 215, 3).tolist(), -1)                
            img = generate_image(full_trajectory_smooth[i]*scale, h*scale, w*scale, background_image=background_image)
        else:
            img = generate_image(full_trajectory_smooth[i]*scale, h*scale, w*scale, background_image=None)
        video.write(img)

        #cv2.imshow('generation', img)
        #cv2.waitKey(0)

    cv2.destroyAllWindows()
    video.release()   
    
if __name__=="__main__":
    
    main(n_trajectories=30,
         filename_output="output_bubble.avi",
         use_bubble_background=True)