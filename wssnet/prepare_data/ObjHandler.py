import os
import numpy as np

class ObjHandler(object):
    """
        Based on
        https://inareous.github.io/posts/opening-obj-using-py
    """
    def __init__(self, filename, rounded):
        self.vertices = []  # Vertex coordinates
        self.uvs = []   # UV coordinates
        self.normals = [] # Vector normals on the surface
        self.faces = []
        self.face_strings = []
        self.tmap = {}  # (1-based index) Mapping between UV -> Vertex 
        self.shading = ""

        # Depending on the vertex ordering, vertex/normal can be in different order
        self.v2nmap = {} # mapping between vertex -> normals 
        self.n2vmap = {} # mapping between normals -> vertex 

        # generated after parsing the OBJ file
        self.uv_grid = None # grid version of the UV map
        self.uv_grid_index = None # Remap the UV coordinates into xy grid index

        # parse the file with certain decimal rounding
        self._parse(filename, rounded)

        #convert them into np array
        self.uvs = np.asarray(self.uvs)
        self.vertices = np.asarray(self.vertices)
        self.faces = np.asarray(self.faces)

        self.normals = np.asarray(self.normals)
        if len(self.normals) > 0:
            # re-arrange normals just in case the ordering is different with vertex
            self.reorder_normals()

        # generate the UV grid
        if len(self.uvs) > 0:
            self.uv_grid, self.uv_grid_index = self._generate_uv_grid()

    def stats(self):
        print(len(self.vertices), 'vertices')
        print(len(self.uvs), 'uvs')
        print(len(self.tmap), 'uv map')
        print(len(self.faces), 'faces')
        print(len(self.v2nmap), 'normal mapping')
        # print(f'check normal order: vn 0 = vertex {self.nmap[0]}')
        if len(self.v2nmap) > 0:
            print(f'check vertex/normal order: vertex 0 = normal {self.v2nmap[0]}')

    def save_to_obj(self, output_dir, output_name, header):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f'{output_dir}/{output_name}'
        with open(filename, 'w') as f:
            # write headers
            f.write("# Blender v2.91.2 OBJ File\n")
            f.write(f"# {header}\n")
            f.write(f"o {output_name}\n")

        # reorder normals based on the face v//vn mapping
        index_list = np.arange(0, len(self.vertices)) # 0-based index
        v_idx = [self.n2vmap[x] for x in index_list]
        sorted_normals = self.normals[v_idx]

        with open(filename, 'a') as f:
            np.savetxt(f, self.vertices, fmt="v %.6f %.6f %.6f")
            np.savetxt(f, self.uvs, fmt="vt %.6f %.6f")

            np.savetxt(f, sorted_normals, fmt="vn %.4f %.4f %.4f")
            f.write(self.shading)
            for s in self.face_strings:
                f.write(s)
            

    def fill_grid(self, values):
        grid = self.uv_grid.copy()

        grid[self.uv_grid_index[:,0], self.uv_grid_index[:,1]] = values
        grid = grid[1:,:] # cut the duplicate rows, because of the seam
        return grid

    def reorder_normals(self):
        """
            Re-order normals to have the same ordering as Vertices
            This is based on the mapping retrieved on the face definition
            f v/vt/vn ... v/vt/vn 
        """
        print("Reordering normals to vertex index...")
        index_list = np.arange(0, len(self.vertices)) # 0-based index
        v_idx = [self.v2nmap[x] for x in index_list]
        self.normals = self.normals[v_idx]
        
    def get_sorted_uv_based_vertex_index(self):
        """
            Returns a list of the CFD index retrieved from csv
        """
        index_list = np.arange(1, len(self.uvs)+1) # 1-based index
        # get vertex index from UV index

        v_idx = [self.tmap[x]-1 for x in index_list] # v_idx is also 1-based index
        # make it 0 based
        return v_idx

    def _get_coord_map(self, uvx):
        """
            Map UV coordinates to an 0-based index coordinate
            Get a dictionary of 'coodinates': 'index' 
        """
        uvx = np.unique(uvx)
        uvx = np.sort(uvx)

        uvx_map = {}
        for i in range(len(uvx)):
            uvx_map[uvx[i]] = i
        return uvx_map
        
    def _generate_uv_grid(self):
        uvx = self.uvs[:,0]
        uvx_map = self._get_coord_map(uvx)

        uvy = self.uvs[:,1]
        uvy_map = self._get_coord_map(uvy)
            
        # print(uvx_map, len(uvx_map))
        # print(uvy_map, len(uvy_map))

        uv_index = self.uvs.copy()
        # get (x,y) grid index from the UV coordinates, per (row) index
        for uv_coord in uv_index:
            uv_coord[0] = uvx_map[uv_coord[0]]
            uv_coord[1] = uvy_map[uv_coord[1]]
            # print(uv_coord)
            # break
        uv_index = uv_index.astype(int)

        # create the uv_grid
        uv_grid = np.zeros((len(uvx_map), len(uvy_map)))
        
        # fill in the grid with vertex ids
        index_list = np.arange(1, len(uv_index)+1) # based-1 index
        uv_grid[uv_index[:,0], uv_index[:,1]] = index_list

        return uv_grid, uv_index

    def _parse(self, filename, rounded):
        """
            Parse obj file manually
            https://en.wikipedia.org/wiki/Wavefront_.obj_file#Texture_maps
        """
        try:
            f = open(filename)
            for line in f:
                # Read vertices coordinates
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = [float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1])]
                    vertex = [round(vertex[0], rounded), round(vertex[1], rounded), round(vertex[2], rounded)]

                    self.vertices.append(vertex)
                if line[:2] == "s ":
                    self.shading = line
                if line[:3] == "vn ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vn = [float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1])]
                    vn = [round(vn[0], rounded), round(vn[1], rounded), round(vn[2], rounded)]

                    self.normals.append(vn)
                elif line[:3] == "vt ":
                    # Read UV coordinates
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)

                    vertex = [float(line[index1:index2]), float(line[index2:-1])]
                    self.uvs.append(vertex)
                elif line[:2] == "f ":
                    self.face_strings.append(line) # keep the string

                    # Read mapping between V-index and VT-index
                    # The indexes are 1-based, not 0-based
                    face = []
                    # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 
                    text = line.replace("f ", "")
                    points = text.split(" ")
                    for p in points:
                        elements = p.split("/")
                        v = elements[0]
                        vt = elements[1]
                        vn = elements[2] # we need a mapping also, v may not be in the same order as vn

                        face.append(int(v))
                        if vt not in self.tmap and len(self.uvs) > 0:
                            # mapping from vt to v
                            self.tmap[int(vt)] = int(v) # use 1-based index here
                        
                        if vn not in self.v2nmap and len(self.normals) > 0:
                            # mapping from v to vn
                            self.v2nmap[int(v)-1] = int(vn) - 1 # convert to 0-based index directly
                            self.n2vmap[int(vn)-1] = int(v) - 1

                    #end for p / points
                    self.faces.append(face)

            f.close()
        except IOError:
            print(".obj file not found.")