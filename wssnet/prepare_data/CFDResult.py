import numpy as np

class CFDResult():
    def __init__(self, filepath, rounding, read_wss, read_vel):
        self.wss = None
        self.wssx = None
        self.wssy = None
        self.wssz = None

        self.vx = None
        self.vy = None
        self.vz = None
        self.speed = None
        self.read_csv(filepath, rounding, read_wss, read_vel)
        

    def read_csv(self, filepath, rounding, read_wss, read_vel):
        """
            Read xyz coordinates and wss from CFD results (.csv)
            XYZ is converted to mm
        """
        arr = np.genfromtxt(filepath, delimiter=',', names=True, skip_header=0)
        # print(arr.dtype.names)
        # ic(filepath)
        # ic(arr.dtype.names)
        
        x, y, z = arr['xcoordinate'], arr['ycoordinate'], arr['zcoordinate']
        
        # print(wss.shape)
        
        # convert this to mm
        self.x = np.round(x * 1000, rounding)
        self.y = np.round(y * 1000, rounding)
        self.z = np.round(z * 1000, rounding)

        if read_wss:
            self.wss = arr['wallshear']
            self.wssx = arr['xwallshear']
            self.wssy = arr['ywallshear']
            self.wssz = arr['zwallshear']
        
        if read_vel:
            self.vx = arr['xvelocity']
            self.vy = arr['yvelocity']
            self.vz = arr['zvelocity']
            self.speed = arr['velocitymagnitude']