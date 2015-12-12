#coding:utf-8

class Utils(object):

    def __init__(self):
        pass

    def get_data_from_file(self, filePath):
        """
        load data from file
        :type filePath :string
        :param filePath : the absolute path file stored

        :type X: list of float
        :ret X: X coordinate
        
        :type Y: list of float
        :ret Y: Y coordinate

        :type T: list of int
        :ret T: timestamp

        :type P: list of float
        :ret P: pressure value
        """
        with open(filePath) as fp:
            lines = fp.readlines()
            X = []
            Y = []
            T = []
            P = []
            for line in lines[1:]:
                items = line.split()
                X.append(float(items[0]))
                Y.append(float(items[1]))
                T.append(int(items[2]))
                P.append(float(items[6]))
        return X, Y, T, P
