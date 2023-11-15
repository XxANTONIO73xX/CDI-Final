import os
from os import walk
actualPath = os.getcwd()
w = walk(actualPath+"/data")
for (dirpath, dirnames, filenames) in w:
    for file in filenames:
        if file[-4:] == ".csv":
            os.remove(dirpath+"/"+file)