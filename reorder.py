import os.path
from shutil import move
from os import walk
os.chdir("./data/2d_DATA")
currentPath = os.getcwd()
listdir = os.listdir(currentPath)
w = walk(currentPath)
x = 0
for (dirpath, dirnames, filenames) in w:
    if len(filenames) > 0:
        for filename in filenames:
            origin = dirpath + "/" + filename
            destiny = dirpath + "/Subject 0/" + filename
            try:
                os.rename(origin, destiny)
            except:
                pass
    print(dirpath, filenames)