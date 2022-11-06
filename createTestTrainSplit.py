#!/usr/bin/python3
import os, shutil 

if __name__ == "__main__":
    path = "res/Images/"
    test = "res/test/"
    train = "res/train/"
    isExist = os.path.exists(test)
    
    if not isExist: 
        os.makedirs(test)
    isExist = os.path.exists(train)
    if not isExist: 
        os.makedirs(train)

    files = os.listdir(path)
    train_split = int(len(files) * 0.8)
    i = 1 
    while i < len(files)+1:
        src = path + files[i] 
        if i < train_split: 
            dst = train + files[i]
            shutil.copy(src, dst)
        else:
            dst = test + files[i]
            shutil.copy(src, dst)
        i += 1 


                


