import os, sys
import numpy as np
import os.path
     
from PIL import Image
import image_patch

pwd = os.pardir;
scriptDir = "/home/cw2189/ML_project/database/curetgrey"   
pitch_size = 101
save_path = "/scratch/cw2189/database_patch"

for i in range(1,62):  

    sample_file = r"%s/sample%02d" % (scriptDir , i);
    for file in os.listdir(sample_file):
        if file.endswith(".png"):
            name_of_file = 'sample%02d_%s_%dx%d' % (i , file.strip(".png").replace(file[:3], '') , pitch_size , pitch_size) 
            completeName = os.path.join(save_path, name_of_file)      
            f = open(completeName, 'wb+')
            final = sample_file+"/"+file;
#	    print(final)
#            print (completeName);
#            f.write(final+"\n");
            x = image_patch.ImagePatchProcessing(final , pitch_size)
#            filelist.append(final);
            
            
#           for i in x :
            tmp = bytes(x);
            f.write(tmp);
#           f.write("\n");
            
    f.close();       
