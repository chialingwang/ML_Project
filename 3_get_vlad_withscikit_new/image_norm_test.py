'''

Homework 4

Name     : Tailin Lo
NetID    : tl1720
N number : N15116873
Email    : tl1720@nyu.edu

======================================

Load Project File Lib 

'''

import os, sys
import os.path
import struct
import re


def read_fileName_double(file , size):
    result = []
    f = open(r'%s' %(file), 'rb+')
#    print(f)
    read_data = f.read();
    n = size*size
    #for i in read_data:
    #    print(i);
    count = 0
    temp = []
    lable = []
#    for i in range(0 , 24 , 8):
    for i in range(0 , len(read_data) , 8):
       
        b = read_data[i:i+8]
#        print(b)
        data = struct.unpack('d', b)
#        print(data[0])
        temp.append(data[0])
        count += 1
        if(count == n):
            result.append(temp)
            temp = []
            count = 0
    return result

def read_fileName(file , size):
    result = []
    f = open(r'%s' %(file), 'rb+')
#   print(f)
    read_data = f.read().replace('[','').replace(']','').replace(' ','').split(',')
#    print(read_data)
    n = size*size
    #for i in read_data:
    #    print(i);
    count = 0
    temp = []
    lable = []
#    for i in range(0 , 24 , 8):
    for i in range(0 , len(read_data)):
       
        data = read_data[i]
        #print(data)
        temp.append(int(data))
        count += 1
        if(count == n):
            result.append(temp)
            temp = []
            count = 0
    return result

#print(train[10])
def load_data(filelist , patch):

    record = dict([('data', []), ('target', []), ('filename', [])])
    for filename in filelist:
        #print(filename)
	new_name = filename.split('/')[4].split('_')[0]
        #print(new_name)           
        sample = int(re.findall(r'\d+', new_name)[0])
        #print(sample)
        read_file = []
        read_file = read_fileName(filename , patch)
        for element in read_file:
            record['filename'].append(filename)
            record['target'].append(sample)
            record['data'].append(element)

        #print(read_file[0])
        #print(len(read_file))
    return record


def load_sig_data(filename , patch):
    file_target = []
    record = dict([('data', []), ('target', []), ('filename', [])])
    new_name = filename.split('/')[4].split('_')[0]
    sample = int(re.findall(r'\d+', new_name)[0])
        #print(sample)
    read_file = []
    read_file = read_fileName(filename , patch)
    file_target.append(sample)
    for element in read_file:
        record['filename'].append(filename)
        record['target'].append(sample)
        record['data'].append(element)

        #print(read_file[0])
        #print(len(read_file))
    return record , file_target

