from __future__ import print_function
import math
import numpy as np
import os,sys
from PIL import Image
from scipy import ndimage

#First
def readMainH5(data_file,encoded_gt,mode,maxw,maxh,readamount,write_sequences=False,binarize=False):
    #Takes input as image files and groundtruth as text files and collects features, targets and sequence_lengths
    #maxw and maxh are maximum width(cols) and maximum height(rows) of all samples train+test+val
    files= [filename for filename in os.listdir(data_file + "/"+mode+"/"+ "Line_Images")]
    #print (files)
    samplenames=list(files)
    total=len(samplenames)
    #print (total)
    all_x=[]
    all_y=[]
    all_sampleid=[]
    seq_lengths=[]
    if(write_sequences):
        seq_file = open("Sequence_lengths", "w")
    for t in range(total):
        completed=(t/float(total))*100
	
        if (completed >= readamount):
            break
        sample= samplenames[t]
	
	f = open(data_file+"/"+mode+"/"+ encoded_gt)
	read_sample = f.readline()
	while read_sample:
            target_name = read_sample.strip('\n').split('@')[0]
            	
            if(target_name == sample):
		sampleid = sample
		print ("Matched---> "+ sampleid)
		target = read_sample.strip('\n').split('@')[-1]
                sys.stdout.write("\rRead %s target %s-------Completed %0.2f " % (sampleid, target, completed))
                sys.stdout.flush()
                #testing correctness of target
                temp_target=target.split()
                target_length=len(temp_target)
                #if(target_length>=2):
                all_sampleid.append(sampleid)
		
		#Reading Line_images
		line_file = open(data_file + "/"+mode+"/"+ "Line_Images"+"/"+sampleid)
		line = ndimage.imread(line_file, mode = "L")
		print ("Reading images")
                features=np.array(line) # H x W
		print (features.size)
                if(binarize):
                    features=convert_to_binary(features)
                seq = len(features[0])
                #Test if feature sequence is sufficiently long CTC requirement
                #if((seq/8.0)>(target_length*2)):
                #Now Resize Image to given Max_W Max_H
                padded_feature=pad_x_single(features,[maxh,maxw])
                all_x.append(padded_feature) # N x W x H x 1
                all_y.append(target)
                seq_lengths.append(seq)
                if(write_sequences):
                    seq_file.write(sampleid+","+str(seq) + "\n")
		read_sample = f.readline()

	    else:
		read_sample = f.readline()

    print("Reading ",mode," complete")
    if(write_sequences):
        seq_file.close()
    return all_x,all_y,seq_lengths,all_sampleid

#Second
def findDistinctCharacters(targets):
    '''
    Reads all targets (targets) and splits them to extract individual characters
    Creates an array of character-integer map (char_int)
    Finds the maximum target length
    Finds number of distinct characters (nbclasses)
    :param targets:
    :return char_int,max_target_length,nbclasses:
    '''
    total=len(targets)
    max_target_length=0
    char_int=[]
    all_chars=[]
    total_transcription_length=0 #Total number of characters
    for t in range(total):
        this_target=targets[t]
        chars=this_target.split()
        target_length=len(chars)
        total_transcription_length=total_transcription_length+target_length
        if(target_length>max_target_length):
            max_target_length=target_length
        for ch in chars:
            all_chars.append(ch)

    charset = list(set(all_chars))
    '''
    char_int.append("PD") #A special character representing padded value
    for ch in charset:
        char_int.append(ch)
    '''
    nbclasses = len(charset)
    print("Character Set processed for ", total, " data")
    print(charset)
    '''
    f=open("Character_Integer","w")
    for c in char_int:
        f.write(c+"\n")
    f.close()
    '''
    return charset, max_target_length, nbclasses, total_transcription_length

def pad_x_single(x,maxdim):
    rows=maxdim[0]
    cols=maxdim[1]
    padded_x=np.zeros([rows,cols,1])
    for r in range(len(x)):
        for c in range(len(x[r])):
            padded_x[r][c][0]=x[r][c]
    #print("\tPadding complete for ",total," data")
    return padded_x

#Third
def pad_x(x,maxdim):
    total=len(x)
    rows=maxdim[0]
    cols=maxdim[1]
    padded_x=np.zeros([total,rows,cols,1])
    for t in range(total):
        for r in range(len(x[t])):
            for c in range(len(x[t][r])):
                padded_x[t][r][c][0]=x[t][r][c]
    #print("\tPadding complete for ",total," data")
    return padded_x

#Call inside Training Module
def make_sparse_y(targets,char_int,max_target_length):
    total = len(targets)
    indices=[]
    values=[]
    shape=[total,max_target_length]
    for t in range(total):
        chars=targets[t].split()
        for c_pos in range(len(chars)):
            sparse_pos=[t,c_pos]
            sparse_val=char_int.index(chars[c_pos])
            indices.append(sparse_pos)
            values.append(sparse_val)
    return [indices,values,shape]

def find_max_dims(data_file,mode):
    files= [filename for filename in os.listdir(data_file + "/"+mode+"/"+ "Line_Images")]
    maxw,maxh=0,0
    for i in files:	
	name = str(i)
	if os.path.splitext(i)[1].lower() not in ('.jpeg',):
	    continue
	line_file=open(data_file + "/" + mode + "/" + "Line_Images"+"/"+name)
	line = ndimage.imread(line_file)
	line_arr = np.array(line)
	
	h,w = np.shape(line_arr)[0],np.shape(line_arr)[1]
        if(w>maxw):
            maxw=w
        if(h>maxh):
            maxh=h
    return maxw,maxh

#Adjust Sequence lengths after CNN and Pooling
def adjustSequencelengths(seqlen,convstride,poolstride,maxtargetlength):
    total=len(seqlen)
    layers=len(convstride)
    for l in range(layers):
        for s in range(total):
            seqlen[s]=max(maxtargetlength,math.ceil(seqlen[s]/(convstride[l]*poolstride[l])))
    return seqlen

#Main
def load_data(data_file,encoded_gt,batchsize,readamount,generate_char_table):
    train_maxw, train_maxh=find_max_dims(data_file, "Train")
    test_maxw, test_maxh = find_max_dims(data_file, "Test")
    maxw=max(train_maxw,test_maxw)
    maxh=max(train_maxh, test_maxh)
    print("Maximum Image Dimesion %d %d"%(maxw,maxh))

    train_x, train_y, train_seq_lengths,train_sampleids=readMainH5(data_file,encoded_gt,"Train",maxw,maxh,readamount,write_sequences=True,binarize=True)
    print("Hello")
    test_x,test_y,test_seq_lengths,test_sampleids=readMainH5(data_file,encoded_gt,"Test",maxw,maxh,readamount, binarize=True)

    sampleids=[train_sampleids,test_sampleids]

    train_charset, train_max_target_length, train_nbclasses,train_transcription_length=findDistinctCharacters(train_y)
    test_charset, test_max_target_length, test_nbclasses, test_trainscription_length = findDistinctCharacters(test_y)
    print("Train Char Set ", train_nbclasses, " Test Character set ", test_nbclasses)

    if (train_nbclasses < test_nbclasses):
        print("Warning ! Test set have more characters")

    train_charset.extend(test_charset)

    char_int = []
    if (generate_char_table):
        charset = list(set(train_charset))  # A combined Character set is created from Train and test Character set
        charset.sort()
        charset.insert(0, "PD")
        charset.append("BLANK")
        nb_classes = len(charset)  # For Blank

        for ch in charset:
            char_int.append(ch)

        ci = open("Character_Integer", "w")
        for ch in char_int:
            ci.write(ch + "\n")
        ci.close()
        print("Character Table Generated and Written")
    else:
        ci = open("Character_Integer")
        line = ci.readline()
        while line:
            char = line.strip("\n")
            char_int.append(char)
            line = ci.readline()
        nb_classes = len(char_int)
        print("Character Table Loaded from Generated File")
    print(char_int)

    max_target_length=max(train_max_target_length,test_max_target_length)
    max_seq_len=maxw

    nbtrain=len(train_y)
    nbtest=len(train_y)

    y_train=[]
    y_test=[]

    batches=int(np.ceil(nbtrain/float(batchsize)))
    start=0
    for b in range(batches):
        end=min(nbtrain,start+batchsize)
        sparse_target=make_sparse_y(train_y[start:end],char_int,max_target_length)
        y_train.append(sparse_target)
        start=end

    batches = int(np.ceil(nbtest / float(batchsize)))
    start = 0
    for b in range(batches):
        end = min(nbtest, start + batchsize)
        sparse_target = make_sparse_y(test_y[start:end], char_int, max_target_length)
        y_test.append(sparse_target)
        start = end
    transcription_length=[train_transcription_length,test_trainscription_length]

    return [train_x,test_x],nb_classes,[train_seq_lengths,test_seq_lengths],[y_train,y_test],max_target_length,max_seq_len, maxh,char_int,transcription_length,sampleids

#Convert integer representation of string to unicode representation
def int_to_bangla(intarray,char_int_file,dbfile):
    '''
    Takes an array of integers (each representing a character as given in char_int_file
    dbfile contains global mapping
    :param intarray:
    :param char_int_file:
    :param dbfile:
    :return:unicode string,mapped character string
    '''
    char_int=[]
    f=open(char_int_file)
    line=f.readline()
    while line:
        info=line.strip("\n")
        char_int.append(info)
        line=f.readline()
    f.close()

    chars=[]
    #print("Intarray ",intarray)
    for i in intarray:
        chars.append(char_int[i])
    #print("Custom Classes ",chars)

    banglastring=""
    for ch in chars:
        f=open(dbfile)
        line=f.readline()
        while line:
            info=line.strip("\n").split(",")
            if(info[2]==ch):
                banglastring=banglastring+info[1]+" "
            line=f.readline()
        f.close()
    return banglastring,chars

def find_unicode_info(char,dbfile):
    #returns type and actual unicode position of a character
    f=open(dbfile)
    line=f.readline()
    type="v"
    pos="#"
    while line:
        info=line.strip("\n").split(",")
        #print(info)
        if(len(info)>5):
            #skip line
            line=f.readline()
        else:
            if(char==info[1]):#Found it in DB
                type=info[0]
                if(type=="m"):#its a modifier
                    pos=info[-1]
                break
            line=f.readline()
    f.close()
    return [type,pos]


def reset_unicode_order(unicodestring,dbfile):
    #Takes unicodestring seperated by space
    #returns properly ordered unicodestring
    unicodearray=unicodestring.split()
    unicodearray=[ch.decode("utf-8").encode("unicode-escape") for ch in unicodearray]
    nbchars=len(unicodearray)
    i=0
    while (i<nbchars-2):
        [type, pos]=find_unicode_info(unicodearray[i],dbfile)
        if(type=="m"):# May need swap
            if(pos=="p"):#swap
                temp=unicodearray[i]
                unicodearray[i]=unicodearray[i+1]
                unicodearray[i+1]=temp
                i=i+1
        i=i+1
    reorder_string=""
    for u in unicodearray:
        reorder_string=reorder_string+u.encode("utf-8").decode("unicode-escape")
    return reorder_string

def convert_to_binary(image):
    #image shoud have dim W x H
    r=len(image)
    c=len(image[0])
    for i in range(r):
        for j in range(c):
            if(image[i][j]==255):
                image[i][j]=1
            else:
                image[i][j]=0
    return image

dir = "/home/cvpr/Documents/Shahil/Maitreyee/DegradedOCR-master"
dict_cmpd_file = "Dict/BanglaCompositeMap.txt"
dict_single_file = "Dict/AllCharcaters.txt"
dict_file = "Dict/CompositeAndSingleCharacters.txt"
data_file = "Data_main"
Encoded_gt = "Encoded_gt.txt"
mode = "Train"

