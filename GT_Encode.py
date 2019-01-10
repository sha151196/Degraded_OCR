import os

def writetext(filepath,gt):
    f =open(filepath,"a+")
    f.write(str(gt+'\n'))

def create_encodedgt(imgfiles,grt_file,encoded_grt_file,mode):
    #Reads groundtruth files; Match it to the image file, and Write Encoded groundtruth text file
    gt_file = open(dir+"/"+grt_file)
    info = gt_file.readline()
    imgfiles.sort()
    
    while info:
    	gt_filename = info.strip('\n').split("@")[0]
    	gt = info.strip('\n').split("@")[-1]

	for i in imgfiles:
	    if os.path.splitext(i)[1].lower() not in ('.jpeg',):
                continue	
	    if(gt_filename == i):             		
	        print gt,i
	        encoded_line= bangla_2_encode(gt)
	        writetext(dir+"/"+encoded_grt_file,gt_filename+"@"+encoded_line)
	        info = gt_file.readline()
	    else:
	        continue
		print "NO MATCH FOUND"
		info = gt_file.readline()

def bangla_2_encode(bangla_line):
    unicode_line = convert_bangla_line_to_unicode(bangla_line)
    unicode_compound_map = replace_compound_in_unicode_line(unicode_line, dict_cmpd_file)
    encoded_line = convert_unicode_line_to_custom(unicode_compound_map, dict_single_file)
    #Reordering modifiers
    reorderlist = ['m3', 'm8', 'm9']
    reordered_line = reorder_modifier_in_custom_line(encoded_line, reorderlist)
    return reordered_line

def convert_bangla_line_to_unicode(line):
    #unicode line words are seperated by * characters are seperated by space
    words=line.split()
    unicode_line=""
    for w in words:
        unicoded = w.decode("utf-8").encode("unicode-escape")
        unicode_characters = unicoded.split("\\")[1:]
        for uc in unicode_characters:
            uc = uc[:5]
            unicode_line=unicode_line+"\\"+uc+" "
        unicode_line=unicode_line+"* "
    return unicode_line

def convert_unicode_line_to_custom(line,dbfile):
    #custom line words are seperated by * characters are separated by space
    words=line.split("*")
    custom_line=""
    for w in words:
        characters=w.split()
        for ch in characters:
            custom_tag=ch
            f = open(dir+"/"+dbfile)
            line=f.readline()
            # Looking for a custom tag for this unicode character from dbfile
            #if not found then original character will be retained
            while line:
                info=line.strip("\n").split(",")
                if(ch==info[0]):
                    custom_tag=info[-1]
                    break
                else:
                    ch == ch
                    line=f.readline()
            f.close()
            custom_line = custom_line + custom_tag + " "
        custom_line=custom_line+"* "
    return custom_line

def replace_compound_in_unicode_line(line,compounddbfile):
    f=open(dir+"/"+compounddbfile)
    ln=f.readline()
    while ln:
        info=ln.strip("\n").split(",")
        compound_string=info[1].rstrip()
        compound_tag=info[-1]
        line=line.replace(compound_string,compound_tag)
        ln=f.readline()
    return line

def reorder_modifier_in_custom_line(line,reorderlist):
    #reorder list contains custom label of those modifier that has different phonetic positions
    #line has custom labels
    reorder_line=""
    words=line.split("*")
    for w in words:
        #find characters in words
        chars=w.split()
        reposition=False
        for c in range(len(chars)):
            for m in reorderlist:
                if(m==chars[c]):
                    reposition=True
                    break
            if(reposition):
                temp=chars[c-1]
                chars[c-1]=chars[c]
                chars[c]=temp
                reposition=False
        for ch in chars:
            reorder_line+=ch+" "
        reorder_line+="* "
    return reorder_line

dir = "/home/cvpr/Documents/Shahil/DegradedOCR-master"#Directory
mode="Test" #"Train" for creating encoded groundtruth for Train folder and "Test" for Test folder
dict_cmpd_file = "Dict/BanglaCompositeMap.txt"
dict_single_file = "Dict/AllCharcaters.txt"
dict_file = "Dict/CompositeAndSingleCharacters.txt"
data_file = "Data/ICBOCR-D4"
gt_file = data_file+"/"+mode+"/"+"Groundtruth.txt"
encoded_gt_file = data_file+"/"+mode+"/"+"Encoded_gt.txt"

#Reading Img_files
img_files = [img_file for img_file in os.listdir(dir+"/"+data_file+"/"+mode+"/"+"Line_Images")]
create_encodedgt(img_files,gt_file,encoded_gt_file,mode)
