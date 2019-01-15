# -*- coding: utf-8 -*-
import sys
import codecs
import os
print sys.stdout.encoding
outpu_dir ="/data0/xijun1/tfqieci/data/doc"
source_file = "/data0/xijun1/data/result.txt"
if __name__ == '__main__':
   with codecs.open(source_file,'r') as f:
        for file_ind ,line in enumerate(f):
	    print str("%06d" % (file_ind+1))
            output_file = os.path.join(outpu_dir , str("%06d" % (file_ind+1))+".txt")
	    with  codecs.open( output_file ,'w') as fw:
		for data in line.split("。"):
		  #print data.encode("utf8")
		   if len(data) > 0 :
		      #data = data.replace(" ","/x ")
		      fw.write((data+"\n"))
