# -*- coding: utf-8 -*-
# @Author: Koth Chen
# @Date:   2016-10-15 14:49:40
# @Last Modified by:   Koth
# @Last Modified time: 2016-12-09 20:33:30
import sys
import os
import codecs
totalLine = 0
longLine = 0
maxLen = 200


def processToken(token, collect, out, end):
  global totalLine
  global longLine
  global maxLen
  nn = len(token)
  #print token
  while nn > 0 and token[nn - 1] != u'/':
    nn = nn - 1
  if nn ==0  and 0 < len(token):
      nn = len(token)+1;
  token = token[:nn - 1].strip()
  #print unicode(token.decode('utf8')) #token.decode('utf8')
  ustr =token  #unicode(token.decode('utf8'))
  #print ustr.encode("utf8")
  for u in ustr:
    collect.append(u)
  uline = u''
  if token == u'。' or end:
    if len(collect) > maxLen:
      longLine += 1
    totalLine += 1
    for s in collect:
      if uline:
        uline = uline + u" " + s
      else:
        uline = s
    #print uline.encode('utf8')
    #print uline
    out.write("%s\n" % (str(uline.encode('utf8'))) )
    del collect[:]


def processLine(line, out):
  line = line.strip()
  nn = len(line)
  seeLeftB = False
  start = 0
  collect = []
  try:
    for i in range(nn):
      if line[i] == ' ':
        if not seeLeftB:
          token = line[start:i]
          if token.startswith('['):
            tokenLen = len(token)
            while tokenLen > 0 and token[tokenLen - 1] != ']':
              tokenLen = tokenLen - 1
            token = token[1:tokenLen - 1]
            ss = token.split(' ')
            for s in ss:
              processToken(s, collect, out, False)
          else:
            processToken(token, collect, out, False)
          start = i + 1
      elif line[i] == '[':
        seeLeftB = True
      elif line[i] == ']':
        seeLeftB = False
    if start < nn:
      token = line[start:]
      if token.startswith('['):
        tokenLen = len(token)
        while tokenLen > 0 and token[tokenLen - 1] != ']':
          tokenLen = tokenLen - 1
        token = token[1:tokenLen - 1]
        ss = token.split(' ')
        ns = len(ss)
        for i in range(ns - 1):
          processToken(ss[i], collect, out, False)
        processToken(ss[-1], collect, out, True)
      else:
        processToken(token, collect, out, True)
  except Exception as e:
    pass


def main(argc, argv):
  global totalLine
  global longLine
  if argc < 3:
    print("Usage:%s <dir> <output>" % (argv[0]))
    sys.exit(1)
  rootDir = argv[1]
  with  open(argv[2], "w") as out:
      for dirName, subdirList, fileList in os.walk(rootDir):
          curDir = os.path.join(rootDir, dirName)
          for file in fileList:
             if file.endswith(".txt"):
                curFile = os.path.join(curDir, file)
                # print("processing:%s" % (curFile))
                with codecs.open(curFile, "r","utf8") as fp:
                    for line in fp:
                        line = line.strip()
			processLine(line, out)
  print("total:%d, long lines:%d" % (totalLine, longLine))


if __name__ == '__main__':
  main(len(sys.argv), sys.argv)
