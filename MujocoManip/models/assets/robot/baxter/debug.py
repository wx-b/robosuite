import re, sys

f = open("robot.xml").read()

l = re.findall('%s="(.*?)"'%sys.argv[1], f)
print(l)
print(len(l),len(list(set(l))))
