import sys

p=0
for line in open(sys.argv[1], "rt"):
    x,y=line.split()
    if float(x) > 0.5:
        p+=1
print(p)
