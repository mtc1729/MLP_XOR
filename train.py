import random
import sys
orig_stdout = sys.stdout
f = open('train.txt', 'w')
sys.stdout = f

for x in range(200):
   m=random.uniform(0, 1)

   n=random.uniform(0, 1)

   if m<0.5 and n<0.5 :
       t=0
   elif m >=0.5 and n>=0.5 :
       t=1
   else :
       t=0
   print m,",",n,",",t            
sys.stdout = orig_stdout
f.close()
   
