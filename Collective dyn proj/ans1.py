import math
import io

a=1140671485.0
c=12820163.0
m=pow(2,24)
z=[]
u=[]
z.append(1234.0)
for i in range(1,100):
    z.append(math.ceil((a*z[i-1]+c)%m))
    u.append(math.ceil(z[i]/m))

print(z)

