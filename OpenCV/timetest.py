import time

start=time.time()
print("Start time:", start)
for i in range(1000000):
    print(time.time()-start, end="\r")
    i+1