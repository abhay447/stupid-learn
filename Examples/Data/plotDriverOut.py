import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('driver_fleet_out.csv')
cs = {0:'r',1:'g',2:'b',3:'y',4:'m',5:'c',6:'k'}
df.plot(kind = 'scatter',x = 'X',y='Y',c = [cs[x] for x in df.Class])
plt.show()
