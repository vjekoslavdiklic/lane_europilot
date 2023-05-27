import csv
import matplotlib
import matplotlib.pyplot as plt
import glob
import numpy as np
from matplotlib import gridspec
path_to_csv=r'../nn_all4_40step/results/night/ori'
csvfiles=glob.glob(path_to_csv+"/*.csv")

plt.ioff


data=dict()
for eachcsv in csvfiles:
    with open(eachcsv) as csvfile:
        curname=eachcsv.split('/')[-1]
        data[curname]=dict()
        a=csv.reader(csvfile)
        firstrow=1
        for row in a:
            if firstrow:
                z=row
                for zname in z:
                    data[curname][zname]=list()
                firstrow=0
            else:
                i=0
                for zname in z:
                    data[curname][zname].append(np.array(row[i]).astype(float))
                    i=i+1

keylist=(list(data.keys()))
keylist.sort()

'''
for each in keylist[-2:]:
    x=np.array(data[each]['NN Time']).astype(float)
    y=np.array(data[each]['Steering wheel angle']).astype(float)
    y = ((y / (65535 / 2)) * 540)
    plt.plot(x,y,label=' '.join(each.split('_')[0:2]))
    #break
plt.legend()
plt.xlabel("time")
plt.ylabel("controller steering value")
plt.show()


for each in keylist[-4:-2]:
    x=np.array(data[each]['NN Time']).astype(float)
    y=np.array(data[each]['Steering wheel angle']).astype(float)
    y = ((y / (65535 / 2)) * 540)
    plt.plot(x,y,label=' '.join(each.split('_')[0:2]))
    #break
plt.legend()
plt.xlabel("time")
plt.ylabel("controller steering value")
plt.show()


for each in keylist[-6:-4]:
    x=np.array(data[each]['NN Time']).astype(float)
    y=np.array(data[each]['Steering wheel angle']).astype(float)
    y = ((y / (65535 / 2)) * 540)
    plt.plot(x,y,label=' '.join(each.split('_')[0:2]))
    #break
plt.legend()
plt.xlabel("time")
plt.ylabel("controller steering value")
plt.show()

 
for each in keylist[-8:-6]:
    x=np.array(data[each]['NN Time']).astype(float)
    y=np.array(data[each]['Steering wheel angle']).astype(float)
    y=((y/(65535/2))*540)
    plt.plot(x,y,label=' '.join(each.split('_')[0:2]))
    #break
plt.legend()
plt.xlabel("time")
plt.ylabel("controller steering value")
plt.show()



for each in keylist:
    x=np.array(data[each]['NN Time']).astype(float)
    y=np.array(data[each]['Steering wheel angle']).astype(float)
    y=((y/(65535/2))*540)
    plt.plot(x,y,label=' '.join(each.split('_')[0:2]))
    #break
plt.legend()
plt.xlabel("time")
plt.ylabel("controller steering value")
plt.show()
'''

def make1xnplot(data,show=1,savename='False'):
    fig=plt.figure()
    keylist = (list(data.keys()))
    keylist.sort()
    numofplots = len(keylist)
    gs = gridspec.GridSpec(numofplots, 1,)

    maxy=0
    miny=0
    maxx=0
    minx=0

    for each in keylist:
        curx=np.array(data[each]['NN Time']).astype(float)
        cury=np.array(data[each]['Steering wheel angle']).astype(float)
        cury = ((cury / (65535 / 2)) * 540)
        if maxy<max(cury):
            maxy=max(cury)
        if maxx<max(curx):
            maxx=max(curx)

        if miny>min(cury):
            miny=min(cury)
        if minx>min(curx):
            minx=min(curx)
    ##add offset to axis
    maxx=np.ceil(maxx*1.35)
    plotiter=0
    firstpass=1
    for each in keylist:
        if firstpass==1:
            ax0 = plt.subplot(gs[plotiter])
            curax=ax0
            firstpass=0
        else:
            an=gs[plotiter]
            curax = plt.subplot(gs[plotiter])#, sharex=ax0)
            #curax.set_xticks([])

        curx=np.array(data[each]['NN Time']).astype(float)
        curx=curx-curx[0]
        cury=np.array(data[each]['Steering wheel angle']).astype(float)
        cury=((cury/(65535/2))*540)

        curlabel=('e_'.join((''.join("".join(each.split('_')[0:2]).split('PilotNet'))).split('epoch'))).split('v1')[1]

        curax.plot(curx,cury,label=curlabel)
        #curax.minorticks_on
        #curax.grid(True, which='both')
        curax.set_xlim([minx, maxx])
        curax.set_ylim([miny, maxy])
        curax.set_yticks([-100,0,100])
        if plotiter<numofplots-1:
            curax.set_xticks([])
        curax.legend(loc="upper right")
        plotiter+=1
    #curax.set_xticks([])
    #curax.set_xticks(list(range(minx,maxx,10)))
    fig.supylabel('Steering wheel value [deg]')
    fig.supxlabel('Time to crash [sec]')
    plt.subplots_adjust(hspace=.0)
    if savename!=False:
        plt.savefig(savename)
    if show:
        plt.show()


make1xnplot(data,show=1,savename='highwaysunny.pdf')

fig = plt.figure()
#ax = fig.add_subplot(111)
gs = gridspec.GridSpec(8, 1, )

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1], sharex = ax0)
ax2 = plt.subplot(gs[2], sharex = ax0)
ax3 = plt.subplot(gs[3], sharex = ax0)

ax4 = plt.subplot(gs[4], sharex = ax0)
ax5 = plt.subplot(gs[5], sharex = ax0)
ax6 = plt.subplot(gs[6], sharex = ax0)
ax7 = plt.subplot(gs[7], sharex = ax0)

x1 = np.array(data[keylist[0]]['NN Time']).astype(float)
y1 = np.array(data[keylist[0]]['Steering wheel angle']).astype(float)
x2 = np.array(data[keylist[1]]['NN Time']).astype(float)
y2 = np.array(data[keylist[1]]['Steering wheel angle']).astype(float)

x3 = np.array(data[keylist[2]]['NN Time']).astype(float)
y3 = np.array(data[keylist[2]]['Steering wheel angle']).astype(float)
x4 = np.array(data[keylist[3]]['NN Time']).astype(float)
y4 = np.array(data[keylist[3]]['Steering wheel angle']).astype(float)

x5 = np.array(data[keylist[4]]['NN Time']).astype(float)
y5 = np.array(data[keylist[4]]['Steering wheel angle']).astype(float)
x6 = np.array(data[keylist[5]]['NN Time']).astype(float)
y6 = np.array(data[keylist[5]]['Steering wheel angle']).astype(float)

x7 = np.array(data[keylist[6]]['NN Time']).astype(float)
y7 = np.array(data[keylist[6]]['Steering wheel angle']).astype(float)
x8 = np.array(data[keylist[7]]['NN Time']).astype(float)
y8 = np.array(data[keylist[7]]['Steering wheel angle']).astype(float)


ax0.plot(x1,y1,label=('e_'.join((''.join("".join(keylist[0].split('_')[0:2]).split('PilotNet'))).split('epoch'))).split('v1')[1])
ax1.plot(x2,y2,label=('e_'.join((''.join("".join(keylist[1].split('_')[0:2]).split('PilotNet'))).split('epoch'))).split('v1')[1])
ax2.plot(x3,y3,label=('e_'.join((''.join("".join(keylist[2].split('_')[0:2]).split('PilotNet'))).split('epoch'))).split('v1')[1])
ax3.plot(x4,y4,label=('e_'.join((''.join("".join(keylist[3].split('_')[0:2]).split('PilotNet'))).split('epoch'))).split('v1')[1])

ax4.plot(x5,y5,label=('e_'.join((''.join("".join(keylist[4].split('_')[0:2]).split('PilotNet'))).split('epoch'))).split('v1')[1])
ax5.plot(x6,y6,label=('e_'.join((''.join("".join(keylist[5].split('_')[0:2]).split('PilotNet'))).split('epoch'))).split('v1')[1])
ax6.plot(x7,y7,label=('e_'.join((''.join("".join(keylist[6].split('_')[0:2]).split('PilotNet'))).split('epoch'))).split('v1')[1])
ax7.plot(x8,y8,label=('e_'.join((''.join("".join(keylist[7].split('_')[0:2]).split('PilotNet'))).split('epoch'))).split('v1')[1])


ax0.legend(loc="upper right")
ax1.legend(loc="upper right")
ax2.legend(loc="upper right")
ax3.legend(loc="upper right")
ax4.legend(loc="upper right")
ax5.legend(loc="upper right")
ax6.legend(loc="upper right")
ax7.legend(loc="upper right")
fig.suptitle("PilotNet test")
ax7.set(xlabel=("Time to Crash"))
plt.subplots_adjust(hspace=.0)
fig.supylabel('Steering wheel value')
plt.show()
print("stop")