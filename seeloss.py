from matplotlib import pyplot as pl


def readfile(filename):
    '''
    :param filename: file with content: 3 avgloss: 0.619076 lr: 0.000500 correct: 1252.000000 precision: 0.376202 recall: 0.458440 mAP: 0.387398
    :return: 6 lists (loss,lr,correct,pre,rec,map)
    '''
    with open(filename) as fp:
        data = fp.readlines()
    data = [item.rstrip() for item in data]
    trainlosses,corrects,precisions,recalls, fscores = [],[],[],[],[]
    for line in data:
        try:
            if line[0] != '#':
                epoch, _tloss, tloss, _fscore, fscore, _pre, pre, _rec, rec = line.split(' ')
                trainlosses.append(float(tloss))
                fscores.append(float(fscore))
                precisions.append(float(pre))
                recalls.append(float(rec))
        except:
            print('expected: avgloss: float lr: float correct: float precision: float recall: float mAP: float')
            print('received: ',line)

    return trainlosses, fscores,precisions,recalls


def plot_one_curve(curve, name,start=0):
    print('Plot ', name, ' curve from epoch ', start, ' to the end ')
    print('You can start from other epochs by add start value at the end of plot_one_curve(...,start) function')
    # print('epoch = ',len(curve))
    # for i in range(start):
    #     curve[i] = curve[start]
    curve = curve[start:]
    epoch = range(1,len(curve)+1)
    # print('epoch = ',epoch)
    pl.figure(figsize=(8, 5))
    pl.plot(epoch, curve, color='red', label=name)
    pl.xlabel('epochs')
    pl.ylabel('value')
    pl.grid(True)
    pl.legend()
    # pl.show()


def plot_multi_curves(curves, names):
    epoch = range(1,len(curves[0])+1)
    pl.figure(figsize=(8, 5))
    colors = ['black','red','green','blue','pink','red','green','blue', 'orange', 'yellow','cyan']
    linestyles = ['-', '-', '-', '-', '-', '--', '--', '--']
    for i,curve in enumerate(curves):
        pl.plot(epoch, curve, color=colors[i], linestyle=linestyles[i], label=names[i])
    pl.xlabel('epochs')
    pl.ylabel('value')
    pl.grid(True)
    pl.legend()
    # pl.show()

def scale01(curve):
    minv,maxv = min(curve),max(curve)
    rangev = float(maxv-minv)
    curve = [(element - minv)/rangev for element in curve]
    return curve


def scale_under1(curve):
    curve = [min(element,1) for element in curve]
    return curve


def plot(startfrom=0):
    trainloss,fscore,precision,recall = readfile('savelog.txt')
    plot_one_curve(trainloss,'loss training',startfrom)
    for i in range(startfrom):
        trainloss[i] = trainloss[startfrom]
    # trainloss = scale01(trainloss)
    # plot_multi_curves([precision,recall,correct,map,trainloss],['precision','recall','correct','mAP','train loss'])
    plot_multi_curves([fscore, precision, recall],['fscore', 'precision', 'recall'])
    pl.show()

if __name__ == '__main__':
    import sys
    if len(sys.argv) ==1:
        plot()
    elif len(sys.argv)==2:
        startfrom = int(sys.argv[1])
        plot(startfrom)
    else:
        print('Usage:')
        print('python python seeloss_epoch.py [startfrom]')
