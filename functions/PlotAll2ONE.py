import numpy as np
#from dtw import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn import preprocessing

from IDK_T import IDK_T
#from IDKofTwoTS import IDKofTwoTS
from Utilities import prepocessSubsequence


def drawIDK_T_discords(TS, idk_scores, number=3):
    sorted_index = np.argsort(idk_scores)
    color_list = ['r', 'y', 'b', 'g', 'c', 'm', 'k']  # c天蓝,m紫色,k黑色
    cur = 0
    # x_major_locator = MultipleLocator(cycle)
    # # 把x轴的刻度间隔设置为1，并存在变量里
    # y_major_locator = MultipleLocator(10)
    # # 把y轴的刻度间隔设置为10，并存在变量里
    # ax = plt.gca()
    # # ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)

    # plt.xlim(0, 9900)
    plt.plot(TS,color='black')
    for index in sorted_index:
        if number <= 0:
            break;
        number -= 1
        ls = range(index * cycle, (int)(index * cycle + cycle))
        ly = TS[ls]
        plt.plot(ls, ly, color='orange')
        # cur=(cur+1)%7
    plt.ylabel('value')

if __name__ == "__main__":

    cycle = 1000
    anomaly_cycles = [1,9,12];  # 异常序列所在周期  39是周六多一天小供电的异常
    redlist=[]
    redlist.append((0,0))

    df = pd.read_csv(
        "Discords_Data/TEK.txt",
        header=None)
    # df = pd.read_csv("RealDatasets/FEW_DISCORDS_DATASETS/ECG_Hotsax/qtdbsel102.txt", header=None)


    df = np.array(df).reshape((-1,1))
    for it in anomaly_cycles:
        redlist.append((it*cycle,it*cycle+cycle))
    redlist.append((len(df),len(df)))

#    fig, ax1 = plt.subplots()
    scaler = preprocessing.MinMaxScaler()



        # min_max_scaler = preprocessing.MinMaxScaler()
        # result_list = min_max_scaler.fit_transform(result_list.reshape(-1, 1)).reshape(m)
        #plot_y=np.full(m,0.0978)





        #ax = plt.gca()
        #ax.set_xscale('log', basex=2)
        # # y_major_locator = MultipleLocator(0.001)
        # # ax.yaxis.set_major_locator(y_major_locator)
        # x_major_locator = MultipleLocator(5)
        # ax.xaxis.set_major_locator(x_major_locator)
        # plt.xlim(0, m)
        # # y_major_locator = MultipleLocator(5)
        # # ax.yaxis.set_major_locator(x_major_locator)
        # plt.ylim(0, 1300)
        # #plt.ylim(0.09,0.1)
#     data = pd.read_csv("MP_result/TEK_ALL(w=1000).csv")
#     mp_list = np.array(data['MP']).reshape((-1, 1))
#     mp_list = scaler.fit_transform(mp_list)
#     ax1.plot(mp_list, label='MP', color='b')
#     # thredsome = readcsvPlot.findthredsomeInSlidingWindowIDK(mp_list, redlist, width=cycle)
#     # line = np.full(len(mp_list), thredsome)
#     # plt.plot(line, color='b',label='MP threshold',linestyle='--')
#     data = pd.read_csv("NORMA_result/Norm_TEK_1000_1.2_sample.txt")
#     norma_list = np.array(data).reshape((-1, 1))
#     norma_list = scaler.fit_transform(norma_list)
#     ax1.plot(norma_list, label='NormA', color='red')
#     # ax1.set_xlabel(r'$i$ in displacement $\pi/i$')
#     # plt.set_ylabel("anomaly score")
#
#     plt.xlim(0, len(df))
#     ax1.set_ylabel('Anomaly score')
#
#     ax2 = ax1.twinx()
#     plt.subplot(3, 1, 2)

#     barx = []
#     plotx = []
#     lo = 0
#     while lo + cycle - 1 < len(df):
#         barx.append(lo)
#         lo += cycle
#     plt.xlim(0, len(df))
#
#     # IDK_ano=scaler.fit_transform(np.array(-p).reshape((-1,1)))
#     ax2.bar(barx, p.reshape(-1), width=900, color='skyblue', align='edge', label=r'IDK$^2$')
#     for i in barx:
#         plotx.append(i + cycle / 2 - 1)
#     plt.legend(loc='center right', prop={'size': 7})
#
#     plt.ylabel('Similarity score')


#    thredsome = findthredsomeInCycleIDK(p, anomaly_cycles);
#    line = np.full(len(p), thredsome)

    # my_x_ticks = np.arange(len(p))  # 横坐标设置0,1,...,len(acc)-1,间隔为1
    # # my_x_ticks = np.arange(0,len(acc),2) # 横坐标设置0,2,...,len(acc)-1，间隔为2
    # plt.xticks(my_x_ticks)
    # plt.plot(line, color='g')
    # plt.xlim(-0.5, 18)
   # plt.ylabel('Similarity score')
    # legend = plt.legend(title=r'IDK$^2$')

    # legend._legend_box.align = 'left'
    # print("auc", roc_auc_score(get_label(X, cycle, anomaly_cycles), -p))
    # print("acc", get_IDKpAtk(len(anomaly_area_list), p, anomaly_area_list, cycle))
    plt.subplot(3, 1, 1)
    #df=df[230:]
    #redlist=range(2200-230,2551-230)
    X = df.copy()
    prepocessSubsequence(X, cycle)
    p = IDK_T(X, t=100, width=cycle, psi1=2, psi2=8)

    # plt.plot(range(0,redlist[0]),df[range(0,redlist[0])],color='black')
    # plt.plot(range(redlist[1],4800),df[range(redlist[1],4800)],color='black')
    # plt.plot(redlist,df.reshape(-1)[redlist],color='orange')
    # plt.xlim(0, 4800)
    drawIDK_T_discords(df, p, len(anomaly_cycles))
    plt.xlim(0,len(df))
    # # plt.plot(TS, color='b')
    # # for it in redlist:
    # #     ls = range(it[0], it[1])
    # #     ly = TS[ls]
    # #     plt.plot(ls, ly, color='r')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.ylabel('value ',fontsize=22)
    plt.subplot(3, 1, 3)
    data = pd.read_csv("MP_result/TEK(w=1000).csv")

    mp_list = np.array(data['MP']).reshape((-1,1))
    mp_list=scaler.fit_transform(mp_list)
    plt.plot(mp_list,label='STOMP', color='b')
    # thredsome = readcsvPlot.findthredsomeInSlidingWindowIDK(mp_list, redlist, width=cycle)
    # line = np.full(len(mp_list), thredsome)
    # plt.plot(line, color='b',label='MP threshold',linestyle='--')
    data = pd.read_csv("NORMA_result/Norm_TEK_sample_1000.txt")
    norma_list = np.array(data).reshape((-1,1))
    norma_list=scaler.fit_transform(norma_list)
    plt.plot(norma_list, label='NormA',color='red')
    #ax1.set_xlabel(r'$i$ in displacement $\pi/i$')
    #plt.set_ylabel("anomaly score")
    # 设置刻度字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.ylabel('score',fontsize=22)
    # legend = plt.legend(title='DTW')
    # legend._legend_box.align = 'left'

 #   plt.subplot(2, 1, 2)
#    p = IDK_T(df, t=100, width=cycle, psi=8, psi2=4)[0][0]
    p=scaler.fit_transform(np.array(p).reshape(-1,1))
    barx = []
    plotx = []
    lo = 0
    while lo + cycle - 1 < len(df):
        barx.append(lo)
        lo += cycle
#    plt.xlim(0, len(df))
    #
    # #IDK_ano=scaler.fit_transform(np.array(-p).reshape((-1,1)))
    plt.bar(barx, p.reshape(-1), width=cycle-100, color='skyblue' ,align='edge',label=r'IDK$^2$')

    for i in barx:
        plotx.append(i + cycle / 2 - 1)
    plt.legend(loc='upper right', prop={'size': 22},bbox_to_anchor=(1.148,1))
    plt.xlim(0, len(df))
   # plt.ylabel('Similarity score')
 ##   plt.plot(plotx, p, linestyle='--')
    #    thredsome = findthredsomeInCycleIDK(p, anomaly_cycles);
    #    line = np.full(len(p), thredsome)

    # my_x_ticks = np.arange(len(p))  # 横坐标设置0,1,...,len(acc)-1,间隔为1
    # # my_x_ticks = np.arange(0,len(acc),2) # 横坐标设置0,2,...,len(acc)-1，间隔为2
    # plt.xticks(my_x_ticks)
    # plt.plot(line, color='g')
    # plt.xlim(-0.5, 18)
#    plt.ylabel('Similarity score')

    # ax2 = ax1.twinx()
    # ax2.plot(IDK_T(df,t=100,psi=8,width=cycle,psi2=4)[0][0], label=r'IDK^2')
    # # ax2.set_xscale('log', basex=2)
    # # ax2.set_yticks(np.arange(0, 1.1, 0.1))
    # y_major_locator = MultipleLocator(0.1)
    # ax2.yaxis.set_major_locator(y_major_locator)
    # plt.ylim(0, 1.05)
    # ax2.set_ylabel("IDK similarity")
    # plt.legend(loc='center right', bbox_to_anchor=(0.86,0.7))

    # for i in range(m):
    #     plt.text(ivalue[i], idk_similarity[i+1] - 0.001, '%.4f' % idk_similarity[i+1], ha='center', va='bottom', fontsize=9)
    #plt.text(0,5,'1',ha='center', va='bottom', fontsize=9)
    # ax1.legend()
    # plt.legend()

    plt.show()