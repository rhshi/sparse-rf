import numpy as np
import matplotlib.pyplot as plt

def plot_w(w, inds_track):
    w_plt = w.copy()
    w_plt[w_plt == 0] = np.nan
    plt.figure(figsize=(16, 12))
    plt.scatter(range(len(w_plt)), w_plt, s=10)
    # plt.hlines(0, 0, len(w_plt), colors=["black"])
    for i in range(len(inds_track)):
        plt.vlines(i*len(w_plt)/(2*(len(inds_track))), np.min(w), np.max(w), colors=["indianred"], linestyles="dashed")
        plt.text(x=i*len(w_plt)/(2*(len(inds_track)))+len(w_plt)/(4*(len(inds_track))), y=np.max(w)+1, s="{}".format(inds_track[i]), c="indianred")
    for i in range(1, len(inds_track)+1):
        plt.vlines((i*len(w_plt))/(2*(len(inds_track)))+len(w_plt)/2, np.min(w), np.max(w), colors=["maroon"], linestyles="dashed")
        plt.text(x=(i-1)*len(w_plt)/(2*(len(inds_track)))+len(w_plt)/(4*(len(inds_track)))+len(w_plt)/2, y=np.max(w)+1, s="{}".format(inds_track[i-1]), color="maroon")
    plt.vlines(len(w_plt)/2, np.min(w), np.max(w), colors=["brown"], linestyles="dashed")
    plt.show()
    return 