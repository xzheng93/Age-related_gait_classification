import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot

# code from web
csfont = {'fontname':'Times New Roman'}
plt.figure(figsize=(7, 4))

host = host_subplot(111)
# par = host.twinx()

host.set_xlabel("Window size", fontsize=15, **csfont)
# par.axis["right"].line.set_color("darkorange")
# par.axis["right"].major_ticks.set_color("darkorange")
# par.axis["right"].major_ticklabels.set_color("darkorange")

host.set_ylabel("AUC", fontsize=15, **csfont)
# par.set_ylabel("Segments number of Dataset (n)", color="darkorange" , fontsize=13)

# plt.rcParams['font.sans-serif'] = ['Arial']
# plt.rcParams['axes.unicode_minus'] = False

x = np.arange(0, 6)
group_labels = ['128', '256', '512', '1024', '2048', '5120']
# cnn_data = np.array([0.926, 0.941, 0.931, 0.961, 0.910, 0.866])
# gru_data = np.array([0.889, 0.944, 0.954, 0.958, 0.922, 0.910])
# cnn_lstm_data = np.array([0.912, 0.920, 0.934, 0.927, 0.918, 0.881])

# cnn_data = np.array([0.942, 0.926, 0.936, 0.956, 0.913, 0.883])
# cnn_lstm_data = np.array([0.905, 0.943, 0.957, 0.956, 0.943, 0.927])
# gru_data = np.array([0.897, 0.944, 0.948, 0.959, 0.934, 0.917]


cnn_data = np.array([0.93, 0.94, 0.93, 0.96, 0.91, 0.87])
gru_data = np.array([0.89, 0.94, 0.95, 0.96, 0.92, 0.91])
cnn_lstm_data = np.array([0.91, 0.92, 0.93, 0.93, 0.92, 0.88])

# cnn_data = np.array([0.94, 0.93, 0.94, 0.96, 0.91, 0.88])
# cnn_lstm_data = np.array([0.91, 0.94, 0.96, 0.96, 0.94, 0.93])
# gru_data = np.array([0.90, 0.94, 0.95, 0.96, 0.93, 0.92])

seg_number = np.array([21360, 10680, 5340, 2670, 1335, 534])

host.grid(linestyle="--")
# ax = plt.gca()

host.plot(x, cnn_data, marker='o', color="blue", label="CNN", linewidth=1.5)
host.plot(x, gru_data, marker='v', color="green", label="GRU", linewidth=1.5)
host.plot(x, cnn_lstm_data, marker='*', color="red", label="ConvLSTM", linewidth=1.5)
# par.plot(x, seg_number, color="darkorange", marker='D', label="Segments", linewidth=1.5)

plt.xticks(x, group_labels, fontsize=12, **csfont)
plt.ylim(0.86, 0.98)
plt.yticks(fontsize=12, **csfont)
plt.title("Impact of window size on classification performance (gait normalization)", fontsize=16, **csfont)
# plt.xlabel("Window size", fontsize=13)
# plt.ylabel("AUC", fontsize=13)

plt.legend(loc='best', numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=12, **csfont)
plt.tight_layout()

plt.savefig('../result/model_plots/windows_effect_nor.pdf', format='pdf')
plt.show()
print()
