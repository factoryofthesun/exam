import pandas as pd
import numpy as np
import time
import timeit
from timeit import Timer
from exam import compute_probs
import os, sys
import warnings
warnings.filterwarnings("ignore")

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def run_compute(wtp, predicted_effects, iterations_threshold, method):
    compute_probs(wtp, predicted_effects, iterations_threshold = iterations_threshold, method = method)

if __name__ == "__main__":
    # Shorter time efficiency check just varying size
    sizes = [1000, 5000, 10000]
    res_dict = {}
    for method in [2,1]:
        for n_treatments in [2,3]:
            for s in sizes:
                setting_str = f"n_treatments: {n_treatments}, size: {s}"
                res_dict[method] = {setting_str: []}
                tmp_times = []
                for i in range(3):
                    print(f"Trial {i} for size {s}, n_treatments: {n_treatments}, method: {method}")
                    wtp = np.random.uniform(0, 100, size=(s, n_treatments))
                    pte = np.random.uniform(0, 100, size=(s, n_treatments))
                    fcn = Timer(f"run_compute(wtp, pte, 20, {method})",
                                "from __main__ import run_compute, wtp, pte")
                    t = fcn.timeit(number=1)
                    tmp_times.append(t)
                    print(f"Time: {t} seconds")
                avg_time = np.mean(tmp_times)
                res_dict[method][setting_str].append([s, avg_time])

    import matplotlib.pyplot as plt

    formats = ["o-r", "o-b", "o-g", "+--r", "+--b", "+--g", "v:r", "v:b", "v:g"]
    for n, d in res_dict.items():
        for i in range(len(d.keys())):
            setting = list(d.keys())[i]
            data = np.array(d[setting])
            plt.plot(data[:,0], data[:,1], formats[i], label=setting)
        plt.title(f"Timeit: Method {n}, 3 replications")
        plt.xlabel("Size of data")
        plt.ylabel("Time (s)")
        plt.savefig(f"test_figs/clearing_timeit_method_{n}.png")
        plt.clf()

    # Simulate 1e3 - 1e6 sized arrays with different combinations of pbounds, treatments, error thresholds, iterations
    # expsize = 10
    # res_dict = {}
    # sizes = [1000 * 5 ** i for i in range(5)]
    # for pbound in [0,0.1,0.2]:
    #     for iter in [5, 20, 50]:
    #         setting_str = f"p:{pbound},i:{iter}"
    #         for num_treatments in [2,3,4]:
    #             res_dict[num_treatments] = {setting_str: []}
    #             for s in sizes:
    #                 wtp = np.random.uniform(0, 100, size=(s, num_treatments))
    #                 pte = np.random.uniform(0, 100, size=(s, num_treatments))
    #                 fcn = Timer("run_compute(wtp, pte, pbound, iter)",
    #                 "from __main__ import run_compute, wtp, pte, pbound, iter")
    #                 t = fcn.timeit(number=expsize)
    #                 print(f"Setting: {setting_str}, size: {s}, time: {t}")
    #                 res_dict[num_treatments][setting_str].append([s, t])
    #
    # import matplotlib.pyplot as pt
    #
    # formats = ["o-r", "o-b", "o-g", "+--r", "+--b", "+--g", "v:r", "v:b", "v:g"]
    # for n, d in res_dict.items():
    #     for i in range(len(d.keys())):
    #         setting = list(d.keys())[i]
    #         data = np.array(d[setting])
    #         plt.plot(data[:,0], data[:,1], formats[i], label=setting)
    #     plt.title(f"Timeit: {n} treatments, 10 replications")
    #     plt.xlabel("Size of data")
    #     plt.ylabel("Time (s)")
    #     plt.show()
    #     plt.savefig(f"test_figs/clearing_timeit_{n}.png")
    #     plt.clf()
