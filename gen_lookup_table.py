import sys
import os
import numpy as np
sys.path.append("../../release")

from hw_nas_bench_api import HWNASBenchAPI as HWAPI
hw_api = HWAPI("../../release/HW-NAS-Bench-v1_0.pickle", search_space="fbnet")

if not os.path.exists("fbnet"):
    os.mkdir("fbnet")

lookup_tables = hw_api.get_op_lookup_tables()
for k in lookup_tables:
    device_name, metric_type = k.split("_")
    if not os.path.exists(os.path.join("fbnet", device_name)):
        os.mkdir(os.path.join("fbnet", device_name))
    np.save(os.path.join("fbnet", device_name, "{}_lookup_table.npy".format(metric_type)), lookup_tables[k])