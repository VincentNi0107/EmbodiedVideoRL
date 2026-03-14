# usage: python script/collect_eval_results.py ar
import os
import sys
import json
import numpy as np

task_set1 = ['stack_bowls_two', 'place_cans_plasticbox', 'beat_block_hammer', 'pick_dual_bottles', 'click_alarmclock', 'click_bell', 'shake_bottle_horizontally', 'open_laptop', 'turn_switch', 'press_stapler', 'shake_bottle', 'place_bread_basket', 'grab_roller', 'place_burger_fries', 'place_phone_stand', 'place_object_stand', 'place_container_plate', 'place_a2b_right', 'place_empty_cup', 'adjust_bottle', 'dump_bin_bigbin']

def main(prefix):
    no_zero_set = set()
    base_dir = os.path.join("eval_result", prefix)
    for run in sorted(os.listdir(base_dir)):
        run_dir = os.path.join(base_dir, run)
        if not os.path.isdir(run_dir):
            continue
        result = {}
        for task_name in sorted(os.listdir(run_dir)):
            # if task_name not in task_set1:
            #     continue
            task_dir = os.path.join(run_dir, task_name)
            if not os.path.isdir(task_dir):
                continue
            result_file = os.path.join(task_dir, "_result.txt")
            if not os.path.isfile(result_file):
                continue
            with open(result_file, "r") as f:
                result[task_name] = float(f.readlines()[-1].strip())
        average = np.mean(list(result.values()))
        ss = np.sum(list(result.values()))
        no_zero_set = no_zero_set.union(set([ key for key, v in result.items() if v > 0.2 ]))
        result["average"] = average
        result["sum"] = ss
        print(f"{run}: {average:.3f} over {len(result) - 1} tasks")
        with open(os.path.join(run_dir, "_result.json"), "w") as f:
            json.dump(result, f, indent=4)

    # print(list(no_zero_set))

if __name__ == "__main__":
    main(sys.argv[1])
