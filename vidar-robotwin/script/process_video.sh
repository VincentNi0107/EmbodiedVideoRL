python /home/omz1504/code/vidar-robotwin/script/prepare_diffsynth_dataset_per_task.py \
  --data-root /home/omz1504/code/vidar-robotwin/data \
  --task-instruction-dir /home/omz1504/code/vidar-robotwin/description/task_instruction \
  --out-root /home/omz1504/code/DiffSynth-Studio/data/robotwin_longhorizon \
  --fps 20 \
  --num-frames 121 \
  --clean_out \
  --overwrite \
  --success-csv /home/omz1504/code/vidar-robotwin/eval_result/ar/single_test/success_rates.csv \
  --success-threshold 0.6
