python script/collect_initial_obs.py \
  --config policy/AR/deploy_policy.yml \
  --overrides \
  --task_config hd_clean \
  --seed 1234 \
  --per_task 10 \
  --out_json /home/omz1504/code/vidar/data/test/robotwin_put_object_cabinet_2.json \
  --image_dir /home/omz1504/code/vidar/data/test \
  --tasks put_object_cabinet
