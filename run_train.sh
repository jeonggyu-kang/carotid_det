#!/bin/bash
clear

python3 main.py --root ./carotid_pp --json_train ./gTruth_pp_v3.json \
                --json_test ./gTruth_pp_test.json \
                --max_epoch 200 \
                --batch_size 8 \
                --log_dir ./work_dir/epoch200-hardsample



