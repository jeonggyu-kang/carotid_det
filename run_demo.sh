#!/bin/bash

clear 
python3 demo.py --root ./carotid_pp --json_test ./gTruth_pp_test.json \
                --log_dir ./vis \
                --pretrained work_dir/ckpt/best.pth 