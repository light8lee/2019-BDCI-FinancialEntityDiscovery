#!/bin/bash
data_dir=inputs/rc_fullv3
mkdir -p ${data_dir}
python create_squad_data.py ${data_dir}