export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1

nohup python -u driver/TrainTest.py  --config_file config.rst.eduseg.nocnn  > log 2>&1 &
tail -f log
