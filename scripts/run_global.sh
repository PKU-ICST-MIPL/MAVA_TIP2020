echo 'GLOBAL-LEVEL TRAINING START...........'
sh ../global/train_global.sh
echo 'GLOBAL-LEVEL TESTING START...........'
python ../global/test_global.py