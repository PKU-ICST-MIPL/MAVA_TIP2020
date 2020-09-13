echo 'LOCAL-LEVEL TRAINING START...........'
sh ../local/train_local.sh
echo 'LOCAL-LEVEL TESTING START...........'
python ../local/test_local.py