echo 'RELATION-LEVEL TRAINING START...........'
sh ../relation/train_relation.sh
echo 'RELATION-LEVEL TESTING START...........'
python ../relation/test_relation.py