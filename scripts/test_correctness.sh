source ./scripts/config.sh

# ssh $REDIS_SERVER "kill -9 \$(pgrep eloqvec) 2>/dev/null"

# ssh $REDIS_SERVER "cd ~/code/eloqvec && ./build/eloqvec -ip $REDIS_SERVER --core_number=8" &

# sleep 1

python ./scripts/test_recall.py --test_correctness=True --add_num $ADD_NUM --test_num $TEST_NUM --k $K --redis_host=$REDIS_SERVER --redis_port=$REDIS_PORT
# python ./scripts/test_recall.py --test_eloqvec=True --add_num $ADD_NUM --test_num $TEST_NUM --k $K --redis_host=$REDIS_SERVER --redis_port=$REDIS_PORT
