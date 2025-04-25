source ./scripts/config.sh

python ./scripts/test_recall.py --k 100 --add_num $ADD_NUM --test_eloqvec=True --redis_host=$REDIS_SERVER --redis_port=$REDIS_PORT