


20200329:nohup python -u imagenet_arch_search.py --path test --gpu 0,1,2,3,4,5,6,7 --n_worker 8 --train_batch_size 100 100 100 100 100 \
--test_batch_size 100 100 100 100 100 --valid_size 100 100 100 100 100  --arch_algo rl --loss_weight 1.0 1.0 1.0 1.0 1.0 \
--dataset wm_data>test/logs/nas_log_5T_20200329.log 2>&1 &
20200324:nohup python -u imagenet_arch_search.py --path test --gpu 0,1,2,3,4,5,6,7 --n_worker 8 --train_batch_size 128 128 128 128 128 \
--test_batch_size 128 128 128 128 128 --valid_size 128 128 128 128 128  --arch_algo rl --loss_weight 1.0 1.0 1.0 1.0 1.0 \
--dataset wm_data>test/logs/nas_log_5T_20200324.log 2>&1 &
20200316:nohup python -u imagenet_arch_search.py --path test --gpu 0,1,2,3,4,5,6,7 --n_worker 8 --train_batch_size 128 128 128 128 128 \
--test_batch_size 128 128 128 128 128 --valid_size 128 128 128 128 128  --arch_algo rl --loss_weight 1.0 1.0 1.0 1.0 1.0 \
--dataset wm_data>test/logs/nas_log_5T_20200316.log 2>&1 &
20200308:nohup python -u imagenet_arch_search.py --path test --gpu 0,1,2,3,4,5,6,7 --n_worker 8 --train_batch_size 128 128 128 128 \
--test_batch_size 128 128 128 128 --valid_size 128 128 128 128  --arch_algo rl --loss_weight 1.0 1.0 1.0 1.0 \
--dataset wm_data>test/logs/nas_log_4T_20200308.log 2>&1 &
20200304:nohup python -u imagenet_arch_search.py --path test --gpu 0,1,2,3,4,5,6,7 --n_worker 8 --train_batch_size 128 128 128 128 \
--test_batch_size 128 128 128 128 --valid_size 128 128 128 128 --target_hardware gpu8 --arch_algo rl --loss_weight 1.0 1.0 1.0 1.0 \
--dataset wm_data>test/logs/nas_log_4T_20200304.log 2>&1 &


五任务训练：nohup python -u imagenet_arch_search.py --path test --gpu 0,1,2,3,4,5,6,7 --n_worker 8 --train_batch_size 128 128 128 128 128 \
--test_batch_size 128 128 128 128 128 --valid_size 128 128 128 128 128 --target_hardware gpu8 --arch_algo rl --loss_weight 1.0 1.0 1.0 1.0 1.0 \
--dataset wm_data>test/logs/nas_log.log 2>&1 &
五任务调试：python imagenet_arch_search.py --path test --gpu 0,1,2,3,4,5,6,7 --n_worker 8 --train_batch_size 128 128 128 128 128 \
--test_batch_size 128 128 128 128 128 --valid_size 128 128 128 128 128 --target_hardware gpu8 --arch_algo rl --loss_weight 1.0 1.0 1.0 1.0 1.0 \
--dataset wm_data

4:nohup python -u imagenet_arch_search.py --path test --gpu 0,1,2,3,4,5,6,7 --n_worker 8 --train_batch_size 128 128 128 128 \
--test_batch_size 128 128 128 128 --valid_size 128 128 128 128 --target_hardware gpu8 --arch_algo rl --loss_weight 1.0 1.0 1.0 1.0 \
--dataset wm_data>test/logs/nas_log.log 2>&1 &

nohup python -u imagenet_arch_search.py --path test --gpu 0,1,2,3,4,5,6,7 --n_worker 8 --train_batch_size 128 128 128 128 \
--test_batch_size 128 128 128 128 --valid_size 128 128 128 128 --target_hardware gpu8 --arch_algo rl --loss_weight 1.0 1.0 1.0 1.0 \
--dataset wm_data>test/logs/nas_log_4T_20200304.log 2>&1 &


5任务不限定gpu8：nohup python -u imagenet_arch_search.py --path test --gpu 0,1,2,3,4,5,6,7 --n_worker 8 --train_batch_size 128 128 128 128 128 \
--test_batch_size 128 128 128 128 128 --valid_size 128 128 128 128 128 --arch_algo rl --loss_weight 1.0 1.0 1.0 1.0 1.0 \
--dataset wm_data>test/logs/nas_log.log 2>&1 &



python imagenet_arch_search.py --path test --gpu 0,1,2,3,4,5,6,7 --n_worker 8 --train_batch_size 100 100 100 100 \
--test_batch_size 100 100 100 100 --valid_size 100 100 100 100  --arch_algo rl --loss_weight 1.0 1.0 1.0 1.0 \
--dataset wm_data