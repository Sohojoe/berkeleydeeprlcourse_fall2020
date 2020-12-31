

### 2020-12-31

Got hw4 q2 running (note the longer train steps)
``` bash
python hw4/cs285/scripts/run_hw4_mb.py --exp_name q2_hopper_singleiteration --env_name marathon-hopper-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 --num_agent_train_steps_per_iter 500 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10
```

Got hw4 q1 running
``` bash
python hw4/cs285/scripts/run_hw4_mb.py --exp_name q1_hopper_n500_arch1x32 --env_name marathon-hopper-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 1 --size 32 --scalar_log_freq -1 --video_log_freq -1

python hw4/cs285/scripts/run_hw4_mb.py --exp_name q1_hopper_n5_arch2x250 --env_name marathon-hopper-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 5 --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1 

python hw4/cs285/scripts/run_hw4_mb.py --exp_name q1_hopper_n500_arch2x250 --env_name marathon-hopper-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 2 --size 250  --scalar_log_freq -1 --video_log_freq -1
```

Note: updated to python 3.8, but I dont think this is needed;

create conda env with marathon envs
* copy envs
* copy modified gym-unity


create repro from commit 4808ac3


