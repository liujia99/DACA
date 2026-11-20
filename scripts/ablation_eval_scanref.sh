# SCANREF

# skill parameter

CUDA_VISIBLE_DEVICES=5,6,7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste true --mge false --ka_mge true --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/32B-daca_nocore-mg_ste+ka_mge --result_dir 3dagent_result/SCANREF/32B-daca_nocore-mg_ste+ka_mge

CUDA_VISIBLE_DEVICES=1,2,3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste true --mge false --ka_mge true --skill_name 'SKILL_DACA_nophase.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/32B-daca_nophase-mg_ste+ka_mge --result_dir 3dagent_result/SCANREF/32B-daca_nophase-mg_ste+ka_mge

CUDA_VISIBLE_DEVICES=5,6,7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste true --mge false --ka_mge true --skill_name 'SKILL_TAB.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/32B-tab_mg_ste+ka_mge --result_dir 3dagent_result/SCANREF/32B-tab_mg_ste+ka_mge

CUDA_VISIBLE_DEVICES=5,6,7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste true --mge false --ka_mge true --skill_name 'SKILL_DACA_nophase_v1.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/32B-daca_nophase_v1-mg_ste+ka_mge --result_dir 3dagent_result/SCANREF/32B-daca_nophase_v1-mg_ste+ka_mge

# point filter parameter
CUDA_VISIBLE_DEVICES=1,2,3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste true --mge false --ka_mge true --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/32B-daca_nocore-mg_ste+ka_mge-SOR --result_dir 3dagent_result/SCANREF/32B-daca_nocore-mg_ste+ka_mge-SOR

CUDA_VISIBLE_DEVICES=5,6,7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste true --mge false --ka_mge true --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/32B-daca_nocore-mg_ste+ka_mge-Raw --result_dir 3dagent_result/SCANREF/32B-daca_nocore-mg_ste+ka_mge-Raw


# voxel size parameter
CUDA_VISIBLE_DEVICES=1,2,3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste true --mge false --ka_mge true --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/32B-daca_nocore-mg_ste+ka_mge-R01 --result_dir 3dagent_result/SCANREF/32B-daca_nocore-mg_ste+ka_mge-R01

CUDA_VISIBLE_DEVICES=5,6,7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste true --mge false --ka_mge true --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/32B-daca_nocore-mg_ste+ka_mge-R04 --result_dir 3dagent_result/SCANREF/32B-daca_nocore-mg_ste+ka_mge-R04


# extension frame parameter
CUDA_VISIBLE_DEVICES=1,2,3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste true --mge false --ka_mge true --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/32B-daca_nocore-mg_ste+ka_mge-K8 --result_dir 3dagent_result/SCANREF/32B-daca_nocore-mg_ste+ka_mge-K8

CUDA_VISIBLE_DEVICES=5,6,7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste true --mge false --ka_mge true --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/32B-daca_nocore-mg_ste+ka_mge-K32 --result_dir 3dagent_result/SCANREF/32B-daca_nocore-mg_ste+ka_mge-K32

# component parameter
CUDA_VISIBLE_DEVICES=5,6,7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste true --mge false --ka_mge false --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/32B-daca_nocore-mg_ste --result_dir 3dagent_result/SCANREF/32B-daca_nocore-mg_ste

CUDA_VISIBLE_DEVICES=5,6,7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste false --mge false --ka_mge true --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/32B-daca_nocore-ka_mge --result_dir 3dagent_result/SCANREF/32B-daca_nocore-ka_mge

CUDA_VISIBLE_DEVICES=5,6,7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste false --mge false --ka_mge false --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/32B-daca_nocore --result_dir 3dagent_result/SCANREF/32B-daca_nocore

# vlm backbone parameter
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-8B-Instruct --ste false --mg_ste true --mge false --ka_mge true --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/8B-daca_nocore-mg_ste+ka_mge --result_dir 3dagent_result/SCANREF/8B-daca_nocore-mg_ste+ka_mge

CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-4B-Instruct --ste false --mg_ste true --mge false --ka_mge true --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/4B-daca_nocore-mg_ste+ka_mge --result_dir 3dagent_result/SCANREF/4B-daca_nocore-mg_ste+ka_mge

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-2B-Instruct --ste false --mg_ste true --mge false --ka_mge true --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'SCANREF' --cache_dir 3dagent_workspace/SCANREF/2B-daca_nocore-mg_ste+ka_mge --result_dir 3dagent_result/SCANREF/2B-daca_nocore-mg_ste+ka_mge





