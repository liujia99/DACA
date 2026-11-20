
# NR3D
CUDA_VISIBLE_DEVICES=5,6,7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste true --mge false --ka_mge true --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'NR3D' --cache_dir 3dagent_workspace/NR3D/32B-daca_nocore_mg_ste+ka_mge --result_dir 3dagent_result/NR3D/32B-daca_nocore_mg_ste+ka_mge

# component parameter
CUDA_VISIBLE_DEVICES=5,6,7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste true --mge false --ka_mge false --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'NR3D' --cache_dir 3dagent_workspace/NR3D/32B-daca_nocore-mg_ste --result_dir 3dagent_result/NR3D/32B-daca_nocore-mg_ste

CUDA_VISIBLE_DEVICES=5,6,7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste false --mge false --ka_mge true --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'NR3D' --cache_dir 3dagent_workspace/NR3D/32B-daca_nocore-ka_mge --result_dir 3dagent_result/NR3D/32B-daca_nocore-ka_mge

CUDA_VISIBLE_DEVICES=5,6,7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python eval.py --client_id  /data/users/jmliu/model/Qwen3-VL-32B-Instruct --ste false --mg_ste false --mge false --ka_mge false --skill_name 'SKILL_DACA_nocore.md' --dataset_name 'NR3D' --cache_dir 3dagent_workspace/NR3D/32B-daca_nocore --result_dir 3dagent_result/NR3D/32B-daca_nocore



