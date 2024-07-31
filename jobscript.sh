#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-153 -p alvis
#SBATCH -t 03:00:00
#SBATCH --gpus-per-node=A100:1

module purge && \
module load Python/3.11.5-GCCcore-13.2.0 && \
if [[ -d venv ]]
then
	source venv/bin/activate
else
	#virtualenv --system-site-packages venv && \
	python -m venv venv && \
	source venv/bin/activate && \
	pip install -r alvis-requirements.txt
fi || exit

export SEED=1
export NUM_ROUNDS=50
export DATASET_ROOT=/mimer/NOBACKUP/Datasets/ZOD/v20230313/
export DATASET_VERSION=full
export DATASET_MAX_IMAGES=5000
export BATCH_SIZE=4
export MODEL_FN=retina_coco_xs
export MIXED_PRECISION=False
export RECREATE_MODEL=False
export SHARE_MODEL=False
export NUM_CLIENTS=10
export LOCAL_EPOCHS=1
export STEPS_PER_LOCAL_EPOCH=None
export IMPORTANCE_FN=random_based_importance
export SIMULATION_ID=${SEED}_${NUM_ROUNDS}_${MODEL_FN}_${NUM_CLIENTS}_${LOCAL_EPOCHS}_${IMPORTANCE_FN}
export OBJECT_CLASSES='Vehicle Pedestrian PoleObject Animal TrafficSign'
python main.py
