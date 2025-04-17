dset_id=31
task=binary

CONDA=mamba

train: train.py
	python train.py \
	--dset_id $(dset_id) \
	--task $(task) \
	--epochs 25

train_robust: train_robust.py
	python train_robust.py \
	--dset_id $(dset_id) \
	--task $(task) \
	--epochs 25

saint: 
	${CONDA} activate saint_env