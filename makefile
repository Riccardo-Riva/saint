dset_id=41540
task=multiclass

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
	--epochs 5 \
	--pretrain \
	--pretrain_epochs 5

saint: 
	${CONDA} activate saint_env