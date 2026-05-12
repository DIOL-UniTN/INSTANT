# Hydra template

### Pick your config file:
```bash
cp conf/main_local.yaml conf/main.yaml
```

### Run an experiment:
```bash
python src/main.py --multirun exp=myexp seed=1,2,3,4
```
# fff_archs

### Train IFFF model
```bash
python src/main.py exp=train_infa model=infafff
```

### Train CNN model on CIFAR-100 with custom settings
```bash
python src/main.py model=cnn exp=train_cnn epochs=5 exp.scale=32 exp.width=256 loader=cifar100
```

### Train CNN+FFF model on CIFAR-100 with custom settings
```bash
 python src/main.py epochs=5 model.scale=32 model.leaf_width=64 model.depth=2 exp=train_cnn_fff model=cnn_fff loader=cifar100
```

### Train CNN+IFFF model w/o scheduler on CIFAR-100 with custom settings
```bash
 python src/main.py epochs=5 model.scale=32 model.leaf_width=64 model.depth=2 exp=train_cnn_ifff exp.sched=0 model=cnn_ifff loader=cifar100 
```

### Train CNN+IFFF model with scheduler on CIFAR-100 with custom settings
```bash
 python src/main.py epochs=5 model.scale=32 model.leaf_width=64 model.depth=2 exp=train_cnn_ifff exp.sched=1 model=cnn_ifff loader=cifar100 
```

### Train a model on CIFAR-10 with custom settings
```bash
 python src/main.py epochs=200 sched=cos sched.T_max=200 model=simple_dla exp=train_cifar10 optim=sgd loader=cifar10
```

### Train a resnet18 on tiny imagenet with custom settings
```bash
 python src/main.py epochs=10 model=resnet18 exp=train_tinyimagenet optim=sgd2 loader=tinyimagenet
```
