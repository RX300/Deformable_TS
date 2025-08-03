# 常用命令

python train.py -s "../../dataset/D_Nerf data/data/hook" -m output/hook --eval --is_blender

## 腾讯云

### deformgs

python train.py -s /workspace/dataset/vrig-3dprinter -m output/vrig-3dprinter  --eval --iterations 20000 --use_autoencoder --latent_dim 32

python render.py -m output/vrig-3dprinter_base --mode render

python metrics.py -m output/vrig-3dprinter_base

### 4dgs

bash colmap.sh /workspace/dataset/vrig-3dprinter hypernerf

python scripts/downsample_point.py /workspace/dataset/vrig-3dprinter/colmap/dense/workspace/fused.ply /workspace/dataset/vrig-3dprinter/points3D_downsample2.ply

python train.py -s  /workspace/dataset/vrig-3dprinter/ --port 6017 --expname "hypernerf/3dprinter" --configs arguments/hypernerf/3dprinter.py

## autodl

### deformgs

python train.py -s /root/autodl-tmp/vrig-chicken -m output/vrig-chicken_base --eval --iterations 20000

python render.py -m output/vrig-chicken_base

### 4dgs

bash colmap.sh /root/autodl-tmp/vrig-3dprinter hypernerf

python scripts/downsample_point.py /root/autodl-tmp/vrig-3dprinter/colmap/dense/workspace/fused.ply /root/autodl-tmp/vrig-3dprinter/points3D_downsample2.ply

python train.py -s  /root/autodl-tmp/vrig-3dprinter/ --port 6017 --expname "hypernerf/3dprinter" --configs arguments/hypernerf/3dprinter.py

python train.py -s  /root/autodl-tmp/coffee_martini/ --port 6017 --expname "dynerf/coffee_martini" --configs arguments/dynerf/coffee_martini.py

python render.py --model_path "output/hypernerf/3dprinter/"  --skip_train --configs arguments/hypernerf/3dprinter.py

python render.py --model_path "output/dynerf/coffee_martini/"  --skip_train --configs arguments/dynerf/coffee_martini.py

python metrics.py --model_path "output/hypernerf/3dprinter/"

python metrics.py --model_path "output/dynerf/coffee_martini/"
