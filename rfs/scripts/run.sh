# ======================
# exampler commands on miniImageNet
# ======================

# supervised pre-training
python train_supervised.py --trial pretrain --model_path teacher.pth  --data_root /home/koyejolab/CheXpert/CheXpert-v1.0-small

# # distillation
# # setting '-a 1.0' should give similar performance
# python train_distillation.py -r 0.5 -a 0.5 --path_t teacher.pth --trial born1 --model_path student.pth --data_root /home/koyejolab/CheXpert/CheXpert-v1.0-small

# # evaluation
# python eval_fewshot.py --model_path student.pth --data_root /home/koyejolab/CheXpert/CheXpert-v1.0-small
