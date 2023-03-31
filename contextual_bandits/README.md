
## Contextual Multi-Armed Bandits Experiments

---
### Training
```
python main.py --expid lbanp-num_latents-8 --cmab_mode train --model lbanp --num_latents 8
```

The config of hyperparameters of each model is saved in `configs/gp`. If training for the first time, evaluation data will be generated and saved in `evalsets/gp`. Model weights and logs are saved in `results/gp/{model}/{expid}`.

### Evaluation
```
python main.py --expid lbanp-num_latents-8 --cmab_mode eval --cmab_wheel_delta 0.7 --model lbanp --num_latents 8 
python main.py --expid lbanp-num_latents-8 --cmab_mode eval --cmab_wheel_delta 0.9 --model lbanp --num_latents 8
python main.py --expid lbanp-num_latents-8 --cmab_mode eval --cmab_wheel_delta 0.95 --model lbanp --num_latents 8
python main.py --expid lbanp-num_latents-8 --cmab_mode eval --cmab_wheel_delta 0.99 --model lbanp --num_latents 8
python main.py --expid lbanp-num_latents-8 --cmab_mode eval --cmab_wheel_delta 0.995 --model lbanp --num_latents 8
```

Note that `{expid}` must match between training and evaluation since the model will load weights from `results/gp/{model}/{expid}` to evaluate.
