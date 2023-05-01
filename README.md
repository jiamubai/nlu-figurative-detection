# nlu-figurative-detection

## File Outline:
### python and notebook execs:
- FL_pooler: pooler model with 3 bertweet output and a linear layer
- FL_pooler_with_f1: added f1 eval metric to FL_pooler
- FL_train: original model for multi head VAD regression task
- FLtest: notebook 
### slurm job files:
- clf_train: train the FL_pooler
- single_reg_train: train the single VAD regression task
- train_regressor: train newclf with vua dataset
- train_regresssor_bl: train newclf_bl with vua dataset
### csv files:
- sarcasm_test/train: twiiter sarcasm file with prev and next sentence tags removed
- train/test_with_vua: sarcasm dataset combined with vua metaphor data set for the model

*date*: 2023-05-01

