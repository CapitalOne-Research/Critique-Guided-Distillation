To Evaluate instruct or fine-tuned models use `evaluate_chat.sh` and pass the checkpoint location as an argument.

To Evaluate base models without chat templates use `evaluate_base.sh` and pass the checkpoint location as an argument.

To perform validation on multiple checkpoints to identify the best checkpoint on the selected MATH-500 validation set, use `start_validate.sh` and put the checkpoint location in the script along with a desired validation summary file name.