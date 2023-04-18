import os
import torch



def save_model(args, model, tokenizer, config, global_step,tr_loss, logger):
    """
    Save the model, tokenizer, and configuration to the specified directory.
    """
    output_dir = os.path.join(args["output_dir"], "checkpoint-{}".format(global_step))

    # Create the directory if it doesn't already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Saving model to %s", output_dir)
    
    # Save the model, tokenizer, and configuration
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    torch.save(tr_loss, os.path.join(output_dir, "training_loss.pt"))
    torch.save(global_step, os.path.join(output_dir, "global_step.pt")) # Salva global_step

    # Save other hyperparameters used during training
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    logger.info("Saving completed.")

