import datetime
import os

import wandb

from insec.utils import gpt_cost, model_label
from insec import AdversarialTokens

class Logger:
    def __init__(self, output_dir, model_dir, dataset, experiment_name, loss_calculator):
        # Set timestamp and loss calculator instance
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.loss_calculator = loss_calculator
        
        # Initialize wandb to fix the "You must call wandb.init()" error
        wandb.init(
            project="INSEC_Adversarial_Attack",
            name=f"{experiment_name}_{dataset}_{timestamp}",
            config={
                "model_dir": model_dir,
                "dataset": dataset,
                "experiment_name": experiment_name,
                "output_dir": output_dir
            }
        )

    def log(self, candidate_adv_tokens: AdversarialTokens, candidate_loss, best_loss):
        # Print current sample information to console
        print(f"{'samp':<20}" + f"{str(candidate_adv_tokens):<50}" + f"{candidate_loss}")
        self.log_wandb(candidate_loss, best_loss)

    def log_wandb(self, candidate_loss, best_loss):
        # Create the dictionary of metrics for wandb logging
        log_dict = {
            "loss": candidate_loss,
            "prompt_tokens": self.loss_calculator.total_prompt_tokens,
            "completion_tokens": self.loss_calculator.total_completion_tokens,
            "lowest_loss": best_loss
        }
        
        # Estimate cost if using OpenAI API
        if "OpenAI" in self.loss_calculator.model.__class__.__name__:
            log_dict["price"] = gpt_cost(
                self.loss_calculator.total_prompt_tokens,
                self.loss_calculator.total_completion_tokens,
            )
            
        # Send data to the wandb server
        wandb.log(log_dict)