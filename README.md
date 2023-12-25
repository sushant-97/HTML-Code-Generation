# HTML-Code-Generation
Used 4-bit precision model using bitsandbytes library and further used LoRA technique to finetune LLAMA2 for HTML code generation.
Training report can be found here
https://api.wandb.ai/links/kameng/yo4tnfhg

# LLAMA Fine-Tuning for HTML Code Generation

Fine-tune the LLAMA model for code generation tasks using the Python script. The script incorporates a range of libraries and functionalities from the Transformers library, custom model and dataset preprocessors, loss functions, and Weights & Biases (wandb) for experiment tracking.

## Features

- LLAMA model adaptation for code generation.
- Custom data preprocessing and model management.
- Loss function and GPU acceleration.
- Model validation and training functions.
- Weights & Biases integration for tracking and logging.

## Requirements

- Python 3.x
- PyTorch
- Transformers
- bitsandbytes
- wandb
- tqdm

Ensure all dependencies are installed using the requirements file or individually.

## Configuration and Usage

1. **Setting Up**: Define your configuration in the `setup_config()` method. Adjust parameters like learning rate, epochs, and batch size according to your needs.

2. **Preprocessing Data**: Utilize the `DatasetPreprocessor` to handle your data needs. Ensure you have a suitable dataset and adjust the `dataset_id` accordingly.

3. **Model Initialization**: The script uses the "meta-llama/Llama-2-7b-hf" model. Initialize and manage the model using the `ModelManager`.

4. **Training**: Run the `train_model` function, providing it with the model, data loaders, optimizer, scheduler, and config.

5. **Validation and Evaluation**: Validate model performance using the `validate` function and further analyze using predictions and rouge scores.

6. **Model Saving and Logging**: The script saves the model and tokenizer using `save_model` and logs the training process to wandb.

## Customizing the Script

- **Model and Data**: Change the `model_id` and `dataset_id` to use different models or datasets.
- **ModelManager and DatasetPreprocessor**: Adjust these classes as needed for different model configurations or data preprocessing steps.

## Running the Script

To execute the script, run:

```bash
python [script_name].py
