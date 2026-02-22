"""Конфигурация обучения"""

training_config = {
    'batch_size': 8,
    'accumulation_steps': 2,
    'num_epochs': 25,
    'lr_generator': 2e-4,
    'lr_discriminator': 2e-4,
    'beta1': 0.8,
    'beta2': 0.99,
    'weight_decay': 0.01,
    'lambda_fm': 2.0,
    'lambda_mel': 45.0,
    'gradient_clip': 1.0,
    'save_interval': 5,
}

data_config = {
    'sample_rate': 22050,
    'segment_size': 8192,
    'augment': True,
    'train_split': 0.9,
}