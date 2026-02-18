Структура проекта


```
text-to-speech-hw3/
├── README.md
├── requirements.txt
├── setup.py
├── synthesize.py
├── demo.ipynb
├── src/
│   ├── __init__.py
│   ├── configs/
│   │   ├── __init__.py
│   │   ├── config.yaml
│   │   └── mel_config.yaml
│   ├── models/
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   └── hifigan.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base_dataset.py
│   │   ├── ruslan_dataset.py
│   │   └── custom_dir_dataset.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── losses.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── audio_utils.py
│   │   ├── mel_processing.py
│   │   └── utils.py
│   └── inference/
│       ├── __init__.py
│       └── inference.py
├── scripts/
│   ├── download_dataset.sh
│   ├── download_checkpoints.sh
│   ├── preprocess.py
│   └── train.py
├── tests/
│   └── test_models.py
├── checkpoints/
│   └── README.md
└── logs/
    └── README.md

```
