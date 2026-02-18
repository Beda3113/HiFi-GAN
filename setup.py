from setuptools import setup, find_packages

setup(
    name="hifigan-tts",
    version="0.1.0",
    description="HiFi-GAN implementation for TTS homework",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchaudio>=0.10.0",
        "numpy>=1.19.5",
        "librosa>=0.9.0",
        "scipy>=1.7.0",
    ],
)