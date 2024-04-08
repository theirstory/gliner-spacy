from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gliner-spacy',  # Your package name
    version='0.0.3',  # Initial version
    author='William J. B. Mattingly',  # Your name
    description='A SpaCy wrapper for the GLiNER model for enhanced Named Entity Recognition capabilities',  # Short description
    long_description=long_description,
    long_description_content_type='text/markdown',  # Ensures correct rendering on PyPI
    url='https://github.com/theirstory/gliner-spacy',  # Your repository URL
    packages=find_packages(),
    install_requires=[
        'spacy',
        'gliner',  # Assuming 'gliner' is the correct package name on PyPI
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
)
