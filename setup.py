from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gliner-spacy',
    version='0.0.4', 
    author='William J. B. Mattingly',
    description='A SpaCy wrapper for the GLiNER model for enhanced Named Entity Recognition capabilities',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/theirstory/gliner-spacy',
    packages=find_packages(),
    entry_points={
      "spacy_factories": ["gliner_spacy = gliner_spacy.pipeline:GlinerSpacy"],
    },
    install_requires=[
        'spacy',
        'gliner',
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
    python_requires='>=3.7,<=3.10',
)
