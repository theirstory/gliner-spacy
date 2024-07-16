from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Import the version
directory = os.path.dirname(__file__)
version_path = os.path.join(directory, 'gliner_spacy', 'version.py')
about = {}
with open(version_path) as f:
    exec(f.read(), about)

setup(
    name='gliner-spacy',
    version=about['__version__'],
    author='William J. B. Mattingly',
    description='A SpaCy wrapper for the GLiNER model for enhanced Named Entity Recognition capabilities',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/theirstory/gliner-spacy',
    packages=find_packages(),
    entry_points={
      "spacy_factories": ["gliner_spacy = gliner_spacy.pipeline:GlinerSpacy", "gliner_cat = gliner_spacy.pipeline:GlinerCat"],
    },
    install_requires=[
        'spacy>=3.0.0',
        'gliner>=0.2.0',
        'seaborn',
        'matplotlib',
        'numpy'
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
