# GLiNER SpaCy Wrapper

## Introduction
This project is a wrapper for integrating [GLiNER](https://github.com/urchade/GLiNER), a Named Entity Recognition (NER) model, with the SpaCy Natural Language Processing (NLP) library. GLiNER, which stands for Generalized Language INdependent Entity Recognition, is an advanced model for recognizing entities in text. The SpaCy wrapper enables easy integration and use of GLiNER within the SpaCy environment, enhancing NER capabilities with GLiNER's advanced features.

**For GliNER to work properly, you need to use a Python version 3.7-3.10**

## Features
- Integrates GLiNER with SpaCy for advanced NER tasks.
- Customizable chunk size for processing large texts.
- Support for specific entity labels like 'person' and 'organization'.
- Configurable output style for entity recognition results.

## Installation
To install this library, install it via pip:

```bash
pip install gliner-spacy
```

## Usage
To use this wrapper in your SpaCy pipeline, follow these steps:

1. Import SpaCy.
2. Create a SpaCy `Language` instance.
3. Add the `gliner_spacy` component to the SpaCy pipeline.
4. Process text using the pipeline.

Example code:

```python
import spacy

nlp = spacy.blank("en")
nlp.add_pipe("gliner_spacy")
text = "This is a text about Bill Gates and Microsoft."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
```

### Expected Output

```
Bill Gates person
Microsoft organization
```

## Example with Custom Configs

```python
import spacy

custom_spacy_config = { "gliner_model": "urchade/gliner_multi",
                            "chunk_size": 250,
                            "labels": ["people","company","punctuation"],
                            "style": "ent"}
nlp = spacy.blank("en")
nlp.add_pipe("gliner_spacy", config=custom_spacy_config)

text = "This is a text about Bill Gates and Microsoft."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_, ent._.score)

#Output
# Bill Gates person 0.9967108964920044
# Tom person 0.9612326622009277
# Microsoft organization 0.9966742992401123    
```

## Configuration
The default configuration of the wrapper can be modified according to your requirements. The configurable parameters are:
- `gliner_model`: The GLiNER model to be used.
- `chunk_size`: Size of the text chunk to be processed at once.
- `labels`: The entity labels to be recognized.
- `style`: The style of output for the entities (either 'ent' or 'span').
- `threshold`: The threshold of the GliNER model (controls the degree to which a hit is considered an entity)
- `map_location`: The device on which to run the model: `cpu` or `cuda`

## Contributing
Contributions to this project are welcome. Please ensure that your code adheres to the project's coding standards and include tests for new features.
