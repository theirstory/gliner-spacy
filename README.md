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
                            "labels": ["people","company"],
                            "style": "ent"}
nlp = spacy.blank("en")
nlp.add_pipe("gliner_spacy", config=custom_spacy_config)

text = "This is a text about Bill Gates and Microsoft."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_, ent._.score)

#Output
# Bill Gates people 0.9967108964920044
# Microsoft company 0.9966742992401123    
```

## Example with loading onnx model
```python
import spacy

custom_spacy_config = {
    "gliner_model": "onnx-community/gliner_base",
    "chunk_size": 250,
    "labels": ["people", "company"],
    "style": "ent",
    "load_onnx_model": True,
    "onnx_model_file": "onnx/model.onnx",
}
nlp = spacy.blank("en")
nlp.add_pipe("gliner_spacy", config=custom_spacy_config)

text = "This is a text about Bill Gates and Microsoft."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_, ent._.score)

# Output
# Bill Gates people 0.9937531352043152
# Microsoft company 0.994135856628418
```

## Configuration
The default configuration of the wrapper can be modified according to your requirements. The configurable parameters are:
- `gliner_model`: The GLiNER model to be used.
- `chunk_size`: Size of the text chunk to be processed at once.
- `labels`: The entity labels to be recognized.
- `style`: The style of output for the entities (either 'ent' or 'span').
- `threshold`: The threshold of the GliNER model (controls the degree to which a hit is considered an entity)
- `map_location`: The device on which to run the model: `cpu` or `cuda`
- `load_onnx_model`: Whether the `gliner_model` specificied is an ONNX model (False by default)
- `onnx_model_file`: The path to the onnx file in the Huggingface repo. Defaults to `model.onnx`

## Contributing
Contributions to this project are welcome. Please ensure that your code adheres to the project's coding standards and include tests for new features.
