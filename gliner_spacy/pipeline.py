import spacy
from gliner.model import GLiNER
from spacy.language import Language


DEFAULT_SPACY_CONFIG = {
        "gliner_model": "urchade/gliner_base",
        "chunk_size": 250,
        "labels": ["person", "organization"],
        "style": "ent",
        "threshold": .50
    }


@Language.factory("gliner_spacy",
                  assigns=["doc.ents"],
                  default_config=DEFAULT_SPACY_CONFIG)
class GlinerSpacy:
    def __init__(self,
                 nlp: Language,
                 name: str,
                 gliner_model: str,
                 chunk_size: int,
                 labels: list,
                 style: str,
                 threshold: float
                 ):
        
        self.nlp = nlp
        self.model = GLiNER.from_pretrained(gliner_model)
        self.labels = labels
        self.chunk_size = chunk_size
        self.style = style  # Store style as an instance variable
        self.threshold = threshold

    def __call__(self, doc):
        # Tokenize the text
        chunks = []
        start = 0
        text = doc.text
        while start < len(text):
            end = start + self.chunk_size if start + self.chunk_size < len(text) else len(text)
            # Ensure the chunk ends at a complete word
            while end < len(text) and text[end] not in [' ', '\n']:
                end += 1
            chunks.append(text[start:end])
            start = end

        # Process each chunk and adjust entity indices
        all_entities = []
        offset = 0
        for chunk in chunks:
            if self.style == "span":
                chunk_entities = self.model.predict_entities(chunk, self.labels,
                                                             flat_ner=False,
                                                             threshold=self.threshold)
            else:
                chunk_entities = self.model.predict_entities(chunk, self.labels,
                                                             flat_ner=True,
                                                             threshold=self.threshold)
            for entity in chunk_entities:
                all_entities.append({
                    'start': offset + entity['start'],
                    'end': offset + entity['end'],
                    'label': entity['label']
                })
            offset += len(chunk)

        # Create new spans for the entities and add them to the doc
        doc = self._create_entity_spans(doc, all_entities)

        return doc

    def _create_entity_spans(self, doc, all_entities):
        spans = []
        for ent in all_entities:
            span = doc.char_span(ent['start'], ent['end'], label=ent['label'])
            if span:  # Only add span if it is valid
                spans.append(span)
        if self.style == "span":
            doc.spans["sc"] = spans
        else:
            doc.ents = spans
        return doc


