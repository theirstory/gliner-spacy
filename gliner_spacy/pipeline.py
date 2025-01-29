import spacy
from gliner.model import GLiNER
from spacy.language import Language
from spacy.tokens import Span, Doc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

Span.set_extension("score", default=0, force=True)
Span.set_extension("sent_cats", default={}, force=True)
Span.set_extension("raw_scores", default=[], force=True)
Span.set_extension("sent_spans", default=[], force=True)

Doc.set_extension("visualize", default=[], force=True)
Doc.set_extension("graph_data", default=[], force=True)

DEFAULT_SPACY_CONFIG = {
    "gliner_model": "urchade/gliner_base",
    "chunk_size": 250,
    "labels": ["person", "organization", "school"],
    "style": "ent",
    "threshold": .50,
    "map_location": "cpu",
    "load_onnx_model": False,
    "onnx_model_file": "model.onnx",
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
                 threshold: float,
                 map_location: str,
                 load_onnx_model: bool,
                 onnx_model_file: str,
                 ):
        
        self.nlp = nlp
        self.model = GLiNER.from_pretrained(
            gliner_model,
            map_location=map_location,
            load_onnx_model=load_onnx_model, 
            load_tokenizer=load_onnx_model,
            onnx_model_file=onnx_model_file
        )
        self.labels = labels
        self.chunk_size = chunk_size
        self.style = style
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
                    'label': entity['label'],
                    'score': entity['score']
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
                span._.score = ent['score']
                spans.append(span)
        if self.style == "span":
            doc.spans["sc"] = spans
        else:
            doc.ents = spans
        return doc


DEFAULT_SPACYCAT_CONFIG = {
    "cat_data": {"education": ["school"]},
    "style": "span"
}


@Language.factory("gliner_cat",
                  assigns=[],
                  default_config=DEFAULT_SPACYCAT_CONFIG
                  )    
class GlinerCat:
    def __init__(self, nlp: Language, name: str, cat_data: dict, style: str):
        self.nlp = nlp
        self.cat_data = cat_data
        self.style = style

        if "gliner_spacy" not in nlp.analyze_pipes()["summary"]:
            raise ValueError("gliner_cat requires gliner_spacy to be in the pipeline.")
        
        sentence_segmenter_present = any('doc.sents' in component_info['assigns'] for component_name, component_info in nlp.analyze_pipes()["summary"].items())
        
        if not sentence_segmenter_present:
            raise RuntimeError("A sentence-segmenting component is required but not found in the pipeline. "
                               "Please ensure that 'sentencizer' or 'parser' is added to the pipeline before 'gliner_cat'.")


    def __call__(self, doc):
        for sent in doc.sents:
            scores = {label: [] for label in self.cat_data}
            sent_spans = []

            # Choose the span source based on the 'style' configuration
            if self.style == "span":
                span_source = doc.spans.get("sc", [])  # Safely get 'sc' spans or an empty list
            elif self.style == "ent":
                span_source = doc.ents  # Named entities

            # Iterate over chosen span source
            for span in span_source:
                # Check if span is within the current sentence
                if span.start >= sent.start and span.end <= sent.end:
                    # Iterate over categories and labels in configuration
                    for cat, labels in self.cat_data.items():
                        if span.label_ in labels:
                            # Assume spans have a custom attribute 'score', ensure it's set properly
                            scores[cat].append(getattr(span._, 'score', 0))  # Default score is 0 if not present
                    sent_spans.append(span)

            themes = {label: sum(score) for label, score in scores.items()}
            sent._.raw_scores = scores
            sent._.sent_cats = themes
            sent._.sent_spans = sent_spans

        doc._.visualize = lambda **kwargs: self.visualize_cats(doc, **kwargs)
        return doc

    def visualize_cats(self, doc, sent_start=0, sent_end=None, chunk_size=10, fig_w=10, fig_h=10):
        labels = [label for label in self.cat_data]
        doc_sents = list(doc.sents)
        data = [sent._.sent_cats for sent in doc_sents[sent_start:sent_end]]
        graph = []
        
        # Process each sentence's category data for visualization
        for d in data:
            sent = []
            for label, score in d.items():
                num = round(score, 1)
                if num == 0:
                    num = float(0.0)
                sent.append(num)
            graph.append(sent)

        if chunk_size > 1:
            # Chunk the data if the chunk size is specified as more than 1
            if len(graph) % chunk_size != 0:
                extra_rows = chunk_size - (len(graph) % chunk_size)
                graph.extend([[0.0] * len(labels) for _ in range(extra_rows)])
            y_labels = [f"Sentence {i+sent_start}-{i+sent_start+chunk_size}" for i in range(0, len(graph), chunk_size)]
            aggregated_graph = []
            # Aggregate scores by chunk
            for i in range(0, len(graph), chunk_size):
                chunked_data = graph[i:i+chunk_size]
                aggregated_row = np.sum(chunked_data, axis=0)
                aggregated_graph.append(aggregated_row)
            graph = aggregated_graph
        else:
            # Label each sentence if not chunking
            y_labels = [f"Sentence {i+sent_start+1}" for i in range(len(graph))]

        # Create and show the heatmap
        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(graph, annot=True, fmt=".1f", linewidths=.5, cmap="Blues", xticklabels=labels, yticklabels=y_labels)
        plt.title('Category Scores by Sentence')
        plt.xlabel('Categories')
        plt.ylabel('Sentences')
        plt.show()
