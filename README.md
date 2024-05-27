# KBQA
Question Answering over Knowledge Base

```python
from kbqa.utils.data_utils import get_entity_corpus
from kbqa.utils.data_types import Doc, Span, Entity
from kbqa.entity_linking.inference.blink import BlinkEL, BlinkELConfig

entity_corpus = get_entity_corpus(ENTITY_CORPUS_PATH)
config = BlinkELConfig(
    model_name=MODEL_NAME,
)
el_model = BlinkEL(config, entity_corpus)

docs = [
    Doc(text="What year did Michael Jordan win his first NBA championship?")
]

pred_docs = el_model(docs)
print(pred_docs)
```