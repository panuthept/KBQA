import os
import json
from tqdm import tqdm
from typing import List
from kbqa.utils.data_types import Doc, Entity, Span
from kbqa.utils.wikidata_mapper import WikidataMapper


class Zelda:
    def __init__(self, path: str, wikidata_mapper: WikidataMapper):
        self.path = path
        self.wikidata_mapper = wikidata_mapper

    def _read_dataset(self, path: str, verbose: bool = False) -> list[Doc]:
        total_spans = 0
        out_of_kb_spans = 0
        docs: List[Doc] = []
        with open(path, "r") as f:
            for line in tqdm(f, desc="Reading dataset", unit=" docs", disable=not verbose):
                data = json.loads(line)
                text = data["text"]

                spans: List[Span] = []
                for (span_start, span_end), wikipedia_id, wikipedia_title in zip(data["index"], data["wikipedia_ids"], data["wikipedia_titles"]):
                    wikipedia_title = wikipedia_title.replace(" ", "_")
                    qcode = self.wikidata_mapper.map_title_to_wikidata_qcode(wikipedia_title)
                    if qcode is None:
                        out_of_kb_spans += 1
                        continue
                    total_spans += 1
                    spans.append(
                        Span(
                            start=span_start,
                            length=span_end - span_start,
                            surface_form=text[span_start:span_end],
                            gold_entity=Entity(id=qcode),
                        )
                    )
                if len(spans) == 0:
                    continue
                docs.append(Doc(text=text, spans=spans))
        if verbose:
            print(f"Total spans: {total_spans}")
            print(f"Out-of-KB spans: {out_of_kb_spans} / {total_spans} ({round(out_of_kb_spans / total_spans * 100, 2)}%)")
        return docs

    def get_train_docs(self, **kwargs) -> list[Doc]:
        return self._read_dataset(os.path.join(self.path, "train_data/zelda_train.jsonl"), **kwargs)

    def get_aida_docs(self, **kwargs) -> list[Doc]:
        return self._read_dataset(os.path.join(self.path, "test_data/jsonl/test_aida-b.jsonl"), **kwargs)
    
    def get_cweb_docs(self, **kwargs) -> list[Doc]:
        return self._read_dataset(os.path.join(self.path, "test_data/jsonl/test_cweb.jsonl"), **kwargs)
    
    def get_reddit_comments_docs(self, **kwargs) -> list[Doc]:
        return self._read_dataset(os.path.join(self.path, "test_data/jsonl/test_reddit-comments.jsonl"), **kwargs)
    
    def get_reddit_posts_docs(self, **kwargs) -> list[Doc]:
        return self._read_dataset(os.path.join(self.path, "test_data/jsonl/test_reddit-posts.jsonl"), **kwargs)
    
    def get_shadowlinks_shadow_docs(self, **kwargs) -> list[Doc]:
        return self._read_dataset(os.path.join(self.path, "test_data/jsonl/test_shadowlinks-shadow.jsonl"), **kwargs)
    
    def get_shadowlinks_tail_docs(self, **kwargs) -> list[Doc]:
        return self._read_dataset(os.path.join(self.path, "test_data/jsonl/test_shadowlinks-tail.jsonl"), **kwargs)
    
    def get_shadowlinks_top_docs(self, **kwargs) -> list[Doc]:
        return self._read_dataset(os.path.join(self.path, "test_data/jsonl/test_shadowlinks-top.jsonl"), **kwargs)
    
    def get_tweeki_docs(self, **kwargs) -> list[Doc]:
        return self._read_dataset(os.path.join(self.path, "test_data/jsonl/test_tweeki.jsonl"), **kwargs)
    
    def get_wiki_docs(self, **kwargs) -> list[Doc]:
        return self._read_dataset(os.path.join(self.path, "test_data/jsonl/test_wned-wiki.jsonl"), **kwargs)
    

if __name__ == "__main__":
    zelda_dataset = Zelda(
        path="./data/datasets/zelda", 
        wikidata_mapper=WikidataMapper("./data/wikidata_mapper"),
    )

    train_docs = zelda_dataset.get_train_docs(verbose=True)
    print(f"Train docs: {len(train_docs)}")

    aida_docs = zelda_dataset.get_aida_docs(verbose=True)
    print(f"AIDA docs: {len(aida_docs)}")

    cweb_docs = zelda_dataset.get_cweb_docs(verbose=True)
    print(f"CWEB docs: {len(cweb_docs)}")

    reddit_comments_docs = zelda_dataset.get_reddit_comments_docs(verbose=True)
    print(f"Reddit comments docs: {len(reddit_comments_docs)}")

    reddit_posts_docs = zelda_dataset.get_reddit_posts_docs(verbose=True)
    print(f"Reddit posts docs: {len(reddit_posts_docs)}")

    shadowlinks_shadow_docs = zelda_dataset.get_shadowlinks_shadow_docs(verbose=True)
    print(f"Shadowlinks shadow docs: {len(shadowlinks_shadow_docs)}")

    shadowlinks_tail_docs = zelda_dataset.get_shadowlinks_tail_docs(verbose=True)
    print(f"Shadowlinks tail docs: {len(shadowlinks_tail_docs)}")

    shadowlinks_top_docs = zelda_dataset.get_shadowlinks_top_docs(verbose=True)
    print(f"Shadowlinks top docs: {len(shadowlinks_top_docs)}")

    tweeki_docs = zelda_dataset.get_tweeki_docs(verbose=True)
    print(f"Tweeki docs: {len(tweeki_docs)}")

    wiki_docs = zelda_dataset.get_wiki_docs(verbose=True)
    print(f"Wiki docs: {len(wiki_docs)}")