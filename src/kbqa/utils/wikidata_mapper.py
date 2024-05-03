from typing import Optional, Set, Mapping
from kbqa.utils.refined_utils.lmdb_wrapper import LmdbImmutableDict
from kbqa.utils.refined_utils.loaders import load_disambiguation_qcodes


class WikidataMapper:
    def __init__(self, path: str):
        self.redirects: Mapping[str, str] = LmdbImmutableDict(f"{path}/redirects.lmdb")
        self.wiki_to_qcode: Mapping[str, str] = LmdbImmutableDict(f"{path}/wiki_to_qcode.lmdb")
        self.disambiguation_qcodes: Set[str] = load_disambiguation_qcodes(f"{path}/disambiguation_qcodes.txt")
        self.qcode_to_label: Mapping[str, str] = LmdbImmutableDict(f"{path}/qcode_to_label.lmdb")

    def map_title_to_wikidata_qcode(self, wiki_title: str) -> Optional[str]:
        wiki_title = (
            wiki_title.replace("&lt;", "<").replace("&gt;", ">").replace("&le;", "≤").replace("&ge;", "≥")
        )
        if len(wiki_title) == 0:
            return None
        wiki_title = wiki_title[0].upper() + wiki_title[1:]
        if wiki_title in self.redirects:
            wiki_title = self.redirects[wiki_title]
        if wiki_title in self.wiki_to_qcode:
            qcode = self.wiki_to_qcode[wiki_title]
            return qcode
        return None

    def wikidata_qcode_is_disambiguation_page(self, qcode: str) -> bool:
        return qcode in self.disambiguation_qcodes
