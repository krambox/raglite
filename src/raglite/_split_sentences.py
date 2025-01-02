"""Sentence splitter."""

import re

from somajo import SoMaJo
import markdownify

eos_tags = "title h1 h2 h3 h4 h5 h6 br hr dl table tr".split()
no_space_before = re.compile(r" ([,.!:;)\]])")
placeholder_section_delim = "▷"

def split_sentences(doc: str, max_len: int | None = None) -> list[str]:

    cleaned_sentences = []
    sentences = SoMaJo("de_CMC").tokenize_xml(
        doc, eos_tags, strip_tags=False, prune_tags=None
    )
    for sentence in sentences:
        parts = []
        for t in sentence:
            parts.append(t.text)
            if t.space_after:
                parts.append(" ")
        s = "".join(parts)
        s = re.sub(no_space_before, r"\1", s)
        cleaned_sentences.append(s)
    splitted_html=f"{placeholder_section_delim}".join(cleaned_sentences)
    md=markdownify.markdownify(splitted_html,heading_style="ATX")
    return md.split(placeholder_section_delim)
