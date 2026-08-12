#!/usr/bin/env python3
"""Dependency-free YAKE-inspired keyword extraction for Markdown folders.

This is a proof of concept, not a byte-for-byte reimplementation of the
reference YAKE package. It reproduces the core idea: rank keyword candidates
using local statistics from each document only (frequency, position, sentence
spread, casing and left/right contextual relatedness), then deduplicate similar
phrases.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Sequence


FRENCH_STOPWORDS = {
    "a", "ai", "aie", "aient", "ainsi", "alors", "apres", "après", "as", "assez", "au", "aucun",
    "aucune", "aucuns", "aux", "avaient", "avais", "avait", "avec", "avez", "aviez", "avions", "avons",
    "ayant", "beaucoup", "bien", "c", "ca", "car", "ce", "ceci", "cela", "celle", "celles", "celui",
    "cependant", "ces", "cet", "cette", "ceux", "chaque", "chez", "comme", "comment", "contre", "d",
    "dans", "de", "dedans", "dehors", "depuis", "des", "donc", "dont", "du", "elle", "elles", "en",
    "encore", "entre", "est", "et", "etaient", "étaient", "etais", "étais", "etait", "était", "ete", "été",
    "etes", "êtes", "etre", "être", "eu", "eue", "eues", "eurent", "eus", "eusse", "eussent", "eusses",
    "eut", "eux", "fait", "faites", "fois", "font", "ici", "il", "ils", "j", "je", "jusqu", "la", "le",
    "les", "leur", "leurs", "lui", "m", "ma", "mais", "me", "meme", "même", "mes", "moi", "moins", "mon",
    "ne", "ni", "nos", "notre", "nous", "n", "on", "ont", "ou", "où", "par", "parce", "pas", "peu",
    "peut", "plus", "pour", "pourquoi", "qu", "quand", "que", "quel", "quelle", "quelles", "quels", "qui",
    "sa", "sans", "se", "sera", "seraient", "serais", "serait", "seras", "serez", "seriez", "serions", "serons",
    "ses", "si", "soi", "soient", "sommes", "sont", "sous", "sur", "t", "ta", "te", "tes", "toi", "ton",
    "tous", "tout", "toute", "toutes", "tres", "très", "tu", "un", "une", "vos", "votre", "vous", "y"
}

ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could",
    "d", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "ll", "m", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "re", "s", "same", "she", "should", "so", "some", "such", "t", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "ve", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

STOPWORDS_BY_LANGUAGE = {
    "fr": FRENCH_STOPWORDS,
    "en": ENGLISH_STOPWORDS,
    "mixed": FRENCH_STOPWORDS | ENGLISH_STOPWORDS,
}

# Unicode-aware token: first char must be alphanumeric; useful numbers (ports, versions, dates) are retained.
TOKEN_RE = re.compile(r"[^\W_][\w'’\-]*", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?:[.!?]+|[\r\n]+)")
FENCED_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
FRONT_MATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*(?:\n|\Z)", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]+`")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MARKDOWN_MARKERS_RE = re.compile(r"(?m)^\s{0,3}(?:#{1,6}|>|[-+*]|\d+[.)])\s+")
EMPHASIS_RE = re.compile(r"[*_~]+")


@dataclass
class TokenOccurrence:
    text: str
    normalized: str
    sentence_id: int
    position: int
    is_acronym: bool
    is_proper: bool


@dataclass
class TermStats:
    term: str
    tf: int = 0
    acronym_tf: int = 0
    proper_tf: int = 0
    sentence_ids: set[int] = field(default_factory=set)
    sentence_positions: list[int] = field(default_factory=list)
    left_neighbors: Counter[str] = field(default_factory=Counter)
    right_neighbors: Counter[str] = field(default_factory=Counter)
    first_surface: str = ""
    w_rel: float = 1.0
    w_freq: float = 0.0
    w_spread: float = 0.0
    w_case: float = 0.0
    w_pos: float = 1.0
    pl: float = 0.0
    pr: float = 0.0
    h: float = 1.0


@dataclass(frozen=True)
class Keyword:
    keyword: str
    score: float
    occurrences: int


class YakeLikeExtractor:
    """Small YAKE-inspired extractor implemented with Python's standard library."""

    def __init__(
        self,
        *,
        max_ngram: int = 3,
        top: int = 20,
        language: str = "mixed",
        dedup_threshold: float = 0.90,
        window_size: int = 1,
        min_token_length: int = 2,
    ) -> None:
        if max_ngram < 1:
            raise ValueError("max_ngram must be >= 1")
        if top < 1:
            raise ValueError("top must be >= 1")
        if language not in STOPWORDS_BY_LANGUAGE:
            raise ValueError(f"language must be one of: {', '.join(STOPWORDS_BY_LANGUAGE)}")
        if not 0.0 <= dedup_threshold <= 1.0:
            raise ValueError("dedup_threshold must be between 0 and 1")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        self.max_ngram = max_ngram
        self.top = top
        self.stopwords = STOPWORDS_BY_LANGUAGE[language]
        self.dedup_threshold = dedup_threshold
        self.window_size = window_size
        self.min_token_length = min_token_length

    @staticmethod
    def clean_markdown(text: str) -> str:
        text = FRONT_MATTER_RE.sub("\n", text)
        text = FENCED_CODE_RE.sub("\n", text)
        text = HTML_COMMENT_RE.sub("\n", text)
        text = INLINE_CODE_RE.sub(" ", text)
        text = IMAGE_RE.sub(lambda m: f" {m.group(1)} ", text)
        text = WIKILINK_RE.sub(lambda m: f" {m.group(2) or m.group(1)} ", text)
        text = LINK_RE.sub(lambda m: f" {m.group(1)} ", text)
        text = URL_RE.sub(" ", text)
        text = MARKDOWN_MARKERS_RE.sub("", text)
        text = EMPHASIS_RE.sub("", text)
        return text

    def extract(self, markdown_text: str) -> list[Keyword]:
        text = self.clean_markdown(markdown_text)
        sentences = self._tokenize_sentences(text)
        all_tokens = [token for sentence in sentences for token in sentence]
        if not all_tokens:
            return []

        terms = self._build_term_stats(sentences)
        self._compute_term_features(terms, len(sentences))
        candidates = self._build_candidates(sentences, terms)
        return self._rank_and_deduplicate(candidates)

    def _tokenize_sentences(self, text: str) -> list[list[TokenOccurrence]]:
        raw_sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
        sentences: list[list[TokenOccurrence]] = []
        global_position = 0

        for sentence_id, raw_sentence in enumerate(raw_sentences, start=1):
            current: list[TokenOccurrence] = []
            for match in TOKEN_RE.finditer(raw_sentence):
                surface = match.group(0)
                normalized = surface.casefold().strip("-'’_")
                if not normalized or len(normalized) < self.min_token_length:
                    continue
                global_position += 1
                is_acronym = len(surface) > 1 and surface.isupper() and any(ch.isalpha() for ch in surface)
                is_proper = surface[:1].isupper() and not is_acronym
                current.append(
                    TokenOccurrence(
                        text=surface,
                        normalized=normalized,
                        sentence_id=sentence_id,
                        position=global_position,
                        is_acronym=is_acronym,
                        is_proper=is_proper,
                    )
                )
            if current:
                sentences.append(current)

        # Re-index sentence IDs after dropping empty sentences.
        for new_sentence_id, sentence in enumerate(sentences, start=1):
            for token in sentence:
                token.sentence_id = new_sentence_id
        return sentences

    def _build_term_stats(self, sentences: Sequence[Sequence[TokenOccurrence]]) -> dict[str, TermStats]:
        terms: dict[str, TermStats] = {}

        for sentence in sentences:
            for index, token in enumerate(sentence):
                stats = terms.setdefault(token.normalized, TermStats(term=token.normalized, first_surface=token.text))
                stats.tf += 1
                stats.acronym_tf += int(token.is_acronym)
                stats.proper_tf += int(token.is_proper)
                stats.sentence_ids.add(token.sentence_id)
                stats.sentence_positions.append(token.sentence_id)

                left_start = max(0, index - self.window_size)
                for neighbor in sentence[left_start:index]:
                    stats.left_neighbors[neighbor.normalized] += 1

                right_end = min(len(sentence), index + self.window_size + 1)
                for neighbor in sentence[index + 1:right_end]:
                    stats.right_neighbors[neighbor.normalized] += 1

        return terms

    def _compute_term_features(self, terms: dict[str, TermStats], sentence_count: int) -> None:
        frequencies = [stats.tf for stats in terms.values()]
        max_tf = max(frequencies, default=1)
        avg_tf = statistics.fmean(frequencies) if frequencies else 1.0
        std_tf = statistics.pstdev(frequencies) if len(frequencies) > 1 else 0.0
        frequency_denominator = avg_tf + std_tf or 1.0

        for stats in terms.values():
            wdr = len(stats.right_neighbors)
            wir = sum(stats.right_neighbors.values())
            pwr = (wdr / wir) if wir else 0.0

            wdl = len(stats.left_neighbors)
            wil = sum(stats.left_neighbors.values())
            pwl = (wdl / wil) if wil else 0.0

            stats.pl = wdl / max_tf
            stats.pr = wdr / max_tf
            stats.w_rel = (0.5 + pwl * (stats.tf / max_tf)) + (0.5 + pwr * (stats.tf / max_tf))
            stats.w_freq = stats.tf / frequency_denominator
            stats.w_spread = len(stats.sentence_ids) / max(sentence_count, 1)
            stats.w_case = max(stats.acronym_tf, stats.proper_tf) / (1.0 + math.log(stats.tf))

            median_sentence = statistics.median(stats.sentence_positions) if stats.sentence_positions else 1.0
            stats.w_pos = math.log(math.log(3.0 + median_sentence))

            denominator = stats.w_case + (stats.w_freq / stats.w_rel) + (stats.w_spread / stats.w_rel)
            stats.h = (stats.w_pos * stats.w_rel) / denominator if denominator > 0 else float("inf")

    def _build_candidates(
        self,
        sentences: Sequence[Sequence[TokenOccurrence]],
        terms: dict[str, TermStats],
    ) -> dict[str, tuple[str, float, int]]:
        counts: Counter[tuple[str, ...]] = Counter()
        surfaces: dict[tuple[str, ...], str] = {}

        for sentence in sentences:
            normalized_tokens = [token.normalized for token in sentence]
            for start in range(len(sentence)):
                for size in range(1, self.max_ngram + 1):
                    end = start + size
                    if end > len(sentence):
                        break
                    phrase_tokens = normalized_tokens[start:end]

                    # YAKE-style candidates should not begin/end with stopwords.
                    if phrase_tokens[0] in self.stopwords or phrase_tokens[-1] in self.stopwords:
                        continue
                    if all(token in self.stopwords for token in phrase_tokens):
                        continue

                    key = tuple(phrase_tokens)
                    counts[key] += 1
                    surfaces.setdefault(key, " ".join(token.text for token in sentence[start:end]))

        candidates: dict[str, tuple[str, float, int]] = {}
        for key, occurrence_count in counts.items():
            content_terms = [terms[token] for token in key if token not in self.stopwords and token in terms]
            if not content_terms:
                continue

            if len(key) == 1:
                score = content_terms[0].h
            else:
                product_h = math.prod(max(term.h, 1e-12) for term in content_terms)
                sum_tf = sum(term.tf for term in content_terms)
                sum_pl = sum(term.pl for term in content_terms)
                sum_pr = sum(term.pr for term in content_terms)
                score = product_h / (max(sum_tf, 1) * (1.0 + sum_pl) * (1.0 + sum_pr))

                # Reward repeated exact phrases mildly, without letting frequency dominate.
                score /= 1.0 + math.log1p(max(0, occurrence_count - 1))

            normalized_phrase = " ".join(key)
            candidates[normalized_phrase] = (surfaces[key], score, occurrence_count)

        return candidates

    def _rank_and_deduplicate(self, candidates: dict[str, tuple[str, float, int]]) -> list[Keyword]:
        ranked = sorted(candidates.items(), key=lambda item: (item[1][1], -item[1][2], item[0]))
        selected: list[Keyword] = []
        selected_normalized: list[str] = []

        for normalized, (surface, score, occurrences) in ranked:
            if any(self._similarity(normalized, existing) >= self.dedup_threshold for existing in selected_normalized):
                continue
            selected.append(Keyword(keyword=surface, score=score, occurrences=occurrences))
            selected_normalized.append(normalized)
            if len(selected) >= self.top:
                break

        return selected

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def iter_markdown_files(folder: Path, recursive: bool = True) -> Iterable[Path]:
    pattern = "**/*.md" if recursive else "*.md"
    yield from sorted(path for path in folder.glob(pattern) if path.is_file())


def analyze_folder(
    folder: Path,
    extractor: YakeLikeExtractor,
    *,
    recursive: bool = True,
) -> dict:
    documents = []
    for path in iter_markdown_files(folder, recursive=recursive):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")

        keywords = extractor.extract(text)
        documents.append(
            {
                "path": path.relative_to(folder).as_posix(),
                "keywords": [
                    {
                        "keyword": keyword.keyword,
                        "score": round(keyword.score, 8),
                        "occurrences": keyword.occurrences,
                    }
                    for keyword in keywords
                ],
            }
        )

    return {
        "source_directory": str(folder.resolve()),
        "document_count": len(documents),
        "documents": documents,
    }


def print_summary(result: dict) -> None:
    for document in result["documents"]:
        print(f"\n# {document['path']}")
        for index, keyword in enumerate(document["keywords"], start=1):
            print(
                f"{index:>2}. {keyword['keyword']} "
                f"(score={keyword['score']:.8f}, occurrences={keyword['occurrences']})"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract YAKE-inspired keywords from every Markdown file in a folder, without external packages."
    )
    parser.add_argument("folder", type=Path, help="Folder containing Markdown (.md) files")
    parser.add_argument("--top", type=int, default=20, help="Keywords to keep per document (default: 20)")
    parser.add_argument("--max-ngram", type=int, default=3, help="Maximum keyword length in words (default: 3)")
    parser.add_argument(
        "--language",
        choices=sorted(STOPWORDS_BY_LANGUAGE),
        default="mixed",
        help="Stopword set: fr, en or mixed (default: mixed)",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.90,
        help="Sequence similarity threshold used to remove near-duplicate phrases (default: 0.90)",
    )
    parser.add_argument("--window-size", type=int, default=1, help="Context window for relatedness (default: 1)")
    parser.add_argument("--no-recursive", action="store_true", help="Only inspect .md files directly inside the folder")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yake-results.json"),
        help="JSON output path (default: yake-results.json)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    folder: Path = args.folder
    if not folder.exists():
        raise SystemExit(f"Input folder does not exist: {folder}")
    if not folder.is_dir():
        raise SystemExit(f"Input path is not a directory: {folder}")

    extractor = YakeLikeExtractor(
        max_ngram=args.max_ngram,
        top=args.top,
        language=args.language,
        dedup_threshold=args.dedup_threshold,
        window_size=args.window_size,
    )
    result = analyze_folder(folder, extractor, recursive=not args.no_recursive)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(result)
    print(f"\nAnalyzed {result['document_count']} Markdown file(s).")
    print(f"JSON written to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
