import json
import tempfile
import unittest
from pathlib import Path

from poc_yake import YakeLikeExtractor, analyze_folder


class YakeLikeExtractorTests(unittest.TestCase):
    def test_extracts_repeated_technical_phrase(self):
        text = """
# Graphe de connaissances

Un graphe de connaissances relie des notes Markdown entre elles.
Le graphe de connaissances utilise des mots clés et des relations.
Les notes Markdown peuvent ensuite former un graphe navigable.
"""
        extractor = YakeLikeExtractor(top=10, max_ngram=3, language="fr")
        keywords = extractor.extract(text)
        phrases = {keyword.keyword.casefold() for keyword in keywords}
        self.assertTrue(any("graphe" in phrase for phrase in phrases))
        self.assertTrue(any("connaissances" in phrase for phrase in phrases))

    def test_ignores_fenced_code(self):
        text = """
# Analyse Cognos
Cognos utilise un répartiteur.
```python
secretsecret secretsecret secretsecret secretsecret
```
Le répartiteur Cognos écoute sur un port.
"""
        extractor = YakeLikeExtractor(top=20, language="fr")
        phrases = {keyword.keyword.casefold() for keyword in extractor.extract(text)}
        self.assertFalse(any("secretsecret" in phrase for phrase in phrases))

    def test_keeps_useful_numeric_tokens(self):
        text = "Le port 9300 répond. Le port 9300 est utilisé par Cognos."
        extractor = YakeLikeExtractor(top=20, max_ngram=2, language="fr")
        phrases = {keyword.keyword.casefold() for keyword in extractor.extract(text)}
        self.assertTrue(any("9300" in phrase for phrase in phrases))

    def test_analyze_folder_recurses_and_ignores_non_markdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.md").write_text("Python analyse des notes Python.", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "b.md").write_text("Foundry VTT contient des scènes Foundry.", encoding="utf-8")
            (root / "ignored.txt").write_text("should not be scanned", encoding="utf-8")

            result = analyze_folder(root, YakeLikeExtractor(top=5), recursive=True)
            self.assertEqual(result["document_count"], 2)
            self.assertEqual({doc["path"] for doc in result["documents"]}, {"a.md", "nested/b.md"})
            json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
