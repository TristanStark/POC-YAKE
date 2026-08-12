# POC-YAKE

POC d'extraction automatique de mots-clés à partir d'un dossier de notes Markdown, **sans LLM, sans accès réseau et sans package Python externe**.

Le script est inspiré de YAKE (Yet Another Keyword Extractor) : il classe les termes et expressions à partir de statistiques locales à chaque document, notamment la fréquence, la position, la dispersion dans les phrases, la casse et la diversité du contexte gauche/droite.

> Ce POC n'est pas une réimplémentation bit-à-bit du package YAKE officiel. Il reprend les principales idées de scoring afin de rester autonome avec la seule bibliothèque standard Python.

## Prérequis

- Python 3.10+
- Aucun `pip install`

## Utilisation

```bash
python poc_yake.py ./mes-notes
```

Le dossier est parcouru récursivement et tous les fichiers `*.md` sont analysés indépendamment.

Par défaut :

- 20 mots-clés par fichier ;
- expressions de 1 à 3 mots ;
- stopwords français + anglais ;
- sortie dans `yake-results.json`.

Exemple plus complet :

```bash
python poc_yake.py ./mes-notes \
  --top 15 \
  --max-ngram 3 \
  --language fr \
  --dedup-threshold 0.90 \
  --output results/keywords.json
```

Pour ne lire que les `.md` directement présents dans le dossier :

```bash
python poc_yake.py ./mes-notes --no-recursive
```

## Exemple de sortie

Le terminal affiche les mots-clés par note :

```text
# cognos/incident.md
 1. IBM Cognos (score=0.01234567, occurrences=3)
 2. URI répartiteur (score=0.02500000, occurrences=2)
 3. port 9300 (score=0.04100000, occurrences=2)
```

Le JSON généré est de la forme :

```json
{
  "source_directory": "/notes",
  "document_count": 1,
  "documents": [
    {
      "path": "cognos/incident.md",
      "keywords": [
        {
          "keyword": "IBM Cognos",
          "score": 0.01234567,
          "occurrences": 3
        }
      ]
    }
  ]
}
```

Comme dans YAKE, **un score plus faible signifie un candidat plus important**.

## Pipeline

1. Suppression du bruit Markdown : front matter YAML, blocs de code, URL, syntaxe de liens et marqueurs de mise en forme.
2. Découpage du texte en phrases et tokens Unicode.
3. Calcul des statistiques de chaque terme :
   - fréquence (`WFreq`) ;
   - dispersion entre phrases (`WSpread`) ;
   - casse / acronymes (`WCase`) ;
   - position dans le document (`WPos`) ;
   - diversité du contexte gauche/droite (`WRel`).
4. Génération de n-grams de 1 à `--max-ngram` mots.
5. Score des expressions à partir des scores de leurs termes.
6. Déduplication avec `difflib.SequenceMatcher`.
7. Conservation des `--top` meilleurs candidats de chaque fichier.

## Tests

Les tests utilisent uniquement `unittest` :

```bash
python -m unittest discover -s tests -v
```

## Limites du POC

- Pas de lemmatisation : `serveur` et `serveurs` restent deux termes distincts.
- La segmentation des phrases est volontairement simple.
- Les stopwords embarqués couvrent surtout le français et l'anglais.
- La déduplication est lexicale et non sémantique.
- Le nettoyage Markdown ne cherche pas à implémenter toute la CommonMark spec.
- L'implémentation reproduit les grandes caractéristiques de YAKE mais pas tous les détails du package de référence (tokenizer, intégrité des candidats, réglages linguistiques, etc.).

Ces limites sont acceptables pour valider le principe sur un corpus réel de notes avant d'ajouter la création automatique des liens entre documents et le graphe de connaissances.

## Références

- YAKE officiel : <https://github.com/INESCTEC/yake>
- Campos et al., *YAKE! Keyword Extraction from Single Documents using Multiple Local Features*, Information Sciences, 2020, DOI: 10.1016/j.ins.2019.09.013
