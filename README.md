# covtl-lambda

**L'effet d'assimilation articulatoire λ sur les doubles consonnes (C1C2 → C1′C2), obtenu par une modification du modèle COVTL — livré comme un patch installable sur [covtl-pipeline](https://github.com/fberthommier/covtl-pipeline) et un ensemble de démonstrations interactives.**

`patch v1.2.0` · GPL-3.0-or-later · 53 tests

> covtl-lambda **n'est pas une copie de covtl-pipeline** : ce dépôt ne
> contient aucun code du moteur. Il fournit (1) un patch installable
> dans un checkout covtl-pipeline, (2) des démonstrations
> autonome pré-synthétisées, consultables sans le moteur. Une console
> web (covtl-console) sera ajoutée ultérieurement.

**Contenu en trois temps :** [1. Lire les manuels](#1-lire-les-manuels) · [2. Visualiser les démonstrations](#2-visualiser-les-démonstrations) · [3. Installer covtl-pipeline et son patch](#3-installer-covtl-pipeline-et-son-patch-en-dernier)

---

## 1. Lire les manuels

| Document | Contenu |
|---|---|
| **[Manuel de référence (PDF, 14 p.)](docs/manual.pdf)** | la sémantique λ complète : §2.1 définition sur les cibles polaires, §4 CLI, §5 API Python, §6 continuum de référence pas à pas, §7 garanties et tests, §8 limites |
| [Rapport d'analyse (EN, PDF)](docs/covtl_lambda_C1C2_report.pdf) | formalisation (ρ, θ), invariance de C2, dérive causale de C1, valeurs mesurées |
| [Rapport d'analyse (FR, PDF)](docs/rapport_lambda_C1C2.pdf) | la même analyse en français, avec l'implication causale pour les sondages de réseaux de neurones (Wav2Vec2) |
| [CHANGELOG](CHANGELOG.md) | historique v1.0 → v1.2.0 (la correction sémantique) |
| [NOTICE — provenance & contributeurs](NOTICE.md) | trois niveaux de provenance, système d'import des données, licences |

L'essentiel en une équation — pour la nasale C1 d'une double
consonne tautosyllabique, avec C2 l'attaque de la même double :

<p align="center"><code>θ_C1(λ) = θ_C1 + λ · wrap(θ_C2 − θ_C1)&nbsp;&nbsp;&nbsp;(ρ optionnel)</code></p>

À λ = 0 la synthèse est strictement C1C2 ; à λ = 1 la cible polaire de
C1 **égale** les coordonnées de la cible C2 (C1′C2, la double
assimilée) — C2 ne bouge jamais, le bloc glottal (voisement, F0,
nasalité) est constant au bit près sur tout le continuum. Les formes
pointées `C.C` (ex. `klin.pan`) sont une synthèse différente, **hors
périmètre**.

## 2. Visualiser les démonstrations

Les démos sont des fichiers HTML **autonomes** : audio, figures et
trajectoires polaires y sont embarqués à la construction (système
d'import décrit dans le [NOTICE](NOTICE.md)) — aucun serveur, aucun
moteur, fonctionne hors ligne. Depuis GitHub, ouvrez un lien puis
**Raw** (ou téléchargement) pour la vue interactive ; avec GitHub
Pages activé, elles se lisent directement à
`https://<org>.github.io/covtl-lambda/demos/…`.

**Exemple travaillé (K3) :** [demos/lambda_demo.html](demos/lambda_demo.html) —
`klinkan → klimkan`, glisseur λ (9 pas), figure polaire avec le nœud
de commande C1 tournant vers la cible C2 /k/ **fixe**, audio par λ,
tableau d'articulateurs.

**Réplication des items de Pouw et al. (2024)** — six cadres de
phrases, chacun un continuum λ confiné re-synthétisé (locuteur jd3) ;
*licensed* = l'assimilation n → m devant labiale est licite,
*unlicensed* = elle ne l'est pas :

| Page | Cadre | Paire | C2 | Licence |
|---|---|---|---|---|
| [pouw_P1](demos/pouw/pouw_P1.html) | *clean pan* | `klinpan → klimpan` | /p/ | licensed |
| [pouw_P2](demos/pouw/pouw_P2.html) | *mean boat* | `minbot → mimbot` | /b/ | licensed |
| [pouw_P4](demos/pouw/pouw_P4.html) | *thin packet* | `Tinpakit → Timpakit` | /p/ | licensed |
| [pouw_NV1](demos/pouw/pouw_NV1.html) | *clean tan* | `klintan → klimtan` | /t/ | unlicensed |
| [pouw_NV3](demos/pouw/pouw_NV3.html) | *clean can* | `klinkan → klimkan` | /k/ | unlicensed |
| [pouw_NV4](demos/pouw/pouw_NV4.html) | *mean goat* | `mingot → mimgot` | /g/ | unlicensed |
| [index](demos/pouw/index.html) | — | vue d'ensemble des six cadres | — | — |

Dans chaque démo : touches `←`/`→` pour λ ±0,125, `espace` lecture de
l'état courant. Le pied de page enregistre la fenêtre C1 (frames,
400 Hz) re-dérivée du timing phonémique `.polar` et la garantie de
localité `outside_c1_arcs_bitwise_canon`.

## 3. Installer covtl-pipeline et son patch (en dernier)

Les manuels et les démos suffisent pour comprendre et montrer l'effet.
Pour **produire** vos propres continiums, il faut le moteur :

1. **Cloner et installer covtl-pipeline** (non inclus ici) :
   ```bash
   git clone https://github.com/fberthommier/covtl-pipeline
   pip install -e covtl-pipeline[plot,dev]
   ```
2. **Appliquer le patch** depuis ce dépôt :
   ```bash
   python patch/apply_patch.py --pipeline-root /chemin/vers/covtl-pipeline
   ```
   Il installe `vtl_synth/core/morphing.py`, `vtl_synth/cli/morph_cmd.py`
   et `tests/test_morphing.py` (53 tests). Le dépôt du moteur ne
   porte **jamais** de code λ — `patch/revert_patch.py` restaure
   l'état d'origine.
3. **Vérifier** :
   ```bash
   python -m pytest tests/test_morphing.py -q        # 53 passed
   vtl-synth morph --phonetic --targets klinpan -o morph_out --label P1
   ```

Le guide pas à pas avec sorties attendues et dépannage :
[covtl-lambda-patch_INSTALL_PROMPT.md](covtl-lambda-patch_INSTALL_PROMPT.md).
Un changement de moteur non couvert se manifeste d'abord par l'échec
des tests de non-régression du patch.

### Écriture SAMPA : la garde et les non-mots anglais

L'analyseur phonétique du moteur supprime silencieusement (désormais :
avec **avertissement** de la garde SAMPA fusionnée en amont) les
caractères hors de son inventaire. Écrire `cleanpan` ne lit donc pas
« clean + pan » : le `c` disparaît et le moteur synthétise
**`leanpan`** — un autre mot, sans le /k/. D'où la règle : écrire
chaque phone avec son symbole SAMPA, sans reporter l'orthographe
anglaise — les paires s'écrivent **`klinpan` / `klimpan`**
(et `klintan/klimtan`, `klinkan/klimkan`, `minbot/mimbot`,
`Tinpakit/Timpakit`, `mingot/mimgot`). Le workflow de
[examples/](examples/) vérifie systématiquement l'inventaire phonémique
attendu avant synthèse.

## Layout du dépôt

```
docs/             manuel de référence (tex + PDF), rapports EN/FR, figures
demos/            démonstrations HTML autonomes (données embarquées)
patch/            le patch du moteur : apply / revert / fichiers installés
examples/         drivers Python des continua de référence + générateurs des démos
covtl-lambda-patch_INSTALL_PROMPT.md   guide d'installation/vérification
```

*Une console web (covtl-console) sera ajoutée ultérieurement.*

## Licence et contributeurs

Code, documentation et payloads des démos : **GPL-3.0-or-later** —
voir [LICENSE](LICENSE) et [NOTICE.md](NOTICE.md) (obligation de lien
avec le moteur GPL ; liste des contributeurs ; système d'import des
données : tout est synthétisé localement, aucune donnée tierce
redistribuée).

## Référence

C. Pouw, M. de Heer Kloots, A. Alishahi, W. Zuidema. *Perception of
phonological assimilation by neural speech recognition models.*
Computational Linguistics 50(4):1557–1585, 2024.
[doi:10.1162/coli_a_00536](https://doi.org/10.1162/coli_a_00536)
