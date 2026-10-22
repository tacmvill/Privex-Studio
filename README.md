# Privex Studio

Privex Studio est un logiciel Windows de montage et de masquage de vidéos et d’images. Il permet de choisir les sujets à protéger, de suivre leurs déplacements et d’exporter le résultat depuis une interface locale.

**Montrez l’essentiel. Protégez le reste.**

## Projet étudiant

Ce projet personnel étudiant a une vocation expérimentale, technologique et pédagogique. Il est en cours de développement. Il ne constitue pas une solution d’anonymisation certifiée.

Le projet est présenté sans affiliation politique, sans soutien à une cause, un parti ou un mouvement particulier. Les éventuelles scènes de manifestations servent uniquement à illustrer les fonctions de traitement vidéo. Leur présence ne traduit aucune adhésion.

## Auteur

Théo-Alexandre VILLAND, étudiant à l'ECE Paris
Linkedin : https://www.linkedin.com/in/theo-alexandre-villand/


## Fonctions

- Importer une vidéo ou une image et ajuster les passages conservés dans la timeline.
- Choisir les catégories à détecter : têtes, personnes et autres objets pris en charge.
- Inspecter, sélectionner et corriger les sujets identifiés.
- Appliquer un flou, un masque noir ou une mosaïque, avec réglages de couleur, de taille et de contour.
- Ajouter un suivi manuel pour un sujet oublié et contrôler les trajectoires.
- Suivre l’avancement et les ressources du PC pendant le traitement.
- Ajuster l’audio, comparer l’original au résultat et exporter la vidéo.
- Enregistrer et rouvrir les projets localement.

## Lancer l’application Windows

Dans l’archive Windows compilée, extraire **tout** le dossier `PrivexStudio`, puis ouvrir `PrivexStudio.exe`. Le dossier `_internal` doit rester à côté de l’exécutable. Python et Node.js sont intégrés ou inutiles pour cette version compilée ; Microsoft Edge WebView2 doit être installé sur Windows.

L’application ouvre sa propre fenêtre et démarre son serveur uniquement sur l’interface locale `127.0.0.1`. Les médias importés, projets, caches et journaux sont enregistrés dans `%LOCALAPPDATA%\PrivexStudio`, séparément du programme. Fermer la fenêtre arrête son serveur et les traitements qu’elle a lancés.

## Installer depuis les sources

Prévoir Windows 11 64 bits, Python 3.12 64 bits, Node.js avec npm et Microsoft Edge WebView2. Les modèles ONNX fournis doivent rester dans `backend/models`.

1. Ouvrir `INSTALLER.bat` pour installer les dépendances et compiler l’interface.
2. Ouvrir `DEMARRER.bat` pour lancer Privex Studio dans sa fenêtre.

Pour le développement : lancer `.venv\Scripts\python.exe launch.py --serve --port 8765`, puis `npm run dev` dans `frontend`. Pour construire l’application Windows : exécuter `scripts/build-windows.ps1` après installation. Le résultat se trouve dans `dist/PrivexStudio`.

La distribution standard utilise ONNX Runtime avec DirectML lorsque le matériel le permet, et un repli CPU. Les moteurs avancés Grounding DINO, SAM 2 et BoT-SORT avec ReID nécessitent leurs dépendances et modèles additionnels ; ils ne sont pas intégrés à l’exécutable standard. Les scripts de préparation sont conservés dans `scripts` pour les installations de développement.

## Limites du masquage

**Certains médias ne peuvent pas être masqués correctement.** Les visages petits, flous, partiellement cachés, les mouvements rapides, les changements de plan, les reflets et les affiches peuvent provoquer des omissions ou de fausses détections. Un suivi peut perdre un sujet ou créer plusieurs identités pour la même personne.

Vérifier l’intégralité du fichier exporté, y compris image par image dans les passages sensibles. Corriger manuellement les zones manquantes ou renoncer à utiliser un résultat insuffisant. Aucun réglage ne garantit que chaque sujet sera masqué sur chaque image. La voix, les vêtements, les tatouages ou le contexte peuvent également permettre une identification.

Les performances dépendent du PC, des dimensions du média et des réglages. Les températures affichées correspondent aux mesures accessibles sous Windows et ne couvrent pas nécessairement tous les composants.

## Organisation

| Dossier / fichier | Contenu |
| --- | --- |
| `launch.py`, `desktop.py` | Lancement et fenêtre Windows |
| `backend/` | API locale, traitement vidéo, détection, suivi et export |
| `backend/models/` | Modèles locaux et inventaire des fichiers |
| `frontend/src/` | Interface React et ressources du kit Apple |
| `frontend/public/` | Logo, polices et ressources statiques |
| `branding/` | Icône Windows |
| `scripts/` | Installation, compilation et préparation des modèles |
| `docs/` | Sources, composants et notices |

Les vidéos personnelles, les projets enregistrés, les environnements Python, les caches, les journaux et le montage publicitaire ne font pas partie de ce dossier source.

## Sources et droits

Le traitement s’appuie sur des modèles de vision préexistants et des bibliothèques tierces. Leur provenance et leurs conditions propres sont décrites dans [docs/SOURCES.md](docs/SOURCES.md) et [docs/RESSOURCES-APPLE.md](docs/RESSOURCES-APPLE.md). Le dépôt conserve les polices, SF Symbols et ressources du kit Apple utilisés par l’interface.
