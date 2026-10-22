# Vérification de la distribution Windows

Vérification locale du 22 octobre 2026 sur Windows 11, Python embarqué 3.12, application 2.9.

- Compilation TypeScript et Vite réussie depuis les sources avec `npm ci` puis `npm run build`.
- Interface d’origine conservée, y compris les ressources Apple.
- API, interface compilée et modèles locaux accessibles depuis la distribution.
- Import d’une vidéo de contrôle de huit images en 640 × 360.
- Détection de personnes / têtes exécutée dans un processus de traitement séparé.
- Application d’un masque manuel puis export d’un MP4 décodable, distinct de l’original.
- Même contrôle réussi depuis les sources et depuis `PrivexStudio.exe`.
- Démarrage de la fenêtre WebView2 : l’interface charge et interroge les mesures système.
- Compilation et vérifications effectuées sans modifier les vidéos et projets personnels d’origine.

Ces contrôles valident le lancement et le parcours de traitement sur un média court. Ils ne garantissent ni la qualité du masquage sur tous les médias, ni le comportement sur toute configuration Windows. Les modules avancés optionnels PyTorch ne font pas partie de cette validation.
