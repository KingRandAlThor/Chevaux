# Chevaux 🐎

Bot Telegram pour les courses hippiques françaises (PMU), sur le modèle de Moula. **Il ne place jamais de pari** : tu joues toi-même. Le solde est **simulé** (10 € au départ).

> Le backtest ne montre **aucun avantage** sur le PMU (voir [RAPPORT_BACKTEST.md](RAPPORT_BACKTEST.md)). C'est un outil d'étude, pas une méthode pour gagner.

## Règles
- Mise max : 6 % du solde par pari.
- Le bot « meurt » uniquement si le solde atteint 0 (pas de mort si tu ne paries pas). Un pari t'est proposé chaque jour dès 9h, avant la fenêtre 12h-14h.
- `solde 10` pour le relancer.

## Commandes Telegram
- `multi 4 2` : multi de 4 courses dont la cote cumulée ≈ 2, sur les courses les plus proches dans le temps (si la cote exacte est impossible, la meilleure approximation est proposée et signalée). Combiné simulé : tu enchaînes les mises à la main.
- `pari` · `solde` · `mission` · `mes paris` · `placé` ou `ok 0.5 @2.1` · `aide`.

## Usage
```
pip install -r requirements.txt
python -m pytest -q tests
python -m chevaux.bot demo                                    # démo hors-ligne, rien n'est envoyé
python -m chevaux.collect 2025-10-17 2026-07-25 history.db    # historique PMU (requêtes espacées)
python -m chevaux.backtest history.db                         # backtest
python -c "from chevaux.model import train_bundle; train_bundle('history.db')"   # -> data/model.joblib
```
Sans `data/model.joblib`, le bot utilise les probabilités implicites des cotes.

## GitHub Actions (gratuit)
`.github/workflows/bot.yml` lance le bot en daemon (cycles de ~6 h relancés par cron), base `bot.db` conservée via `actions/cache`.
Dans *Settings → Secrets and variables → Actions* : secrets `TELEGRAM_TOKEN`, `TELEGRAM_CHAT` ; variable optionnelle `BANKROLL`. Aucune clé dans le code.

## Données
API publique PMU (`offline.turfinfo.api.pmu.fr`), usage personnel, requêtes espacées ; respecte les CGU du PMU.
