# Rapport de backtest (honnête)

Généré par `python -m chevaux.backtest data/history.db --out data/backtest.json` (chiffres bruts dans ce JSON, non versionné car la base est exclue).

## Données
- Source : API publique PMU (turfinfo), courses **françaises** terminées, 17/10/2025 → 25/07/2026 (collecte interrompue à cette date).
- 2 257 courses, 26 170 partants. Cote = dernier rapport « direct » avant le départ ; gains = **rapports définitifs réels** (prélèvement PMU inclus).
- Découpage **temporel** strict : entraînement 17/10→23/03, validation 24/03→02/07, **test 03/07→25/07 (446 courses, jamais vues)**.
- Stats jockey/entraîneur calculées uniquement sur le passé strict (test unitaire anti-fuite). Modèle et seuil de « value » choisis sur la **validation** uniquement ; le test n'est utilisé qu'une fois.

## Qualité des probabilités (log-loss test, plus bas = mieux)
| Modèle | Log-loss |
|---|---|
| Probabilité implicite des cotes (marché) | 0,2501 |
| Régression logistique | 0,2503 |
| Gradient boosting (choisi sur validation) | 0,2516 |

Différence vs marché (IC95 bootstrap par course) : logreg −0,0002 [−0,0013 ; +0,0009] ; boosting −0,0015 [−0,0028 ; 0,0000]. **Aucun modèle ne fait mieux que le marché** (le boosting est plutôt légèrement moins bon).

## ROI test, mise plate 1 € (IC95 bootstrap sur les courses)
| Stratégie | Paris | ROI | IC95 |
|---|---|---|---|
| Favori, simple gagnant | 446 | −9,3 % | [−22,2 % ; +4,2 %] |
| Favori, simple placé | 446 | −4,6 % | [−12,0 % ; +3,2 %] |
| Modèle, meilleur cheval gagnant | 446 | −5,7 % | [−18,8 % ; +7,5 %] |
| Hasard, gagnant (moyenne de 300 tirages) | – | −18,9 % (σ 21 %) | – |
| Hasard, placé | – | −20,5 % (σ 8 %) | – |
| **Value gagnant** (EV modèle > 1, seuil choisi en validation) | 791 | **−41,6 %** | [−68,0 % ; +1,6 %] |
| Value placé | 0 | – | seuil validation non satisfait, aucun pari |

## Conclusion
**Aucun avantage démontré.** Le favori perd moins que le hasard (le marché est informatif) mais reste négatif, cohérent avec le prélèvement PMU (~15-25 %). La recherche de « value » sélectionne surtout des outsiders surévalués par le modèle et perd fortement. Les ROI du favori et du modèle ont des IC contenant 0 : on ne peut pas les distinguer.

## Limites
- Test court (3 semaines, 446 courses) : faible puissance statistique, seuls de gros effets seraient détectables.
- Historique de 9 mois seulement ; pas de données de terrain/pénétromètre ni forme détaillée par course passée.
- Cote prise juste avant le départ ; le pari réel se fait avant, à des cotes différentes (mutuel : la cote bouge avec les mises).
- Pari placé : le dividende n'est pas connu à l'avance, il est estimé à partir de la cote (régression).
- Le bot simule le solde ; rien n'est garanti. Ne jouez que ce que vous pouvez perdre.
