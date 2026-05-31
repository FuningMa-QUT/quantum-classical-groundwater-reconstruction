# Paper Budget-Ablation Notes

## Main table takeaway

The budget-matched phase protocol achieves the lowest adaptive mean overall regret (0.000322), improving over the uniform reference (0.053227) and the full higher-budget phase protocol (0.002788).

## Dense regime takeaway

At `pathline_monitoring64` with exact solver and observation noise 0.02, the budget-matched phase protocol reduces adaptive RMSE to 0.095308 while holding the total operator-observation budget at 4800.

## Budget framing

The fairness claim is now simple to state: the budget-matched protocol uses the same dense-regime total evidence budget as the uniform reference, while the full phase protocol still represents the higher-budget upper-bound protocol.
