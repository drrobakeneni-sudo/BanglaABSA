def branch_analysis(y, p4, p5, p6=None):
    from bangla_absa.evaluation.metrics import validate_predictions

    y, p4 = validate_predictions(y, p4)
    _, p5 = validate_predictions(y, p5)
    mask = y >= 0
    a, b, gold = p4.argmax(-1)[mask], p5.argmax(-1)[mask], y[mask]
    ac, bc, agree = a == gold, b == gold, a == b
    result = {
        "decisions": int(mask.sum()),
        "agree_correct": int((agree & ac).sum()),
        "agree_incorrect": int((agree & ~ac).sum()),
        "m4_only_correct": int((ac & ~bc).sum()),
        "m5_only_correct": int((bc & ~ac).sum()),
        "disagree_neither_correct": int((~agree & ~ac & ~bc).sum()),
        "hard_label_oracle_accuracy": float((ac | bc).mean()) if len(gold) else None,
    }
    if p6 is not None:
        _, p6 = validate_predictions(y, p6)
        c = p6.argmax(-1)[mask]
        correct = c == gold
        result.update(
            {
                "m6_corrects_m5": int((correct & ~bc).sum()),
                "m6_regresses_m5": int((~correct & bc).sum()),
                "m6_corrects_m4": int((correct & ~ac).sum()),
                "m6_regresses_m4": int((~correct & ac).sum()),
                "m6_selects_neither": int(((c != a) & (c != b)).sum()),
                "m6_neither_and_correct": int(((c != a) & (c != b) & correct).sum()),
            }
        )
    result["oracle_scope"] = "Hard-label branch selection only; not an upper bound on stacking"
    return result
