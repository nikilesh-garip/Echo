import time
from risk_scorer import RiskScorer

def test_risk_scorer():
    print("Testing Risk Scorer Engine...")
    
    # 1. Normal conversation test
    scorer = RiskScorer()
    score, level = scorer.calculate_risk(
        primary_conf=0.1,
        verification_conf=0.1,
        media_playback=False,
        sudden_motion=False,
        current_class="normal"
    )
    print(f"Normal: Score={score}, Level={level}")
    # Expected: (0.35*0.1 + 0.35*0.1)/1.15 = 0.07/1.15 = 6.08% -> 6% -> NORMAL
    assert score == 6, f"Expected 6, got {score}"
    assert level == "NORMAL", f"Expected NORMAL, got {level}"
    
    # 2. Isolated weak gunshot candidate
    score, level = scorer.calculate_risk(
        primary_conf=0.55,
        verification_conf=0.45,
        media_playback=False,
        sudden_motion=False,
        current_class="gunshot"
    )
    print(f"Weak Gunshot: Score={score}, Level={level}")
    # Expected: (0.35*0.55 + 0.35*0.45)/1.15 = 0.35/1.15 = 30.43% -> 30% -> NORMAL
    assert score == 30, f"Expected 30, got {score}"
    assert level == "NORMAL", f"Expected NORMAL, got {level}"

    # Reset scorer to avoid influence of history
    scorer = RiskScorer()
    
    # 3. Strong verified gunshot
    score, level = scorer.calculate_risk(
        primary_conf=0.95,
        verification_conf=0.90,
        media_playback=False,
        sudden_motion=True,
        current_class="gunshot"
    )
    print(f"Strong Verified Gunshot: Score={score}, Level={level}")
    # Expected: (0.35*0.95 + 0.35*0.90 + 0.15)/1.15 = 0.7975/1.15 = 69.34% -> 69% -> POSSIBLE_DANGER
    assert score == 69, f"Expected 69, got {score}"
    assert level == "POSSIBLE_DANGER", f"Expected POSSIBLE_DANGER, got {level}"

    # Reset scorer
    scorer = RiskScorer()

    # 4. Movie/media gunshot
    score, level = scorer.calculate_risk(
        primary_conf=0.95,
        verification_conf=0.90,
        media_playback=True,
        sudden_motion=False,
        current_class="gunshot"
    )
    print(f"Movie Gunshot: Score={score}, Level={level}")
    # Expected: (0.35*0.95 + 0.35*0.90 - 0.25)/1.15 = 0.3975/1.15 = 34.56% -> 35% -> SUSPICIOUS
    assert score == 35, f"Expected 35, got {score}"
    assert level == "SUSPICIOUS", f"Expected SUSPICIOUS, got {level}"

    # Reset scorer
    scorer = RiskScorer()

    # 5. Distress scream
    score, level = scorer.calculate_risk(
        primary_conf=0.85,
        verification_conf=0.80,
        media_playback=False,
        sudden_motion=True,
        current_class="scream"
    )
    print(f"Distress Scream: Score={score}, Level={level}")
    # Expected: (0.35*0.85 + 0.35*0.80 + 0.15)/1.15 = 0.7275/1.15 = 63.26% -> 63% -> POSSIBLE_DANGER
    assert score == 63, f"Expected 63, got {score}"
    assert level == "POSSIBLE_DANGER", f"Expected POSSIBLE_DANGER, got {level}"

    # 6. Multi-event dangerous sequence (Accumulating Temporal History)
    scorer = RiskScorer()
    
    # Event 1: Gunshot
    score1, level1 = scorer.calculate_risk(
        primary_conf=0.80,
        verification_conf=0.80,
        media_playback=False,
        sudden_motion=True,
        current_class="gunshot"
    )
    # Expected: (0.35*0.80 + 0.35*0.80 + 0.15)/1.15 = 0.71/1.15 = 61.7% -> 62% -> POSSIBLE_DANGER
    print(f"Seq Event 1 (Gunshot): Score={score1}, Level={level1}")
    assert score1 == 62, f"Expected 62, got {score1}"
    
    # Event 2: Scream (within 1s)
    score2, level2 = scorer.calculate_risk(
        primary_conf=0.80,
        verification_conf=0.80,
        media_playback=False,
        sudden_motion=True,
        current_class="scream"
    )
    # Expected: repeats=1. (0.35*0.80 + 0.35*0.80 + 0.15 + 0.10*1)/1.15 = 0.81/1.15 = 70.43% -> 70% -> POSSIBLE_DANGER
    print(f"Seq Event 2 (Scream): Score={score2}, Level={level2}")
    assert score2 == 70, f"Expected 70, got {score2}"
    
    # Event 3: Shouting (within 2s)
    score3, level3 = scorer.calculate_risk(
        primary_conf=0.90,
        verification_conf=0.90,
        media_playback=False,
        sudden_motion=True,
        current_class="shouting"
    )
    # Expected: repeats=2. (0.35*0.90 + 0.35*0.90 + 0.15 + 0.10*2)/1.15 = 0.98/1.15 = 85.21% -> 85% -> HIGH_RISK
    print(f"Seq Event 3 (Shouting): Score={score3}, Level={level3}")
    assert score3 == 85, f"Expected 85, got {score3}"
    assert level3 == "HIGH_RISK", f"Expected HIGH_RISK, got {level3}"
    
    print("All Risk Scorer Unit Tests Passed Successfully!")

if __name__ == "__main__":
    test_risk_scorer()
