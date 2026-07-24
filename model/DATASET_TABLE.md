# ECHO — Dataset Table

| Dataset | Source | Classes Used | Original Labels | Mapped Echo Labels | Number of Samples | Sampling Rate | License | Train/Val/Test Split | Limitations |
|---------|--------|--------------|-----------------|--------------------|-------------------|---------------|---------|----------------------|-------------|
| UrbanSound8K | Kaggle/UrbanSound | gunshot, siren, etc. | gun_shot, siren, etc. | Gunshot, Siren, Normal | ~8732 | Varies (resampled to 16kHz) | CC BY-NC | 70/15/15 | Clean/street recordings, possible domain mismatch. |
| ESC-50 | GitHub/Kaggle | glass_breaking, siren, etc. | glass_breaking, siren, etc. | Glass breaking, Siren, Normal | 2000 | 44.1 kHz (resampled to 16kHz) | CC0 | 70/15/15 | Short clips (5s). |
| Smartphone Test Set | Self-recorded | All | All | All | TBD (~30 mins total) | 16 kHz | Proprietary/Team | Test only | Played from speaker, not real environmental hazards. |
