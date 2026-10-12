# PDP-linked immersion dataset

Current dataset: 18 rows from 5 papers. Source counts: Coat24 4, Sherif14 2, Eyu16 3, EH47_21 3, N80_24 6. The original training_51.csv remains the audit source. RE-only continuous-immersion subset remains 4 rows from Coat24. Noncontinuous auxiliary data contains 15 rows.

N80_24 (10.3390/app14020728) adds six unique Table 5 PDP/Tafel measurements. Table 1 supplies quantitative steel composition. Section 2.3 explicitly describes electrochemical measurement after autoclave immersion. Table 2 and Figure 8 link durations and temperatures to Table 5. Durations are 84, 168, 336 and 672 hours at 80 C, plus 168 hours at 40 and 120 C. The repeated 80 C / 7 day measurement is deduplicated.

N80 PDP was measured ex situ at ambient temperature and pressure. Autoclave temperature and pressure belong only to pre-exposure columns. The 3600 second OCP waiting period is separate from immersion duration. Half-immersed two-phase coupons and the unspecified tested face are flagged. Unreported RE is missing, not zero. Tables 3/4 corrosion-rate values are not targets.

Eligibility requires quantitative composition and explicitly PDP-linked positive continuous pre-exposure duration. Separate mass-loss durations and unknown durations are not substituted. Cyclic and salt-spray exposure remain auxiliary. Composition can come from a sample table, prose or documented base steel plus measured RE, with provenance retained.

Rebuild: build_time_verified.py. Modeling entry point: corrosion_time_verified.py. Any benchmark files predating this 18-row rebuild describe the previous dataset and have not been rerun for this addition. Input policy: at most 40 raw features and at most 20% missing before imputation; unknown/not_reported counts as missing. Source and target metadata are not predictors.
