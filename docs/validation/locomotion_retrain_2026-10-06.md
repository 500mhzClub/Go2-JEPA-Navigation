# Locomotion policy: retrained from scratch vs the shipped policy (6 October 2026)

A = shipped `model_500.pt` (trained 16 May 2026 under an earlier Genesis); B = retrained with `go2nav.locomotion.train`
(seed 11, 4096 robots, 501 iterations, Genesis 0.4.6, rsl-rl 5.4.1; 8 min on one R9700). Both benchmarked with
`go2nav.locomotion.benchmark` (32 robots per primitive, 8 s, 2-s warm-up; switching every 2 s). One retraining seed only.

## Normal floor

```
primitive        axis    command  A achieved  B achieved   A rms   B rms resets A/B
hold             vx        0.000       0.001      -0.001   0.001   0.001     0/0   
hold             vy        0.000       0.001      -0.002   0.001   0.002     0/0   
hold             yaw       0.000      -0.002      -0.004   0.003   0.005     0/0   
forward_slow     vx        0.200       0.181       0.196   0.026   0.015     0/0   
forward_slow     vy        0.000       0.009      -0.001   0.015   0.015     0/0   
forward_slow     yaw       0.000      -0.006       0.004   0.019   0.023     0/0   
forward_medium   vx        0.250       0.251       0.243   0.020   0.019     0/0   
forward_medium   vy        0.000       0.013       0.003   0.018   0.017     0/0   
forward_medium   yaw       0.000      -0.008       0.001   0.025   0.014     0/0   
forward_fast     vx        0.300       0.304       0.295   0.021   0.022     0/0   
forward_fast     vy        0.000      -0.001      -0.003   0.014   0.019     0/0   
forward_fast     yaw       0.000      -0.011       0.004   0.022   0.020     0/0   
backward         vx       -0.200      -0.189      -0.169   0.015   0.037     0/0   
backward         vy        0.000       0.018      -0.022   0.030   0.035     0/0   
backward         yaw       0.000       0.011      -0.003   0.030   0.029     0/0   
yaw_left         vx        0.000      -0.001      -0.002   0.013   0.013     0/0   
yaw_left         vy        0.000      -0.008      -0.015   0.018   0.027     0/0   
yaw_left         yaw       0.450       0.451       0.462   0.047   0.035     0/0   
yaw_right        vx        0.000       0.010       0.011   0.014   0.019     0/0   
yaw_right        vy        0.000       0.015       0.017   0.026   0.026     0/0   
yaw_right        yaw      -0.450      -0.456      -0.450   0.038   0.048     0/0   
arc_left         vx        0.200       0.195       0.200   0.017   0.019     0/0   
arc_left         vy        0.000      -0.005      -0.002   0.015   0.019     0/0   
arc_left         yaw       0.450       0.448       0.460   0.023   0.023     0/0   
arc_right        vx        0.200       0.199       0.204   0.019   0.018     0/0   
arc_right        vy        0.000       0.017      -0.002   0.024   0.023     0/0   
arc_right        yaw      -0.450      -0.457      -0.460   0.024   0.026     0/0   
switching: first-second error A [0.039, 0.017, 0.034] B [0.037, 0.02, 0.032]; settled A [0.013, 0.014, 0.021] B [0.012, 0.016, 0.02]; resets 0/0
```

## Uniform friction 0.3

```
primitive        axis    command  A achieved  B achieved   A rms   B rms resets A/B
hold             vx        0.000       0.000      -0.001   0.000   0.001     0/0   
hold             vy        0.000       0.000      -0.001   0.001   0.002     0/0   
hold             yaw       0.000      -0.005      -0.002   0.006   0.004     0/0   
forward_slow     vx        0.200       0.145       0.227   0.058   0.033     0/0   
forward_slow     vy        0.000       0.059       0.005   0.061   0.018     0/0   
forward_slow     yaw       0.000      -0.021      -0.012   0.046   0.049     0/0   
forward_medium   vx        0.250       0.161       0.277   0.091   0.033     0/0   
forward_medium   vy        0.000       0.074       0.011   0.076   0.022     0/0   
forward_medium   yaw       0.000      -0.027      -0.021   0.058   0.057     0/0   
forward_fast     vx        0.300       0.166       0.318   0.136   0.028     0/0   
forward_fast     vy        0.000       0.095       0.015   0.098   0.026     0/0   
forward_fast     yaw       0.000      -0.027      -0.028   0.068   0.064     0/0   
backward         vx       -0.200      -0.218      -0.246   0.021   0.048     0/0   
backward         vy        0.000       0.052       0.004   0.056   0.031     0/0   
backward         yaw       0.000       0.032      -0.016   0.059   0.075     0/0   
yaw_left         vx        0.000      -0.016      -0.004   0.024   0.016     0/0   
yaw_left         vy        0.000      -0.008       0.005   0.021   0.027     0/0   
yaw_left         yaw       0.450       0.471       0.512   0.066   0.082     0/0   
yaw_right        vx        0.000      -0.008       0.022   0.013   0.026     0/0   
yaw_right        vy        0.000       0.013      -0.010   0.026   0.025     0/0   
yaw_right        yaw      -0.450      -0.478      -0.494   0.044   0.057     0/0   
arc_left         vx        0.200       0.171       0.215   0.043   0.024     0/0   
arc_left         vy        0.000       0.044       0.018   0.053   0.029     0/0   
arc_left         yaw       0.450       0.466       0.460   0.056   0.065     0/0   
arc_right        vx        0.200       0.216       0.221   0.029   0.029     0/0   
arc_right        vy        0.000       0.013       0.001   0.022   0.028     0/0   
arc_right        yaw      -0.450      -0.460      -0.479   0.042   0.047     0/0   
switching: first-second error A [0.067, 0.034, 0.057] B [0.061, 0.023, 0.063]; settled A [0.037, 0.035, 0.04] B [0.022, 0.018, 0.045]; resets 1/0
```

Lineage of B: `outputs/locomotion/retrain_seed11/lineage.json` (not committed; checkpoints are artefacts).
