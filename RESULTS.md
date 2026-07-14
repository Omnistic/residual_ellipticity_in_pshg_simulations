# Simulation 1
5x Protected Silver Mirrors & Ideal Waveplates (HWP=180°|QWP=90°)
## Parameters
- Sampling (half-waveplate, quarter-waveplate, linear polarizer): (10, 19, 18)
- Number of runs: 10
- Ground truth `theta_0, phi_0, alpha_0` are randomized every run
- Every fit is the best from 15 different starting points (randomized every run)
## Results (mean ± std)
- `I_0 = 0.843463 ± 0.000000`
- `gamma = 0.995381 ± 0.000000`
- `delta = -32.646824 ± 0.000000`
- `theta_0_error = 0.000000 ± 0.000000`
- `phi_0_error = 0.000000 ± 0.000000`
- `alpha_0_error = 0.000000 ± 0.000000`

#### Main result 0
`delta` is exactly **-32.646824°**
#### Main result 1
all angles `theta_0, phi_0, alpha_0` are exactly fitted

# Simulation 2
5x Protected Silver Mirrors + Dichroic & Ideal Waveplates (HWP=180°|QWP=90°)
## Parameters
- Dichroic retardance fixed (`dic_rng_seed=5`): **14.150088°**
## Results (mean ± std)
- `I_0 = 0.843463 ± 0.000000`
- `gamma = 0.995381 ± 0.000000`
- `delta = -18.496736 ± 0.000000`
- `theta_0_error = 0.000000 ± 0.000000`
- `phi_0_error = 0.000000 ± 0.000000`
- `alpha_0_error = 0.000000 ± 0.000000`

#### Main result 0
`I_0` and `gamma` are identical to [Simulation 1](#simulation-1)

#### Main result 1
`delta` is exactly **-18.496736°** and when subtracted with the dichroic retardance (**14.150088°**), it results in the five protected silver mirrors retardance: **-32.646824°**

#### Main result 2
all angles `theta_0, phi_0, alpha_0` are exactly fitted

# Simulation 3
5x Protected Silver Mirrors + Dichroic & Real Waveplates (HWP=185.7492°|QWP=92.8728°)
## Parameters
- Dichroic retardance fixed (`dic_rng_seed=5`): **14.150088°**
## Results (mean ± std)
- `I_0 = 0.842418 ± 0.001151`
- `gamma = 0.996286 ± 0.001507`
- `delta = -16.097984 ± 1.046766`
- `theta_0_error = 0.451064 ± 1.517613`
- `phi_0_error = -0.068558 ± 0.060938`
- `alpha_0_error = 1.057792 ± 3.199740`

#### Main result 0
`I_0` and `gamma` are fitted down to the third decimal

#### Main result 1
`delta` shows significant variation

#### Main result 2
`theta_0` and `alpha_0` show significant variation and the error and associated STD are related by a factor of 2 (in the general intensity equation, `theta` and `alpha` are sometimes bundled as `alpha-2*theta`)

#### Main result 3
`phi_0` is fitted down to the second decimal

# Simulation 4
5x Protected Silver Mirrors + Dichroic & Real Waveplates (HWP=185.7492°|QWP=92.8728°)
## Parameters
- Dichroic retardance fixed (hard-coded in line 756): **20.000000°**
## Results (mean ± std)
- `I_0 = 0.842303 ± 0.001006`
- `gamma = 0.996529 ± 0.001281 `
- `delta = -9.971424 ± 0.647908`
- `theta_0_error = -0.114494 ± 2.235158`
- `phi_0_error = -0.069634 ± 0.069760`
- `alpha_0_error = -0.112006 ± 4.603963`

#### Main result 0
When running [Simulation 3](#simulation-3) multiple times, we realized that the smaller `delta`, the more incertain was the error on `theta_0` and `alpha_0`. Assuming the five mirrors give a retardance of about -32°, a dichroic retardance of 20° gives a delta about -12°, which is what we had previously measured on our Leica system. In fact, it can be shown that if `I_0` and `gamma` are 1, and `delta` is 0, there is an infinite number of combinations possible for `theta_0` and `alpha_0`.