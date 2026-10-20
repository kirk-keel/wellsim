# G2 data consistency screen: do Table A.1's two columns agree?

Run before any scoring. Nothing is tuned to either column here; this only asks
whether the reported gradient and the reported holdup describe the same run.

## Method

The reported gradient is total. Strip friction and acceleration from it, using
the closure's own mixture velocity, and what is left must be hydrostatic, which
fixes the liquid holdup:

    H_l_implied = (dp/dz - dp/dz_friction - dp/dz_accel) / (rho_l g)

with rho_l g = 62.3 / 144 = 0.4326 psi/ft per unit holdup. The gas contributes
alpha rho_g to the real mixture density, under 0.15% of rho_l here, and is
dropped as the formula drops it. Friction and acceleration come from the closure
because the reported holdup is the thing under test and cannot also be the input.

The difference is compared against the run's own reported uncertainty, combining
the holdup tolerance with the holdup-equivalent of the gradient tolerance in
quadrature (BOEM M15PC00007 Table 4, p.34):

    section      d(dp/dz)   d(H_l)   d(dp/dz) as H_l   combined
    ------------------------------------------------------------
    3.83 in         0.040     0.19             0.092      0.211
    7.83 in         0.006     0.03             0.014      0.033
    11.7 in         0.020     0.05             0.046      0.068

The 3.83 in section's holdup tolerance of +/-0.19 is very wide: on that section
almost nothing can fail the screen, which matters for reading the table below.

## Row 1, where the owner started

    reported            0.10 m, u_sl 0.140, u_sg 0.134 m/s, dp/dz 7.759 kPa/m, H_l 0.99, Bubbly
    dp/dz reported      0.3430 psi/ft
    friction            0.00032 psi/ft
    acceleration        0.00009 psi/ft
    hydrostatic left    0.3426 psi/ft
    H_l implied         0.792
    H_l reported        0.99
    difference          -0.198   tolerance 0.211  -> inside

Put the other way, as the owner did: a holdup of 0.99 implies a hydrostatic
gradient of 9.689 kPa/m,
against the 7.759 kPa/m the same row reports. The row cannot be both.

It passes the screen only because the 3.83 in holdup tolerance is +/-0.19 and
the discrepancy is 0.198. That is the tolerance being useless, not the row being
sound. An independent check settles it, below.

## An independent check on the holdup column alone

How fast would the gas have to move in situ for the reported holdup to be true?
v_g = u_sg / alpha, against the drift-flux value C0 v_m + v_d (C0 1.2,
v_d 0.8 ft/s). This uses neither the gradient column nor any pressure assumption.

    run   ID     u_sl    u_sg   H_l   alpha   v_g needed   drift flux   ratio
    ------------------------------------------------------------------------
       1  0.10   0.140   0.134  0.99   0.010       43.96         1.88   23.40
     126  0.30   0.724   1.782  0.83   0.170       34.39        10.67    3.22
     118  0.30   0.460   1.149  0.72   0.280       13.46         7.13    1.89
      97  0.30   0.090   0.678  0.64   0.360        6.18         3.82    1.62
      75  0.30   0.023   0.687  0.60   0.400        5.63         3.60    1.57

Only 2 rows of 93 need more than twice the drift-flux gas velocity.
Row 1 needs **23 times** it: 43.96 ft/s of in-situ gas in a bubbly flow whose
whole mixture velocity is 0.90 ft/s. That is not a measurement. Row 126 at 3.2x
is the only other one, and it is also the largest negative difference in the
whole set (-0.281). Both holdup values are bad; both gradients look ordinary.

## Distribution of the difference (implied minus reported)

REFRESHED at g2-4. The screen strips friction and acceleration from the reported
gradient before implying a holdup, and the acceleration it strips is now the
march's `dp/dx x (1 - det)` rather than the homogeneous 1/(1-E_k) form the gate
used to carry. It is smaller at high gas rate, so the screen is less aggressive
there: 35 rows fail instead of 39. The screen's own logic did not change, and
neither did the verdict below.

    rows with a reported holdup   93
    mean                          +0.009
    median                        -0.024
    min                           -0.281
    max                           +0.616
    negative (gradient too small) 59 of 93
    outside the uncertainty       35 (37.6%)

## Split by diameter

    ID      n    mean   median    fail   rate   tolerance
    --------------------------------------------------------
    0.10    40  +0.092  +0.062      8    20%   0.211
    0.20    26  -0.042  -0.034     15    58%   0.033
    0.30    27  -0.066  -0.064     12    44%   0.068

## Split by mixture velocity

    v_m bin        n    mean   median    fail   rate
    --------------------------------------------------
        0-2 ft/s    1  -0.198  -0.198      0     0%
        2-5 ft/s   13  -0.016  -0.023      0     0%
       5-15 ft/s   31  -0.009  -0.021      9    29%
        >15 ft/s   48  +0.032  -0.032     26    54%

That split is confounded: the tolerance differs sixfold between sections and the
velocity ranges are not evenly spread across them. The honest cut is both at once.

## Split by mixture velocity WITHIN each diameter

    ID     v_m bin        n     mean   median      min      max
    --------------------------------------------------------------
    0.10         0-2 ft/s    1  -0.198  -0.198   -0.198   -0.198
    0.10         2-5 ft/s    3  -0.006  +0.003   -0.029   +0.006
    0.10        5-15 ft/s   12  +0.062  +0.068   -0.144   +0.208
    0.10         >15 ft/s   24  +0.132  +0.098   -0.189   +0.616
    0.20         2-5 ft/s    3  +0.003  -0.000   -0.011   +0.019
    0.20        5-15 ft/s    7  -0.011  -0.012   -0.049   +0.043
    0.20         >15 ft/s   16  -0.063  -0.066   -0.134   -0.021
    0.30         2-5 ft/s    7  -0.028  -0.027   -0.044   -0.006
    0.30        5-15 ft/s   12  -0.080  -0.070   -0.281   -0.021
    0.30         >15 ft/s    8  -0.078  -0.076   -0.099   -0.056

## Verdict: (iii), something else

Not (i): row 1 is not an isolated outlier in the statistics. 35 of 93 rows, 38%,
fall outside their own reported uncertainty, and 58% of the 0.20 m section does.

Not (ii): the two columns do NOT disagree at low mixture velocity. Below 5 ft/s
they agree about as well as the instruments allow, in every section: mean
differences +0.003 (0.20 m), -0.028 (0.30 m), -0.006 (0.10 m), with nothing
failing the screen. Row 1 is the single row below 2 ft/s and it is the exception
in that band, not the rule.

What is actually there, (iii), has two separate parts:

1. **A velocity-dependent divergence that changes sign with diameter.** As
   mixture velocity rises the two columns pull apart, and they pull apart in
   opposite directions in different sections:

       0.20 m   2-5 ft/s  +0.003    5-15  -0.011    >15  -0.063
       0.30 m   2-5 ft/s  -0.028    5-15  -0.080    >15  -0.078
       0.10 m   2-5 ft/s  -0.006    5-15  +0.062    >15  +0.132

   The two steel-and-large sections drift negative, meaning the reported gradient
   is smaller than the reported holdup can account for. The small PVC section
   drifts positive, the other way. A single physical effect would not flip sign
   with pipe size; this looks like the holdup measurement degrading differently
   in each rig at high rate. Consistent with the method: BOEM M15PC00007 p.35
   obtains holdup by shutting the flow in and weighing the trapped liquid via the
   bottom pressure, which at high gas rate has to fight carry-over and drainage.

2. **Row 1 is separately and simply wrong**, on the holdup column. Its reported
   0.99 needs 23 times the physically available in-situ gas velocity. The
   gradient it reports is ordinary and consistent with its neighbours. Whatever
   0.99 is, it is not a holdup measurement of that run.

## What this changes about scoring

The owner's instruction attached a remedy to (ii). The finding is (iii), but the
consequence for scoring is the same one and I have applied it, flagging the
extension so it can be overruled:

- **gradient is scored on all 131 rows**, unchanged. Nothing in this screen
  impugns the gradient column: it is internally consistent, it varies smoothly
  within each liquid-rate block, and where the two columns disagree it is the
  holdup that is implausible, not the gradient.
- **holdup is scored only on the 58 rows that pass the screen.** 35 of
  93 rows with a reported holdup are excluded, because their two reported
  columns cannot both be true and the model cannot be graded against a number
  the source contradicts. The excluded rows are listed by run number in
  G2-results.md.

Nothing was tuned to the holdup column, and nothing was tuned at all.

## Caveat on the screen itself

The screen uses the closure's own mixture state for the friction and acceleration
terms, so it is not fully independent of the model. At these conditions that
hardly matters: friction and acceleration together are under 2% of the total
gradient on the low-velocity rows where the verdict rests, and row 1's friction
is 0.09% of its gradient. On the high-velocity rows
they are larger and the screen is correspondingly softer there, which if anything
makes the divergence reported above an underestimate.
