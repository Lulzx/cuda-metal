# MR !6137 water-box reproduction

These inputs reconstruct the two shapes published in the overview of GROMACS
MR !6137. The MR does not publish its complete MDP or equilibration procedure,
so this is a transparent reconstruction, not a claim to possess the author's
original input files.

Use one GROMACS source revision for every build. In the recorded comparison it
was `c7fc4ef64a23f2fe4795d6342af5bcb769d9ca9a`.

## 98,319 atoms / 84 cubed PME

```bash
mkdir -p work/small
cp demos/gromacs/mr6137-water/topol.top work/small/topol.top
<native-gmx> solvate -cs spc216.gro -box 10 10 10 \
  -o work/small/solvated.gro -p work/small/topol.top
# Must print 32,773 SOL molecules / 98,319 atoms.

<native-gmx> grompp -f demos/gromacs/mr6137-water/em-small.mdp \
  -c work/small/solvated.gro -p work/small/topol.top -o work/small/em.tpr
<native-gmx> mdrun -s work/small/em.tpr -deffnm work/small/em \
  -ntmpi 12 -ntomp 1 -nb cpu -pme cpu

<native-gmx> grompp -f demos/gromacs/mr6137-water/equil-small.mdp \
  -c work/small/em.gro -p work/small/topol.top -o work/small/equil.tpr
<native-gmx> mdrun -s work/small/equil.tpr -deffnm work/small/equil \
  -ntmpi 12 -ntomp 1 -nb cpu -pme cpu

<native-gmx> grompp -f demos/gromacs/mr6137-water/bench-small.mdp \
  -c work/small/equil.gro -p work/small/topol.top -o work/small/bench.tpr
```

Run one conditioning CuMetal process, discard it, then alternate at least five
independent warm processes per backend:

```bash
<gmx> mdrun -s work/small/bench.tpr -deffnm <output-prefix> \
  -ntmpi 1 -ntomp 12 -nb gpu -pme gpu -pmefft gpu -noconfout
```

For CuMetal, put the Release runtime directory first in `DYLD_LIBRARY_PATH`
and use one persistent `CUMETAL_CACHE_DIR` across conditioning and warm runs.
Do not set `CUMETAL_TRACE_GPU` during timing because per-kernel timing
intentionally disables command-buffer batching.

Build the correctness TPR from the same equilibrated coordinates with
`correctness-small.mdp`, run both backends, then gate the logs:

```bash
python3 demos/gromacs/gate.py native/run.log cumetal/run.log \
  --label mr6137-water-small
```

## 1,005,375 atoms / 192 cubed PME

`gmx solvate -box 21.70 21.70 21.70` deterministically produced 335,125
waters / 1,005,375 atoms with the recorded GROMACS revision. The checked-in
`bench-large-structural.mdp` reproduces the structural stress run only; perform
and disclose a full large-box equilibration before treating it as a physical
production benchmark.
