import timeit
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs
import ase.io
import pathlib
from nequip.ase import NequIPCalculator

import torch

_WARMUP_ITER = 1000
_RUN_ITER = 5000
STRUCTURE_DIR = pathlib.Path(__file__).parent / ".." / "structure"

def get_structure(supercell=1):
    """
    Load the Li6PS5Cl structure and create supercell if requested.

    Args:
        supercell: Size of supercell (int or tuple). Default is 1 (no supercell).

    Returns:
        ase.Atoms: The structure with supercell applied
    """
    structure_path = STRUCTURE_DIR / "Li6PS5Cl_conventional_cell_POSCAR"
    atoms = ase.io.read(structure_path, format="vasp")

    if supercell != 1:
        if isinstance(supercell, int):
            supercell = (supercell, supercell, supercell)
        atoms = atoms.repeat(supercell)

    return atoms


def benchmark_md(atoms, warmup_iter=_WARMUP_ITER, run_iter=_RUN_ITER, verbose=True):
    """
    Benchmark an ASE calculator by timing MD force calculations.

    Args:
        atoms: ASE Atoms object with calculator attached
        warmup_iter: Number of warmup iterations
        run_iter: Number of benchmark iterations per run
        verbose: Whether to print timing information

    Returns:
        # TODO fix
        (T1) in timesteps per second
    """
    # Initialize velocities with Maxwell-Boltzmann distribution at 300K
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, force_temp=True)
    # run NVE
    dyn = VelocityVerlet(
        atoms,
        timestep=1.0 * fs,
        loginterval=1000000,
    )

    # warmup
    dyn.run(warmup_iter)

    torch.cuda.synchronize()

    # first benchmark
    start = timeit.default_timer()
    dyn.run(run_iter)
    end = timeit.default_timer()
    T = run_iter / (end - start)

    torch.cuda.synchronize()

    if verbose:
        print(f"T: {T:>10.5f} timesteps/s")

    return T


def benchmark_calculator(calculator):
    """
    Benchmark an ASE calculator across different supercell sizes.

    Args:
        calculator: ASE calculator to benchmark

    Returns:
        dict: Results with supercell sizes as keys and throughput data (timesteps/s) as values
    """
    results = {}

    for supercell in [1, 2, 3]:
        atoms = get_structure(supercell=supercell)
        atoms.calc = calculator
        T = benchmark_md(
            atoms, warmup_iter=_WARMUP_ITER, run_iter=_RUN_ITER, verbose=False
        )
        results[supercell] = {
            "num_atoms": len(atoms),
            "T": T,
        }

    print("\n=== Summary ===")
    for supercell, data in results.items():
        print(
            f"Supercell {supercell}: {data['num_atoms']:>4d} atoms, {data['T']:>10.5f} timesteps/s"
        )

    return results

def main():
    calculator = NequIPCalculator.from_compiled_model(
        compile_path= "mir-group__NequIP-OAM-L__0.1.nequip.pt2",
        device="cuda"
    )


    res = benchmark_calculator(calculator)


if __name__ == '__main__':
    main()