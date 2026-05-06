Pipeline Overview
=================

spectrex is a Python library for spectral extraction from grism wide-field
slitless spectroscopy (WFSS) images, with a focus on crowded fields such as
those produced by NIRISS on JWST. It encodes the instrument's dispersive
mapping as a sparse forward operator :math:`H`, decomposes stellar spectral
energy distributions onto a PCA basis, and recovers source coefficients by
solving a weighted inverse problem. Two operator/solver pairs are provided —
one based on SciPy sparse matrices for exploration, one based on JAX for
JIT-compiled production runs on crowded scenes.

Pipeline Architecture
---------------------

The following diagram shows the full data-flow from raw inputs to recovered
spectra:

.. graphviz::

   digraph spectrex_pipeline {
       rankdir=LR;
       node [fontname="Helvetica", fontsize=11];

       // Inputs
       conf_files   [label="NIRISS\nconf files",   shape=parallelogram, style=filled, fillcolor="#f0f0f0"];
       eigen_csv    [label="eigenspectra\nCSV",     shape=parallelogram, style=filled, fillcolor="#f0f0f0"];
       dispersed_img [label="dispersed\nimage",     shape=parallelogram, style=filled, fillcolor="#f0f0f0"];

       // Stage 1 — instrument & basis (blue)
       subgraph cluster_stage1 {
           label="Stage 1 — Instrument & Basis";
           style=filled; fillcolor="#eef5ff"; color="#6baed6";
           InstrumentConfig  [label="InstrumentConfig",  style=filled, fillcolor="#cce5ff"];
           EigenspectraBasis [label="EigenspectraBasis", style=filled, fillcolor="#cce5ff"];
       }

       // Stage 2 — operators (green)
       subgraph cluster_stage2 {
           label="Stage 2 — Forward Operator";
           style=filled; fillcolor="#efffef"; color="#74c476";
           op_scipy [label="SciPySparseOperator", style=filled, fillcolor="#d4edda"];
           op_jax   [label="JAXOperator",         style=filled, fillcolor="#d4edda"];
       }

       // Stage 3 — solvers (orange)
       subgraph cluster_stage3 {
           label="Stage 3 — Solver";
           style=filled; fillcolor="#fffbef"; color="#fd8d3c";
           solver_scipy [label="SpectralSolver",    style=filled, fillcolor="#fff3cd"];
           solver_jax   [label="JAXProximalSolver", style=filled, fillcolor="#fff3cd"];
       }

       // Output
       output [label="recovered\nspectra", shape=parallelogram, style=filled, fillcolor="#f0f0f0"];

       // Edges
       conf_files    -> InstrumentConfig;
       eigen_csv     -> EigenspectraBasis;
       InstrumentConfig  -> op_scipy;
       InstrumentConfig  -> op_jax;
       EigenspectraBasis -> op_scipy;
       EigenspectraBasis -> op_jax;
       dispersed_img -> solver_scipy;
       dispersed_img -> solver_jax;
       op_scipy -> solver_scipy;
       op_jax   -> solver_jax;
       solver_scipy -> output;
       solver_jax   -> output;
   }


Forward Model
-------------

spectrex solves the linear system

.. math::

   \mathbf{f} = H\,\mathbf{a} + \boldsymbol{\varepsilon},
   \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(0,\,\mathrm{diag}(\boldsymbol{\sigma}^2))

where:

- :math:`\mathbf{f}` — vectorised dispersed detector image (the observation)
- :math:`H` — the forward operator, built from the instrument configuration and the PCA spectral basis
- :math:`\mathbf{a}` — PCA coefficient vector (:math:`K \times M`, one row per source group)
- :math:`\boldsymbol{\varepsilon}` — pixel noise (modelled by :class:`~spectrex.NoiseModel`)

Recovering :math:`\mathbf{a}` from :math:`\mathbf{f}` is the spectral
extraction problem.


Which Solver Should I Use?
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 30 40

   * - Scenario
     - Operator
     - Solver
     - Notes
   * - Sparse field, few sources (≲20)
     - :class:`~spectrex.SciPySparseOperator`
     - :class:`~spectrex.SpectralSolver`
     - Fast to build; unconstrained LSQR / LSMR; suited to exploration
   * - Crowded field, many sources
     - :class:`~spectrex.JAXOperator`
     - :class:`~spectrex.JAXProximalSolver`
     - Compact trace layout; group-L1 FISTA with adaptive restart; JIT-compiled


.. seealso::

   - :doc:`instrument_basis` — loading configuration files and the PCA basis
   - :doc:`operators` — building the forward operator
   - :doc:`solvers` — choosing and tuning a solver
   - :doc:`/content/mock_example` — end-to-end worked example
   - :doc:`/content/comparison_solver_accuracy` — LSQR vs FISTA accuracy benchmark
   - :doc:`/content/comparison_computational` — runtime and memory comparison
   - :doc:`/content/analysis_rmse_vs_density` — RMSE as a function of source density
