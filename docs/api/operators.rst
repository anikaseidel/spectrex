Forward Operators
=================

The forward operator :math:`H` encodes the mapping from source spectral
coefficients :math:`\mathbf{a}` to detector pixel values :math:`\mathbf{f} = H\mathbf{a}`.
It is built once from an :class:`~spectrex.InstrumentConfig` and an
:class:`~spectrex.EigenspectraBasis`, then passed to a solver.

spectrex ships two concrete operator implementations, both satisfying
:class:`~spectrex.ForwardOperatorProtocol`.  Any class that implements
``apply`` and ``apply_adjoint`` with matching signatures can be substituted
transparently.


ForwardOperatorProtocol
-----------------------

:class:`~spectrex.ForwardOperatorProtocol` is a
:func:`~typing.runtime_checkable` :class:`~typing.Protocol` that defines the
interface every operator must satisfy.  The two required methods are:

- ``apply(a)`` — computes :math:`H\mathbf{a}`; takes a coefficient array of
  shape ``(K, M)`` and returns a detector array of shape ``image_shape``.
- ``apply_adjoint(f)`` — computes :math:`H^\top \mathbf{f}`; the transpose
  operation used by iterative solvers.

Both ``n_coefficients`` and ``image_shape`` must be properties so that solvers
can allocate correctly sized buffers without inspecting the operator internals.


SciPySparseOperator
-------------------

:class:`~spectrex.SciPySparseOperator` stores :math:`H` as a CSR sparse
matrix.  Build it with :meth:`~spectrex.SciPySparseOperator.build`, which
loops over sources and orders, calling
:meth:`~spectrex.InstrumentConfig.get_trace` and the basis
:meth:`~spectrex.EigenspectraBasis.integrated_weights` to fill the matrix
entries.  The resulting matrix is K-independent in *structure* — all sources
and all PCA components are packed into a flat coefficient vector.

This operator is best suited to sparse fields (fewer than ~20 sources) or to
exploratory analysis where quick iteration matters more than peak throughput.
:meth:`~spectrex.SciPySparseOperator.save` / :meth:`~spectrex.SciPySparseOperator.load`
serialise and restore the matrix from disk so expensive builds need not be
repeated.


JAXOperator
-----------

:class:`~spectrex.JAXOperator` uses a *compact trace layout* that avoids
storing a full :math:`N_\text{pix} \times K` matrix:

- ``trace_indices[K, O, L]`` — int32 array of detector pixel indices,
  shape ``(n_sources, n_orders, n_lambda)``
- ``weights[O, L, M]`` — float32 array of sensitivity-weighted basis values,
  shape ``(n_orders, n_lambda, n_components)``

Pixel index arithmetic is vectorised by JAX and JIT-compiled, so
:meth:`~spectrex.JAXOperator.apply` and
:meth:`~spectrex.JAXOperator.apply_adjoint` run at near-hardware throughput on
both CPU and GPU.  Out-of-bounds wavelengths are routed to a *ghost pixel* at
index ``n_pix``; the ghost pixel value is discarded before returning the
detector image, ensuring no index-out-of-range errors from JAX's
:func:`jax.numpy.at` semantics.

:meth:`~spectrex.JAXOperator.n_active` reports how many sources have at least
one valid trace pixel; :meth:`~spectrex.JAXOperator.n_components` reports the
number of PCA components.


H-matrix Schematic
-------------------

The diagram below illustrates schematically how ``apply`` accumulates source
coefficients onto detector pixels.  Not all :math:`K \times N_\text{pix}`
edges are drawn; three representative source groups and six pixels are shown:

.. graphviz::

   digraph hmatrix {
       rankdir=LR;
       node [fontname="Helvetica", fontsize=11];
       splines=polyline;

       // Source coefficient nodes
       a0 [label="a₀", shape=ellipse, style=filled, fillcolor="#d4edda"];
       a1 [label="a₁", shape=ellipse, style=filled, fillcolor="#d4edda"];
       a2 [label="a₂", shape=ellipse, style=filled, fillcolor="#d4edda"];

       // Pixel nodes
       p0 [label="p₀", shape=box, style=filled, fillcolor="#fff3cd"];
       p1 [label="p₁", shape=box, style=filled, fillcolor="#fff3cd"];
       p2 [label="p₂", shape=box, style=filled, fillcolor="#fff3cd"];
       p3 [label="p₃", shape=box, style=filled, fillcolor="#fff3cd"];
       p4 [label="p₄", shape=box, style=filled, fillcolor="#fff3cd"];
       p5 [label="p₅", shape=box, style=filled, fillcolor="#fff3cd"];

       // Weighted edges (representative)
       a0 -> p0 [label="w₀₀"];
       a0 -> p1 [label="w₀₁"];
       a0 -> p2 [label="w₀₂"];
       a1 -> p1 [label="w₁₁"];
       a1 -> p3 [label="w₁₃"];
       a1 -> p4 [label="w₁₄"];
       a2 -> p2 [label="w₂₂"];
       a2 -> p4 [label="w₂₄"];
       a2 -> p5 [label="w₂₅"];
   }

Each edge label :math:`w_{kp}` is the sensitivity-weighted sum of the PCA
basis integrated over the trace of source group *k* at detector pixel *p*.
:meth:`~spectrex.JAXOperator.apply` performs this scatter–add; ``apply_adjoint``
performs the corresponding gather.


API Reference
-------------

.. py:class:: spectrex.ForwardOperatorProtocol

   Protocol for the grism dispersion operator :math:`H`.

   Any object satisfying this protocol can be passed to
   :class:`~spectrex.SpectralSolver` or :class:`~spectrex.JAXProximalSolver`.
   Conformance is structural — no inheritance from this class is required.
   Use ``isinstance(obj, ForwardOperatorProtocol)`` to check at runtime.

   .. py:attribute:: image_shape
      :type: tuple[int, int]

      ``(n_rows, n_cols)`` of the detector image.

   .. py:attribute:: n_coefficients
      :type: int

      Total length of the flattened coefficient vector
      (``n_sources * n_components`` for :class:`~spectrex.JAXOperator`).

   .. py:method:: apply(a_tilde)

      Forward pass: :math:`H\,\mathbf{a}`.

      :param a_tilde: Coefficient vector, shape ``(n_coefficients,)``.
      :type a_tilde: numpy.ndarray
      :returns: Flattened dispersed image, shape ``(n_rows * n_cols,)``.
      :rtype: numpy.ndarray

   .. py:method:: apply_adjoint(f)

      Adjoint pass: :math:`H^\top \mathbf{f}`.

      :param f: Flattened dispersed image, shape ``(n_rows * n_cols,)``.
      :type f: numpy.ndarray
      :returns: Coefficient vector, shape ``(n_coefficients,)``.
      :rtype: numpy.ndarray

.. autoclass:: spectrex.SciPySparseOperator
   :members:
   :exclude-members: image_shape, n_coefficients
   :show-inheritance:

.. autoclass:: spectrex.JAXOperator
   :members:
   :exclude-members: image_shape, n_coefficients
   :show-inheritance:


.. seealso::

   - :doc:`instrument_basis` — building the inputs to the operator
   - :doc:`solvers` — passing the operator to a solver
   - :doc:`/content/comparison_computational` — memory footprint and runtime benchmarks for both operators
