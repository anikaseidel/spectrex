Instrument Configuration & Spectral Basis
==========================================

Building the forward operator :math:`H` requires two inputs: a description of
how the grism disperses light across the detector (the *instrument
configuration*), and a compact representation of the source spectral energy
distributions in terms of a PCA basis (the *eigenspectra basis*). The classes
on this page provide both.


InstrumentConfig
----------------

:class:`~spectrex.InstrumentConfig` reads a NIRISS WFSS configuration file
and wraps the per-order :class:`~spectrex.instrument.GrismTrace` objects that
map (x, y) source positions to detector pixel positions, wavelengths, and
sensitivity curves.  It is a plain class (not a dataclass) because
:class:`~spectrex.instrument.GrismTrace` is not hashable.

Construct it with :meth:`~spectrex.InstrumentConfig.from_files`, passing the
path to the configuration file and the wavelength range over which extraction
should be performed.  Once built, call
:meth:`~spectrex.InstrumentConfig.get_trace` for any source at detector
position ``(x, y)`` and a diffraction order to obtain the arrays of pixel
positions, wavelengths, and sensitivity values needed to populate the forward
operator.


EigenspectraBasis
-----------------

:class:`~spectrex.EigenspectraBasis` stores a truncated PCA decomposition of
a library of stellar SEDs.  It is a frozen dataclass — all arrays are marked
read-only via :func:`numpy.ndarray.setflags` — so it can be safely shared
between operators and solvers without defensive copying.

Load it from a CSV of eigenspectra with
:meth:`~spectrex.EigenspectraBasis.from_csv`, specifying the wavelength range
that matches the instrument configuration.  A given source's SED is
approximated as :math:`\mathrm{SED}(\lambda) = \mathbf{E}(\lambda)\,\mathbf{c}`,
where :math:`\mathbf{E}` is the matrix of eigenspectra components and
:math:`\mathbf{c}` is the coefficient vector.

:meth:`~spectrex.EigenspectraBasis.reconstruct` recovers the SED from
coefficients; :meth:`~spectrex.EigenspectraBasis.broadband_image` integrates
the reconstructed SED over the sensitivity curves of an image filter to
produce a predicted broadband map; :meth:`~spectrex.EigenspectraBasis.integrated_weights`
returns the pre-integrated weight matrix used internally by the operators.


Data Flow
---------

.. graphviz::

   digraph instrument_basis {
       rankdir=LR;
       node [fontname="Helvetica", fontsize=11];

       subgraph cluster_instrument {
           label="InstrumentConfig";
           style=filled; fillcolor="#eef5ff"; color="#6baed6";

           conf_file [label="conf file", shape=parallelogram, style=filled, fillcolor="#f0f0f0"];
           from_files [label="from_files(\nwav_min, wav_max)", shape=box, style=filled, fillcolor="#cce5ff"];
           IC [label="InstrumentConfig", style=filled, fillcolor="#cce5ff"];
           get_trace [label="get_trace(x, y, order)", shape=box, style=filled, fillcolor="#cce5ff"];
           trace_out [label="(positions,\nwavelengths,\nsensitivities)", shape=parallelogram, style=filled, fillcolor="#f0f0f0"];

           conf_file -> from_files -> IC -> get_trace -> trace_out;
       }

       subgraph cluster_basis {
           label="EigenspectraBasis";
           style=filled; fillcolor="#efffef"; color="#74c476";

           eigen_csv [label="eigenspectra\nCSV", shape=parallelogram, style=filled, fillcolor="#f0f0f0"];
           from_csv  [label="from_csv(\nwav_min, wav_max)", shape=box, style=filled, fillcolor="#d4edda"];
           EB [label="EigenspectraBasis", style=filled, fillcolor="#d4edda"];
           reconstruct [label="reconstruct(c)", shape=box, style=filled, fillcolor="#d4edda"];
           bb_image    [label="broadband_image(\nc, image_shape)", shape=box, style=filled, fillcolor="#d4edda"];
           sed_out [label="SED\n(n_wav,)", shape=parallelogram, style=filled, fillcolor="#f0f0f0"];
           img_out [label="broadband map\n(H, W)", shape=parallelogram, style=filled, fillcolor="#f0f0f0"];

           eigen_csv -> from_csv -> EB;
           EB -> reconstruct -> sed_out;
           EB -> bb_image    -> img_out;
       }
   }


API Reference
-------------

.. autoclass:: spectrex.InstrumentConfig
   :members:
   :show-inheritance:

.. autoclass:: spectrex.EigenspectraBasis
   :members:
   :show-inheritance:


.. seealso::

   - :doc:`/content/mock_example` — end-to-end example loading config files and building a forward operator
   - :doc:`operators` — how :class:`~spectrex.InstrumentConfig` and :class:`~spectrex.EigenspectraBasis` feed into the forward operator
