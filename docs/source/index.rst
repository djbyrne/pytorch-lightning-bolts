.. PyTorchLightning-Bolts documentation master file, created by
   sphinx-quickstart on Wed Mar 25 21:34:07 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyTorch-Lightning-Bolts documentation
=====================================
.. toctree::
   :maxdepth: 1
   :name: start
   :caption: Start here

   introduction_guide

.. toctree::
   :maxdepth: 2
   :name: callbacks
   :caption: Callbacks

   callbacks
   info_callbacks

.. toctree::
   :maxdepth: 2
   :name: datamodules
   :caption: DataModules

   datamodules
   sklearn_datamodule
   vision_datamodules

.. toctree::
   :maxdepth: 2
   :name: losses
   :caption: Losses

   losses

.. toctree::
   :maxdepth: 2
   :name: loggers
   :caption: Experimental Loggers

   loggers

.. toctree::
   :maxdepth: 2
   :name: models
   :caption: Models

   autoencoders
   classic_ml
   convolutional
   gans
   reinforce_learn
   self_supervised_models

.. toctree::
   :maxdepth: 2
   :name: transforms
   :caption: Data Processing

   transforms
   self_supervised_utils
   semi_sup_processing

.. toctree::
   :maxdepth: 2
   :name: ssl
   :caption: Learning Tasks

   vision_tasks


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. This is here to make sphinx aware of the modules but not throw an error/warning
.. toctree::
   :hidden:

   readme
   api/pl_bolts.callbacks
   api/pl_bolts.datamodules
   api/pl_bolts.metrics
   api/pl_bolts.models
   api/pl_bolts.callbacks
   api/pl_bolts.losses
   api/pl_bolts.loggers
   api/pl_bolts.optimizers
   api/pl_bolts.transforms
   api/pl_bolts.utils
