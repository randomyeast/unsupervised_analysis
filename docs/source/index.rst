.. Kennedy lab unsupervised analysis tools documentation master file, created by
   sphinx-quickstart on Thu Jul 14 13:46:27 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation for the Kennedy lab's unsupervised analysis tools!
===============================================================================

++++++++++++++++++++++++++++++++++++++++++++++++++
Overview
++++++++++++++++++++++++++++++++++++++++++++++++++
-       These pages contain documentation relevant for unsupervised
        analysis methods used in Ann Kennedy's lab at Northwestern
        University.
-       There are currently three models available for use: the
        Trajectory Variational Autoencoder (TVAE), TREBA, and Vector
        Quantized TREBA with triplet loss (VQ-TREBA).
-       To generate clusters corresponding to behavioral motifs,
        the TVAE and TREBA both require some post-hoc clustering
        of the generated embeddings and VQ-TREBA does not.
-       The goal of this documentation is both to explain how each
        model variant works and make it as easy as possible to adapt
        the models to work on new datasets.
-       The documentation for each model is split up into a reference
	document, meant for those implementing / modifying the models,
	as well as a section with more explicit mathematical details.
-       Any suggestions are more than welcome.


Quickstart guide
==================================================

+++++++++++++++++++++++++++
Setting up a new experiment
+++++++++++++++++++++++++++

*       To setup the directory structure for a new experiment, in the terminal
        type ``bash setup.bash <your_new_project_name>``, replacing ``<your_new_project_name>``
        with the name of your new project.
        
        *       ``./experiments/<your_new_project_name>/checkpoints`` will be where the
                model checkpoints will be store.
        
        *       ``./experiments/<your_new_project_name/log`` will be where the log used
                to visualize the loss in tensorboard will be stored.

*       Next, you should create a copy of the template configuration file available in
        ``./experiments/example_tvae/config.json`` and add the copy to 
        ``./experiments/<your_new_project_name/``. The ``config.json`` has three dictionaries
        within it:
        
        *       ``data_config``: This is passed to the instantiation of a new dataset object
                which will be explained in the next step. Any additional parameters needed
                for loading and preprocessing data should be added here. After you create
                a new dataset object in the next step, you will need to change the ``name``
                parameter here to reflect the new dataset object you created.

        *       ``model_config``: This is used to pick the model architecture being used (e.g.
                ``TVAE`` in the example configuration) and to configure the dimensions of the
                different layers in the model. If you're creating a new dataset object, the
                main parameter you should update here is ``state_dim`` which represents the 
                dimensionality of each frame of each trajectory in the inputs. For additional
                information on what the different keys represent, check the model documentation. 

        *       ``train_config``: This is used to configure training parameters such as the 
                batch size, device to use for training, how many epochs etc.
                
                *       One important thing to recognize is that ``num_epochs`` is a list.
                        This is because some models use staged training. The current support
                        for and explanation of different stages of training is in the
                        documentation for each model. 

++++++++++++++++++++++++
Structuring your folders
++++++++++++++++++++++++
*       In all likelihood, you are learning representations of video data. It will be easiest
        to adapt the existing code and structure the outputs of the model, if you make a 
        root directory with sub folders for each video. The model training code does not 
        require this structure, but the code for generating embeddings and reconstruction does.


+++++++++++++++++++++++++++++
Creating a new dataset object
+++++++++++++++++++++++++++++

*       To train any of the model variants, you first need to create
        a new dataset object which inherits from the base module 
        :mod:`TrajectoryDataset <lib.util.datasets.core.TrajectoryDataset>`.
        An example dataset is available 
        :mod:`MouseV1Dataset <lib.util.datasets.mouse_v1.core.MouseV1Dataset>`
*       The first method from :mod:`TrajectoryDataset <lib.util.datasets.core.TrajectoryDataset>`
        you need to override in your new dataset object is ``load_data(data_config)``.

        *       This function is called to load the trajectories you are interested in 
                embedding and should return a ``np.array`` or ``torch.tensor`` of the
                shape ``[num_trajs, traj_len, num_features]``
        
        *       Any paths to files storing data can be passed to ``load_data`` using
                the ``data_config`` dictionary in ``config.json`` in 
                your experiment folder. For example, 
                :mod:`MouseV1Dataset <lib.util.datasets.mouse_v1.core.MouseV1Dataset>`
                object expects a key called ``root_data_dir`` to be in ``data_config``. 

*       The second method you need to override is ``preprocess(data_config, trajectories)``.
        This is where you will do any preprocessing of the data loaded in via ``load_data``.
        This function must also return a ``np.array`` or ``torch.tensor`` of the shape
        ``[num_trajs, traj_len, num_features]``. 

++++++++++++++++++
Training the model
++++++++++++++++++
-       Running ``python train.py --config_dir ./experiments/<your_experiment_name>`` will
        train the model. It will likely take several hours to converge, depending on your
        hardware and how much data you are using.
-       Once the model has completed training, there will be a directory titled ``stage_<x>``
        within the ``./experiments/<your_experiment_name>/`` directory, for each stage of 
        training you have. Within each of these directories, there will be a file called
        ``best.pt`` which will be the checkpoint with the lowest negative-log-likelihood
        on the evaluation set for that stage.
-       To monitor the model's training progress with tensorboard, in the terminal type:
        ``tensorboard --logdir=./experiments/example_tvae`` and click on link to a port
        on ``localhost`` that is printed to the terminal. To monitor the training process
        across machines see this note <link>.

++++++++++++++++++++++++++
Generating reconstructions
++++++++++++++++++++++++++
-       Running ``python reconstruct.py --config_dir ./experiments/<your_experiment_name>`` will
        generate reconstructions for all videos in the ``root_data_dir`` specified
        in ``eval_config`` and outputs them to their respective folders.


++++++++++++++++++++++++++++++++++++
Generating embeddings using the TVAE
++++++++++++++++++++++++++++++++++++
-       Running ``python embed.py --config_dir ./experiments/<your_experiment_name>`` will
        iterate through and generate embeddings for all of the video subdirectories in the 
        ``root_data_dir`` specified in ``eval_config``
-       All of the models currently supported are based on the variational autoencoder
        architecture and use the mean of the inferred posterior for each input as the 
        embedding for each trajectory.

++++++++++++++++++++++
TREBA quickstart guide
++++++++++++++++++++++
-       Under construction


+++++++++++++++++++++++++
VQ-TREBA quickstart guide
+++++++++++++++++++++++++
-       Under construction


++++++++++++++++++++
Module documentation
++++++++++++++++++++

.. toctree::
   :maxdepth: 4

   lib

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
