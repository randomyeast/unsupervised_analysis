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
        type ``bash setup.bash <your_new_project_name>``
        
        *       ``./experiments/<your_new_project_name>/checkpoints`` will be where the
                model checkpoints will be stored.
        
        *       ``./experiments/<your_new_project_name/log`` will be where the log used
                to visualize the loss in tensorboard will be stored.

*       Next, you should create a copy of the template configuration file available in
        ``./experiments/example_tvae/config.json`` and add the copy to 
        ``./experiments/<your_new_project_name/``. The ``config.json`` has three dictionaries
        within it:
        
        *       ``data_config``: This is passed to the instantiation of a new dataset object.
        	Dataset objects will be explained in the third step. Any additional parameters 
        	needed for loading and preprocessing data should be added here. After you create
                a new dataset object in the next step, you will need to change the ``name``
                parameter here to reflect the new dataset object you created as well as in 
                ``./lib/util/datasets/__init__.py``.

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

++++++++++++++++++++++++++++++++++++++++++++
Structuring the folders containing your data
++++++++++++++++++++++++++++++++++++++++++++
*       In all likelihood, you are learning representations of video data. Much of the 
	code in this repo assumes your data has the following structure:

        .. _file_structure:
	.. image:: images/data_structure.png
	
	where ``vid_1``, ``vid_2``, and ``vid_3`` corresponds to directories which 
	contain the data you are learning representations of. The outputs of the 
	model corresponding to a given video will be stored in that video's directory.	


+++++++++++++++++++++++++++++
Creating a new dataset object
+++++++++++++++++++++++++++++

*       To train any of the model variants, you first need to create
        a new dataset object which inherits from the base module 
        :mod:`TrajectoryDataset <lib.util.datasets.core.TrajectoryDataset>`.
        An example dataset is available here:
        :mod:`MouseV1Dataset <lib.util.datasets.mouse_v1.core.MouseV1Dataset>`
*       The first method from :mod:`TrajectoryDataset <lib.util.datasets.core.TrajectoryDataset>`
        you need to override in your new dataset object is ``load_data(data_config)``.

        *       This function is called to load the trajectories you are interested in 
                embedding and should return a ``np.array`` or ``torch.tensor`` of the
                shape ``[num_trajs, traj_len, num_features]``

        *       There should be a ``root_data_dir`` field in your ``data_config`` that
                is the path to the video directories explained in the previous step.
        
*       The second method you will need to override is the static method ``load_video``  
        which tells :mod:`TrajectoryDataset <lib.util.datasets.core.TrajectoryDataset>`
        how to load a single video's data for your given dataset. The output expected
        for ``load_video`` is ``[num_trajs, traj_len, num_features]`` 

*       The third method you need to override is ``preprocess(data_config, trajectories)``.
        This is where you will do any preprocessing of the data loaded in via ``load_data``.
        This function must also return a ``np.array`` or ``torch.tensor`` of the shape
        ``[num_trajs, traj_len, num_features]``. 

*       The fourth method you need to override is ``postprocess(trajectories)``. If you
        perform any preprocessing of your input data (e.g. decomposition) and you want
        to see what the reconstructions of the original inputs look like, you should
        make this function perform the inverse of whatever your postprocessing steps 
        are.

++++++++++++++++++
Training the model
++++++++++++++++++
-       Running ``python train.py --config_dir ./experiments/<your_experiment_name>`` will
        train the model. It will likely take several hours to complete, depending on your
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
-       You can also generate random reconstructions from one of the videos in ``root_data_dir``
        using ``python plot_random_reconstructions --config_dir 
        ./experiments/<your_experiment_name> --num_reconstructions <some_integer>``. The output
        of this will be stored in a ``random_reconstructions`` folder within your experiment
        folder.
-       All supported models are based on the variational autoencoder and the reconstruction 
        process samples ``eval_config[num_samples]`` embeddings from the inferred posterior, 
        decodes each embedding to get a reconstruction, and then picks the "best" reconstruction
        as the one with the lowest negative-log-likelihood. Increasing ``eval_config[num_samples]``
        will dramatically increase the amount of time it takes for ``reconstruct.py`` to complete,
        but you may get more realistic reconstructions of the input data.

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
