# Addendum

To setup your repository and experiments before running the scripts, check the following steps.

## Staging the Data

1. **Download**: The accompanying data set, *Moral Stories*, can be downloaded from a shared Google Drive file [here](https://tinyurl.com/moral-stories-data). This file should be in the same file depth as the cloned repository with the source code.

	```
	moral_stories
	moral_stories_datasets.tar.xz
	```

2. **Extract**: The data is provided as a file `moral_stories_datasets.tar.xz`, which you can extract using the following command on many Linux-based systems with `tar xf moral_stories_datasets.tar.xz`.

	```
	moral_stories
	moral_stories_datasets.tar.xz
	moral_stories_datasets
	```

3. **Rename folder**: The extracted folder is called `moral_stories_datasets` but to more closely align with the provided scripts you will need to rename it to `data` with `mv -v moral_stories_datasets data`.

	```
	moral_stories
	moral_stories_datasets.tar.xz
	data
	```


4. **Re-structure data**: The renamed `data` folder looks as follows below.

	```
	moral_stories
	moral_stories_datasets.tar.xz
	data
	├──classification
	│   ├──action
	│   ├──action+context
	│   ├──action+context+consequence
	│   ├──action+norm
	│   ├──consequence+action
	│   └──consequence+action+context
	└──generation
		├──action|context
		├──action|context+consequence
		├──consequence|action
		├──consequence|action+context
		├──norm|actions
		├──norm|actions+context
		└──norm|actions+context+consequences
	```

	The provided code in this repository to run experiments expects the following directory structure instead.

   	```
   	moral_stories
	moral_stories_datasets.tar.xz
	data
	├──action_cls
	├──action+context_cls
	├──action+context+consequence_cls
	├──action+norm_cls
	├──consequence+action_cls
	├──consequence+action+context_cls
	├──action|context_gen
	├──action|context+consequence_gen
	├──consequence|action_gen
	├──consequence|action+context_gen
	├──norm|actions_gen
	├──norm|actions+context_gen
	└──norm|actions+context+consequences_gen
	```

	This can be restructured using the following commands. These commands should be run in the `data` folder.
	
	For the classification tasks: 
	```
	for i in $(ls -d classification/*); do echo $i; ln -s $i "`basename ${i}`_cls"; done
	```

	For the generation tasks: 
	
	```
	for i in $(ls -d generation/*); do echo $i; ln -s $i "`basename ${i}`_gen"; done
	```

## Running Experiments

At this point the default bash scripts in this folder should be able to run. Each script is pre-populated with hyper-parameters and experiment settings. The major flags to edit would be `--model_type` or `--task_name` to try the different models and tasks. For classification or generation there are different bash scripts in this folder altogether. 
