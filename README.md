# projeto-mc906
Projeto MC906

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/andrac-23/projeto-mc906/)


## Instructions


### Cloning this repository and keeping up-to-date

`git clone git@github.com:andrac-23/projeto-mc906.git`

`cd projeto-mc906`

Before making changes:

`git pull`

### Installing environment

Follow the instructions [here](https://docs.anaconda.com/free/miniconda/index.html#quick-command-line-install) to install Miniconda.

Once Miniconda is installed, create environment and install depedencies:

`conda create -n mc906 python=3.12 --file requirements.txt --yes`

Activate environment:

`conda activate mc906`

After you're done and want to leave the environment:

`conda deactivate`

### Adding new dependency to environment

`conda install {dependency}`

`conda list -e | grep {dependency} >> requirements.txt`
