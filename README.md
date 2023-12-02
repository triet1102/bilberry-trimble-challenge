# Introduction
This repository contains the solution for technical exercise for AI Engineer position at Trimble / Bilberry.
Your goal is to create a two class classifier : **Field** & **Road** using the available data [here](https://drive.google.com/file/d/1pOKhKzIs6-oXv3SlKrzs0ItHI34adJsT/view?usp=sharing).

# Results

# Directory structure

# Installation and running
## Install Pyenv
```bash
## get the shell name
SHELL_NAME=${0#-}
SHELL_CONFIG=~/.${SHELL_NAME}rc

## install pyenv
brew install pyenv

## configure the shell
echo >>$SHELL_CONFIG 'eval "$(pyenv init --path; pyenv init -; pyenv virtualenv-init -)"'

## reload the shell's configuration
source $SHELL_CONFIG
```

## Install pre-commit hooks
```bash
pre-commit install --allow-missing-config
```

## Install dependencies

## Auto activate python environment
```bash
source .venv/bin/activate
unset PS1
```
