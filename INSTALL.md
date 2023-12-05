# Instruction to install and run the code (for macOS)

### Clone the repository
```bash
cd ~
git clone git@github.com:triet1102/bilberry-trimble-challenge.git
cd bilberry-trimble-challenge
```

### Install `Pyenv`
```bash
SHELL_NAME=${0#-}
SHELL_CONFIG=~/.${SHELL_NAME}rc
brew install pyenv pyenv-virtualenv
echo >>$SHELL_CONFIG 'eval "$(pyenv init --path; pyenv init -; pyenv virtualenv-init -)"'
source $SHELL_CONFIG
brew install openssl readline sqlite3 xz zlib
xcode-select --install
```
### Install pre-commit hooks
```bash
pre-commit install --allow-missing-config
```

### Install python 3.10.9 and the virtual environment
```bash
# install python
CONFIGURE_OPTS=--enable-shared pyenv install 3.10.9
pyenv shell 3.10.9

# create virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements/requirements.txt --no-deps
```


# Bonus
### Auto activate python environment when cd into the project with `direnv`
```bash
brew install direnv
echo >> $SHELL_CONFIG 'eval "$(direnv hook $SHELL_NAME)"'

# allow config in .envrc file
direnv allow
```

Every time you `cd` into the project, the virtual environment will be activated automatically. It should print something like this:
```bash
direnv: loading ~/bilberry-trimble-challenge/.envrc
direnv: export +VIRTUAL_ENV +VIRTUAL_ENV_PROMPT ~PATH
```

### Install `pip-compile` and create lock dependencies
```bash
pip install pip-tools
pip-compile --output-file=requirements/requirements.txt requirements/requirements.in
```
