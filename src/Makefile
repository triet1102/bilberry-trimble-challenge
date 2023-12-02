VENV_PATH ?= .venv
VENV_PATH := $(abspath $(VENV_PATH))
VENV_COMMAND ?= python -m venv
VENV_BIN_PATH = $(if $(VENV_PATH),$(VENV_PATH)/bin/)
VENV_ACTIVATE = $(if $(VENV_BIN_PATH),. $(VENV_BIN_PATH)activate &&)
