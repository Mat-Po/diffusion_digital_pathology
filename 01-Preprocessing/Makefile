# FOLDERS
VENV := venv
PROJECT_NAME := diffusion

# PROGRAMS AND FLAGS
PYTHON := python3
PYFLAGS := -m
#MAIN := xxxxx
#MAIN_FLAGS :=
DATA_GENERATION := data_generation
QUESTIONNAIRE_GENERATION := questionnaire_generation
PIP := pip

# COLORS
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
NONE := \033[0m

# COMMANDS
ECHO := echo -e
MKDIR := mkdir -p
CD := cd
CP := cp

# RULES
.PHONY: help env install generate-tiles normalize-tiles generate-previews generate-questionnaires

help:
	@$(ECHO) '$(YELLOW)Makefile help$(NONE)'
	@$(ECHO) " \
	* env 				: generates the virtual environment using venv\n \
	* install			: install the requirements listed in $(YELLOW)requirements.txt$(NONE)\n\n \
	$(GREEN)Data Generation Module$(NONE)\n \
	* generate-tiles		: generate tiles from slides listed in $(YELLOW)data_generation/settings.json$(NONE)\n \
	* normalize-tiles		: normalize tiles as specified in $(YELLOW)data_generation/settings.json$(NONE)\n \
	* generate-previews		: generate slides with possible extracted tiles outlined\n\n \
	$(GREEN)Questionnaire Generation Module$(NONE)\n \
	* generate-questionnaires	: generates questions for questionnaires that have a structure specified in  $(YELLOW)questionnaire_generation/settings.json$(NONE)"
env:
	@$(ECHO) '$(BLUE)Creating the virtual environment..$(NONE)'
	@$(MKDIR) $(VENV)
	@$(eval PYTHON_VERSION=$(shell $(PYTHON) --version | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]' | cut -f1,2 -d'.'))
	@$(PYTHON_VERSION) -m venv $(VENV)/$(PROJECT_NAME)
	@$(ECHO) '$(BLUE)Done$(NONE)'

install:
	@$(ECHO) '$(BLUE)Installing requirements..$(NONE)'
	@$(PIP) install -r requirements.txt
	@$(ECHO) '$(BLUE)Done$(NONE)'

generate-tiles:
	@$(ECHO) '$(BLUE)Generating tiles..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(DATA_GENERATION) 
	@$(ECHO) '$(BLUE)Tiles Generated$(NONE)'

normalize-tiles:
	@$(ECHO) '$(BLUE)Normalizing tiles..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(DATA_GENERATION) --normalize
	@$(ECHO) '$(BLUE)Tiles Normalized$(NONE)'

generate-previews:
	@$(ECHO) '$(BLUE)Generating previews..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(DATA_GENERATION) --preview
	@$(ECHO) '$(BLUE)Previews generated$(NONE)'

generate-questionnaires:
	@$(ECHO) '$(BLUE)Generating questionnaires..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(QUESTIONNAIRE_GENERATION)
	@$(ECHO) '$(BLUE)Questionnaires generated$(NONE)'