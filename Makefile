lint: ## check style with flake8 and isort
	flake8 viabel
	isort -c viabel

fix-lint: ## fix lint issues using autoflake, autopep8, and isort
	find viabel -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables
	autopep8 --in-place --recursive --aggressive viabel
	isort --atomic viabel
