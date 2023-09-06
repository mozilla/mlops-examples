
# mlops-platform-spike-library
A collection of experimental integrations for mozilla ML projects.


## Getting started with conda environments (used by Metaflow)

- Install `conda`
  - On mac run `brew install conda`
- Add the environment variable `CONDA_CHANNELS=anaconda,conda-forge`
- Run `conda env create -f environment-mlops.yml`
- Run `conda activate mlops-metaflow-1`

### Running on Apple Silicon

On Apple Silicon, the snowflake library (and some scipy libraries) have issues. To work around this you need to setup `conda` to be able to create x86 environments.

Add the following to your `~/.zshrc`

```zsh
# Create x86 conda environment
create_x86_conda_environment () {
# example usage: create_x86_conda_environment
 CONDA_SUBDIR=osx-64 conda env create $@
}

# Create ARM conda environment
create_ARM_conda_environment () {
# example usage: create_ARM_conda_environment myenv_x86 python=3.9
 CONDA_SUBDIR=osx-arm64 conda env create $@
}
```
Then when you create an environment use `create_x86_conda_environment create `...  instead of `conda env create`

# Running Code

## Running Local Code, Locally

- inspect step definitions `python <flow filename>.py --environment=conda show`
- run flow in local namespace `python <flow filename>.py --environment=conda run`
- inspect results: see [this link](https://docs.metaflow.org/metaflow/client)
- optionally build on dev `git push -f origin <branch name>:dev` which allows testing whether conda environment can build on AWS
