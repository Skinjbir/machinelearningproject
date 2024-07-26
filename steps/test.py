from zenml.client import Client

artifact = Client().get_artifact_version('5afe1ce0-d229-4ec4-89c4-60da519745d1')
loaded_artifact = artifact.load()

print(type(loaded_artifact))