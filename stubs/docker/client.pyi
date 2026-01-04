from .models.containers import ContainerCollection
from .models.images import ImageCollection

class DockerClient:
    containers: ContainerCollection
    images: ImageCollection

