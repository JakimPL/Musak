from typing import Generic, Protocol, TypeVar

TRequest_contra = TypeVar("TRequest_contra", contravariant=True)
TResponse_co = TypeVar("TResponse_co", covariant=True)
TConfig_co = TypeVar("TConfig_co", covariant=True)


class MusicServiceProtocol(Protocol, Generic[TRequest_contra, TResponse_co, TConfig_co]):
    def get_config(self) -> TConfig_co: ...
    def generate(self, request: TRequest_contra) -> TResponse_co: ...
