from typing import Generic, Protocol, TypeVar

TRequest = TypeVar("TRequest", contravariant=True)
TResponse = TypeVar("TResponse", covariant=True)
TConfig = TypeVar("TConfig", covariant=True)


class MusicServiceProtocol(Protocol, Generic[TRequest, TResponse, TConfig]):
    def get_config(self) -> TConfig: ...
    def generate(self, request: TRequest) -> TResponse: ...
