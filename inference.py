import asyncio
import grpc

from farm_ng.core.event_service import (
    EventServiceGrpc,
)
from farm_ng.core.event_service_pb2 import (
    EventServiceConfig,
    SubscribeRequest,
)
from farm_ng.core.event_client import EventClient
from farm_ng.core import uri_pb2
from google.protobuf.message import Message


class CameraInferenceService:
    def __init__(
        self,
        event_service: EventServiceGrpc,
        configs: EventServiceConfig
    ) -> None:
        self._camera_client = EventClient(config)
        self._event_service = event_service
        self._event_service.request_reply_handler = self._request_reply_handler

    async def run(self):
        async for event, message in self._camera_client.subscribe(
            SubscribeRequest(uri=uri_pb2.Uri(path="/frame"), every_n=1),
            decode=True
        ):
            print(event)
            image_tensor = kornia.io.decode_image(
                message.image_data,
                message.encoding_type,
            )
            output = self.model(image_tensor)
            output_proto = ...

            # NOTE: you can read later from browser or fastapi
            await self._event_service.publish(
                "/results", output_proto
            )
    
    async def request_reply_handler(
        self, request: RequestReplyRequest
    ) -> Message:
        if request.event.uri.path == "/load_model":
            self.event_service.logger.info("Loading model")
            # query_dict = parser_event_uri(request.event.uri.query)
            # self.model = kornia.models.load_model(
            #     query_dict["model_path"]
            # )
            # if query_dict["compile"]:
            #     self.model = kornia.models.compile_model(
            #         self.model,
            #     )


if __name__ == "__main__":

    camera_config = EventServiceConfig(
        name="camera",
        port=50051,
        host="localhost",
    )

    inference_config = EventServiceConfig(
        name="inference",
        port=50052,
        host="localhost"
    )

    event_service: EventServiceGrpc = EventServiceGrpc(
        grpc.aio.server(), inference_config
    )

    inference_server = CameraInferenceService(
        event_service, camera_config
    )

    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(inference_server.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()