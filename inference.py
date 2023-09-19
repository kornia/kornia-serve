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
from kornia_rs import ImageDecoder
import numpy as np
import cv2
import image_pb2


class InferenceService:
    def __init__(
        self,
        event_service: EventServiceGrpc,
        camera_client: EventClient,
    ) -> None:
        self._camera_client = camera_client
        self._event_service = event_service

        self._image_decoder = ImageDecoder()

    async def viz_image(self, image: np.ndarray) -> None:
        cv2.imshow("frame", image)
        cv2.waitKey(1)

    async def run(self):
        async for event, message in self._camera_client.subscribe(
            SubscribeRequest(uri=uri_pb2.Uri(path="/frame"), every_n=1),
            decode=True
        ):
            # decode the image
            image_dl = self._image_decoder.decode(
                message.image_data,
            )

            image_np = np.from_dlpack(image_dl)

            # NOTE: you can read later from browser or fastapi
            await self._event_service.publish(
                "/results",
                image_pb2.ImageSize(
                    height=image_np.shape[0],
                    width=image_np.shape[1],
                ),
            )

            # visualize the image in a separate task
            asyncio.create_task(self.viz_image(image_np))
    
    async def serve(self):
        await asyncio.gather(
            self.run(),
            self._event_service.serve(),
        )


if __name__ == "__main__":

    camera_config = EventServiceConfig(
        name="camera",
        port=5001,
        host="localhost",
    )

    inference_config = EventServiceConfig(
        name="inference",
        port=5002,
        host="localhost"
    )

    event_service: EventServiceGrpc = EventServiceGrpc(
        grpc.aio.server(), inference_config
    )

    camera_client = EventClient(camera_config)

    inference_service = InferenceService(
        event_service, camera_client
    )

    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(inference_service.serve())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()