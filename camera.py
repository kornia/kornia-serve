from __future__ import annotations
import asyncio
import cv2
import numpy as np
from kornia_rs import ImageEncoder
import grpc

from farm_ng.core.event_service import (
    EventServiceGrpc,
    EventServiceConfig,
)
from farm_ng.core.stamp import StampSemantics, get_monotonic_now
from farm_ng.core.timestamp_pb2 import Timestamp
import image_pb2


class CameraGrabberService:
    def __init__(self, events_service: EventServiceGrpc) -> None:
        self._events_service = events_service

        self._grabber: cv2.VideoCapture | None = None
        self._encoder = ImageEncoder()

        self._frame_counter: int = 0

    def initialize(self) -> bool:
        if self._grabber is not None:
            return True

        try:
            self._grabber = cv2.VideoCapture(0)
        except Exception as exc:
            self._events_service.logger.warning("Cannot initialize camera")
            return False

        return True

    async def run(self) -> None:
        if self._grabber is None:
            self.initialize()

        frame: np.ndarray

        while True:
            # read frame from the camera
            _, frame = self._grabber.read()

            # encode the frame into a JPEG
            image_data: list[int] = self._encoder.encode(
                frame.tobytes(), frame.shape
            )

            host_recv_stamp: Timestamp = get_monotonic_now(
                StampSemantics.DRIVER_RECEIVE
            )

            # convert the frame to a protobuf message
            frame_proto = image_pb2.Image(
                image_data=bytes(image_data),
                frame_number=self._frame_counter,
                encoding_type="jpeg",
                image_size=image_pb2.ImageSize(
                    height=frame.shape[0],
                    width=frame.shape[1]
                ),
            )

            # publish the frame
            await self._events_service.publish(
                "/frame",
                frame_proto,
                [host_recv_stamp]
            )

            self._frame_counter += 1

    async def serve(self) -> None:
        async_tasks: list[asyncio.Task] = []
        async_tasks.append(asyncio.create_task(self.run()))
        async_tasks.append(asyncio.create_task(self._events_service.serve()))
        await asyncio.gather(*async_tasks)


if __name__ == "__main__":

    server_config = EventServiceConfig(
        port=5001,
        host="localhost",
    )

    events_service = EventServiceGrpc(
        grpc.aio.server(), server_config
    )

    camera_service = CameraGrabberService(events_service)

    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(camera_service.serve())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
