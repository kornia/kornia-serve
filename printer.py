import asyncio

from farm_ng.core.event_service_pb2 import (
    EventServiceConfig,
    SubscribeRequest,
)
from farm_ng.core.event_client import EventClient
from farm_ng.core import uri_pb2


async def main(client: EventClient) -> None:
    async for event, message in client.subscribe(
        SubscribeRequest(uri=uri_pb2.Uri(path="/results"), every_n=1),
        decode=True
    ):
        print(message)


if __name__ == "__main__":

    inference_config = EventServiceConfig(
        name="inference",
        port=5002,
        host="localhost"
    )

    inference_client = EventClient(inference_config)

    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(main(inference_client))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
