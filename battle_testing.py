import asyncio

import websockets

message = """>start {"formatid":"gen7randombattle"}
>player p1 {"name":"Alice"}
>player p2 {"name":"Bob"}
' | ./pokemon-showdown simulate-battle"""


async def main():
    """
    Loading function. Connect websocket then launch bot.
    """
    async with websockets.connect('ws://localhost:8000/showdown/websocket') as websocket:
        await websocket.send(""">start {"formatid":"gen8randombattle"}
>player p1 {"name":"cincocosine"}
>player p2 {"name":"Bob"}""")

asyncio.get_event_loop().run_until_complete(main())
