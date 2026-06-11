"""
Patient Simulator Engine — entry point.

Usage:
    python -m engine.main

Starts the gRPC server on PORT (default 50051).
"""

from __future__ import annotations

import asyncio
import os
import signal

import grpc
import structlog

from engine.proto.v1 import simulation_pb2_grpc as pb_grpc
from engine.server import SimulationServicer, add_servicer_to_server
from engine.store.postgres import PostgresStore

logger = structlog.get_logger()

PORT = int(os.getenv("ENGINE_PORT", "50051"))
SHUTDOWN_EVENT = asyncio.Event()


def _handle_signal(sig: int, _) -> None:
    logger.info("signal_received", signal=sig)
    SHUTDOWN_EVENT.set()


async def serve() -> None:
    store = PostgresStore()
    # Attempt DB connection; if it fails we still start the gRPC server
    # so the binary is demonstrable, but log a loud warning.
    try:
        await store.connect()
        logger.info("postgres_connected", dsn=store.dsn)
    except Exception as exc:
        logger.warning("postgres_unavailable", error=str(exc))
        store = None  # type: ignore[assignment]

    servicer = SimulationServicer(store=store)
    server = grpc.aio.server()
    add_servicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{PORT}")

    await server.start()
    logger.info("grpc_server_started", port=PORT)

    await SHUTDOWN_EVENT.wait()

    logger.info("grpc_server_shutting_down")
    await server.stop(5)
    if store:
        await store.close()
    logger.info("grpc_server_stopped")


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    asyncio.run(serve())


if __name__ == "__main__":
    main()
