import json
import os
from typing import Optional, Dict, Any
from datetime import datetime

from kafka import KafkaProducer
from kafka.errors import KafkaError

from logger import get_logger

logger = get_logger(__name__)

DEFAULT_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


class KafkaClient:
    def __init__(
        self,
        bootstrap_servers: str = None,
        client_id: str = "fulin-api-service"
    ):
        self.bootstrap_servers = bootstrap_servers or DEFAULT_BOOTSTRAP_SERVERS
        self.client_id = client_id
        self._producer = None

    @property
    def producer(self) -> KafkaProducer:
        if self._producer is None:
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                client_id=self.client_id,
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                retry_backoff_ms=1000,
                max_in_flight_requests_per_connection=5,
            )
            logger.info("Kafka producer connected: %s", self.bootstrap_servers)
        return self._producer

    def send(
        self,
        topic: str,
        message: Dict[str, Any],
        key: str = None,
        headers: Dict[str, str] = None
    ) -> bool:
        """发送消息到 Kafka"""
        try:
            enriched_message = {
                "_metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": self.client_id,
                    "version": "1.0",
                },
                **message
            }

            future = self.producer.send(
                topic=topic,
                key=key,
                value=enriched_message,
                headers=[(k, v.encode()) for k, v in (headers or {}).items()]
            )

            record_metadata = future.get(timeout=10)
            logger.debug(
                "Message sent to %s partition=%d offset=%d",
                topic,
                record_metadata.partition,
                record_metadata.offset
            )
            return True

        except KafkaError as e:
            logger.error("Failed to send message to %s: %s", topic, e)
            return False

    def close(self):
        """关闭连接"""
        if self._producer:
            self._producer.close()
        logger.info("Kafka client closed")


# 全局客户端实例
kafka_client = KafkaClient()
