from confluent_kafka import Consumer, KafkaException, KafkaError
from app import config

consumer_config = {
    "bootstrap.servers": config.KAFKA_BROKER,
    "group.id": "fastapi-group",
    "auto.offset.reset": "earliest"
}

consumer = Consumer(consumer_config)
consumer.subscribe([config.KAFKA_TOPIC])

def consume_messages():
    """ Consume messages from Kafka """
    try:
        while True:
            msg = consumer.poll(1.0)  # Wait for message
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    raise KafkaException(msg.error())
            print(f"Received message: {msg.value().decode('utf-8')}")
    finally:
        consumer.close()
