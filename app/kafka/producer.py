from confluent_kafka import Producer
from app import config

def delivery_report(err, msg):
    """ Callback for checking message delivery success or failure """
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

producer_config = {
    "bootstrap.servers": config.KAFKA_BROKER
}

producer = Producer(producer_config)

def send_message(key: str, value: str):
    """ Send message to Kafka """
    producer.produce(config.KAFKA_TOPIC, key=key, value=value, callback=delivery_report)
    producer.flush()  # Ensure messages are delivered
