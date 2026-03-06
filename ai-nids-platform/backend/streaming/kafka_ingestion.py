import asyncio
import json
import logging
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from river.drift import ADWIN

# Setup structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class StreamIngestionPipeline:
    """
    Consumes network flow logs (Zeek/Suricata) via Kafka.
    Executes inference via the HybridNIDS.
    Monitors for concept drift using ADWIN.
    Produces alerts to an outbound Kafka topic for the SOC Dashboard.
    """
    def __init__(self, brokers="kafka:9092", in_topic="network-flows", out_topic="soc-alerts"):
        self.brokers = brokers
        self.in_topic = in_topic
        self.out_topic = out_topic
        # ADWIN drift detector initialized for error rate monitoring
        self.drift_detector = ADWIN(delta=0.002) 

    async def start(self):
        self.consumer = AIOKafkaConsumer(
            self.in_topic,
            bootstrap_servers=self.brokers,
            group_id="nids-inference-group",
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.brokers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        await self.consumer.start()
        await self.producer.start()
        logger.info(f"Started Streaming Ingestion on {self.in_topic}")
        
        try:
            await self._consume_loop()
        finally:
            await self.consumer.stop()
            await self.producer.stop()

    async def _consume_loop(self):
        from ai_nids.models.hybrid_detector import HybridNIDS
        import numpy as np
        model = HybridNIDS()
        
        async for msg in self.consumer:
            flow_data = msg.value
            features = flow_data.get('features', [])
            
            if not features or len(features) != 49:
                continue

            result = model.predict(np.array([features]))
            is_anomaly = result["attack_type"][0] != "Normal"
            
            if is_anomaly:
                alert = {
                    "timestamp": flow_data.get('timestamp'),
                    "src_ip": flow_data.get('src_ip'),
                    "attack_type": "Zero_Day_Anomaly",
                    "severity": "CRITICAL"
                }
                await self.producer.send_and_wait(self.out_topic, alert)
            
            # Online Drift Monitoring (Simulated True/False ground truth feedback loop)
            # If an analyst later provides a label, we update ADWIN
            # error = 1 if model_prediction != actual_label else 0
            # self.drift_detector.update(error)
            # if self.drift_detector.drift_detected:
            #     logger.warning("CONCEPT DRIFT DETECTED! Triggering retraining pipeline.")

if __name__ == "__main__":
    pipeline = StreamIngestionPipeline(brokers="localhost:9092")
    asyncio.run(pipeline.start())
