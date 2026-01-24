#!/usr/bin/env python3
"""
Kubernetes Operator to sync API keys from deploymentconfig-* ConfigMaps to a Secret
Watches for ConfigMaps with name pattern 'deploymentconfig-*' and extracts API keys
to create/update a consolidated Secret that can be mounted into pods.
"""

import json
import re
import logging
import base64
from typing import Dict, Optional
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigMapSecretOperator:
    """Operator to sync ConfigMap data to Secret"""
    
    def __init__(
        self,
        target_namespace: str = "nemo-microservices",
        secret_name: str = "model-api-keys",
        configmap_pattern: str = r"^deploymentconfig-(.+)$",
        label_selector: Optional[str] = "app.nvidia.com/config-type=deploymentConfig"
    ):
        self.target_namespace = target_namespace
        self.secret_name = secret_name
        self.configmap_pattern = re.compile(configmap_pattern)
        self.label_selector = label_selector
        
        # Initialize Kubernetes clients
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        
        self.v1 = client.CoreV1Api()
        
    def extract_api_key_from_configmap(self, configmap) -> Optional[tuple]:
        """
        Extract API key and model name from ConfigMap
        Returns: (model_name, api_key) or None
        """
        cm_name = configmap.metadata.name
        
        # Check if ConfigMap name matches pattern
        match = self.configmap_pattern.match(cm_name)
        if not match:
            return None
        
        model_name = match.group(1)
        
        # Parse deploymentConfig JSON
        if not configmap.data or 'deploymentConfig' not in configmap.data:
            logger.warning(f"ConfigMap {cm_name} missing deploymentConfig data")
            return None
        
        try:
            config_data = json.loads(configmap.data['deploymentConfig'])
            api_key = config_data.get('ExternalEndpoint', {}).get('ApiKey')
            
            if not api_key or api_key == "REDACTED":
                logger.warning(f"No valid API key found in {cm_name}")
                return None
            
            logger.info(f"Extracted API key for model: {model_name}")
            return (model_name, api_key)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse deploymentConfig in {cm_name}: {e}")
            return None
    
    def get_all_api_keys(self) -> Dict[str, str]:
        """
        Scan all namespaces for matching ConfigMaps and extract API keys
        Returns: Dict mapping model names to API keys
        """
        api_keys = {}
        
        try:
            # List all ConfigMaps across all namespaces with label selector
            if self.label_selector:
                configmaps = self.v1.list_config_map_for_all_namespaces(
                    label_selector=self.label_selector
                )
            else:
                configmaps = self.v1.list_config_map_for_all_namespaces()
            
            for cm in configmaps.items:
                result = self.extract_api_key_from_configmap(cm)
                if result:
                    model_name, api_key = result
                    # Use uppercase env var style: MLIS_QWEN3_8B
                    env_var_name = model_name.upper().replace('-', '_')
                    api_keys[env_var_name] = api_key
            
            logger.info(f"Found {len(api_keys)} API keys across all namespaces")
            return api_keys
            
        except ApiException as e:
            logger.error(f"Error listing ConfigMaps: {e}")
            return {}
    
    def create_or_update_secret(self, api_keys: Dict[str, str]) -> bool:
        """
        Create or update the consolidated Secret with API keys
        """
        if not api_keys:
            logger.info("No API keys to store, skipping secret creation")
            return False
        
        # Kubernetes expects base64-encoded strings in the data field
        import base64
        secret_data = {k: base64.b64encode(v.encode()).decode() for k, v in api_keys.items()}
        
        secret = client.V1Secret(
            metadata=client.V1ObjectMeta(
                name=self.secret_name,
                namespace=self.target_namespace,
                labels={
                    "app": "model-api-keys",
                    "managed-by": "configmap-secret-operator"
                }
            ),
            data=secret_data,
            type="Opaque"
        )
        
        try:
            # Try to get existing secret
            existing_secret = self.v1.read_namespaced_secret(
                name=self.secret_name,
                namespace=self.target_namespace
            )
            
            # Update existing secret
            self.v1.replace_namespaced_secret(
                name=self.secret_name,
                namespace=self.target_namespace,
                body=secret
            )
            logger.info(f"Updated Secret {self.secret_name} in {self.target_namespace}")
            return True
            
        except ApiException as e:
            if e.status == 404:
                # Secret doesn't exist, create it
                self.v1.create_namespaced_secret(
                    namespace=self.target_namespace,
                    body=secret
                )
                logger.info(f"Created Secret {self.secret_name} in {self.target_namespace}")
                return True
            else:
                logger.error(f"Error managing secret: {e}")
                return False
    
    def handle_configmap_event(self, event_type: str, configmap):
        """Handle ConfigMap add/modify/delete events"""
        cm_name = configmap.metadata.name
        cm_namespace = configmap.metadata.namespace
        
        logger.info(f"Event {event_type} for ConfigMap {cm_name} in {cm_namespace}")
        
        # Rescan all ConfigMaps and update secret
        api_keys = self.get_all_api_keys()
        self.create_or_update_secret(api_keys)
    
    def run(self):
        """Main operator loop - watches for ConfigMap events"""
        logger.info("Starting ConfigMap Secret Operator...")
        
        # Initial sync
        logger.info("Performing initial sync...")
        api_keys = self.get_all_api_keys()
        self.create_or_update_secret(api_keys)
        
        # Watch for changes
        w = watch.Watch()
        logger.info("Watching for ConfigMap changes...")
        
        while True:
            try:
                if self.label_selector:
                    stream = w.stream(
                        self.v1.list_config_map_for_all_namespaces,
                        label_selector=self.label_selector,
                        timeout_seconds=300
                    )
                else:
                    stream = w.stream(
                        self.v1.list_config_map_for_all_namespaces,
                        timeout_seconds=300
                    )
                
                for event in stream:
                    event_type = event['type']
                    configmap = event['object']
                    
                    # Only process if matches our pattern
                    if self.configmap_pattern.match(configmap.metadata.name):
                        self.handle_configmap_event(event_type, configmap)
                        
            except ApiException as e:
                logger.error(f"API exception in watch loop: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in watch loop: {e}")
            
            # Brief pause before restarting watch
            import time
            time.sleep(5)


if __name__ == "__main__":
    operator = ConfigMapSecretOperator(
        target_namespace="nemo-microservices",
        secret_name="model-api-keys",
        configmap_pattern=r"^deploymentconfig-(.+)$",
        label_selector="app.nvidia.com/config-type=deploymentConfig"
    )
    operator.run()