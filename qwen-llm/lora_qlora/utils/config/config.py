"""
Configuration utilities.

This module provides utilities for loading, saving, and merging configurations.
"""

import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    🎯 LOAD CONFIGURATION
    
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path], format: str = 'json'):
    """
    🎯 SAVE CONFIGURATION
    
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        format: File format ('json' or 'yaml')
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if format.lower() == 'json':
            json.dump(config, f, indent=2, ensure_ascii=False)
        elif format.lower() in ['yaml', 'yml']:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported format: {format}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    🎯 MERGE CONFIGURATIONS
    
    Merge two configurations with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    🎯 VALIDATE CONFIGURATION
    
    Validate that configuration contains required keys.
    
    Args:
        config: Configuration to validate
        required_keys: List of required keys
        
    Returns:
        True if valid, False otherwise
    """
    for key in required_keys:
        if key not in config:
            return False
    return True


def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    🎯 GET CONFIG VALUE
    
    Get configuration value with default fallback.
    
    Args:
        config: Configuration dictionary
        key: Key to retrieve
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    return config.get(key, default)


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    🎯 UPDATE CONFIGURATION
    
    Update configuration with new values.
    
    Args:
        config: Configuration to update
        updates: Updates to apply
        
    Returns:
        Updated configuration
    """
    config.update(updates)
    return config
