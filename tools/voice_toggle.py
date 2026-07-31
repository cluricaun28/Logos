"""Voice mode runtime toggle."""
import os
import yaml

def voice_toggle(action: str = "status") -> str:
    """Toggle voice mode on/off/status.
    
    Args:
        action: 'status', 'on', or 'off'
        
    Returns:
        Status message
    """
    config_path = os.path.expanduser("~/.hermes/config.yaml")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        return f"Error reading config: {e}"
    
    voice_config = config.get("voice", {})
    
    if action == "status":
        is_active = voice_config.get("active", True)
        return f"Voice mode: {'ACTIVE' if is_active else 'INACTIVE'}"
    
    elif action == "on":
        voice_config["active"] = True
        config["voice"] = voice_config
        try:
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            return "Voice mode: ENABLED"
        except Exception as e:
            return f"Error writing config: {e}"
    
    elif action == "off":
        voice_config["active"] = False
        config["voice"] = voice_config
        try:
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            return "Voice mode: DISABLED"
        except Exception as e:
            return f"Error writing config: {e}"
    
    else:
        return f"Usage: voice_toggle [on|off|status]"

if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    print(voice_toggle(action))
