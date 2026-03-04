import os

from unitree_sdk2py.core.channel import ChannelFactoryInitialize


def init_dds_channel(simulation_mode: bool) -> None:
    """Initialize Unitree DDS with optional explicit network interface.

    Environment overrides:
    - UNITREE_DDS_IFACE: network interface name (e.g. "eno1", "enp2s0")
    - UNITREE_DDS_DOMAIN_ID: integer DDS domain id
    """
    default_domain_id = 1 if simulation_mode else 0
    domain_override = os.getenv("UNITREE_DDS_DOMAIN_ID", "").strip()
    domain_id = default_domain_id if domain_override == "" else int(domain_override)

    network_iface = os.getenv("UNITREE_DDS_IFACE", "").strip()
    if network_iface:
        ChannelFactoryInitialize(domain_id, network_iface)
    else:
        ChannelFactoryInitialize(domain_id)
