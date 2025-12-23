"""
Dataset configuration for Toronto Open Data Portal.

All datasets are configured with their CKAN dataset keys.
Resource IDs are discovered automatically via CKAN API.
"""

from typing import Dict, Optional

TORONTO_OPEN_DATA_BASE = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
TORONTO_CKAN_API = f"{TORONTO_OPEN_DATA_BASE}/api/3/action"

# All Toronto crime-related datasets
TORONTO_DATASETS: Dict[str, Dict[str, Optional[str]]] = {
    "major-crime-indicators": {
        "name": "Major Crime Indicators",
        "resource_id": None,  # Auto-discover via CKAN API (old resource ID was outdated)
        "filename": "major-crime-indicators.csv",
        "description": "Major crime indicators from Toronto Police Service",
    },
    "police-annual-statistical-report-shooting-occurrences": {
        "name": "Shooting Occurrences",
        "resource_id": None,  # Will be discovered via API
        "filename": "shooting-occurrences.csv",
        "description": "Annual statistical report on shooting occurrences",
    },
    "wellbeing-toronto-safety": {
        "name": "Wellbeing Toronto Safety",
        "resource_id": None,
        "filename": "wellbeing-safety.csv",
        "description": "Wellbeing Toronto safety indicators",
    },
    "police-annual-statistical-report-reported-crimes": {
        "name": "Reported Crimes",
        "resource_id": None,
        "filename": "reported-crimes.csv",
        "description": "Annual statistical report on reported crimes",
    },
    "police-annual-statistical-report-miscellaneous-data": {
        "name": "Miscellaneous Police Data",
        "resource_id": None,
        "filename": "miscellaneous-data.csv",
        "description": "Miscellaneous police statistical data",
    },
    "police-annual-statistical-report-homicide": {
        "name": "Homicide Data",
        "resource_id": None,
        "filename": "homicide.csv",
        "description": "Annual statistical report on homicides",
    },
    "police-annual-statistical-report-miscellaneous-firearms": {
        "name": "Miscellaneous Firearms",
        "resource_id": None,
        "filename": "miscellaneous-firearms.csv",
        "description": "Miscellaneous firearms-related data",
    },
    "theft-from-motor-vehicle": {
        "name": "Theft from Motor Vehicle",
        "resource_id": None,
        "filename": "theft-from-motor-vehicle.csv",
        "description": "Theft from motor vehicle incidents",
    },
    "shootings-firearm-discharges": {
        "name": "Shootings and Firearm Discharges",
        "resource_id": None,
        "filename": "shootings-firearm-discharges.csv",
        "description": "Shootings and firearm discharge incidents",
    },
    "neighbourhood-crime-rates": {
        "name": "Neighbourhood Crime Rates",
        "resource_id": None,
        "filename": "neighbourhood-crime-rates.csv",
        "description": "Neighbourhood crime rates data",
    },
}


def get_all_dataset_keys() -> list[str]:
    """Get list of all dataset keys."""
    return list(TORONTO_DATASETS.keys())


def get_dataset_info(dataset_key: str) -> Optional[Dict[str, Optional[str]]]:
    """Get dataset configuration."""
    return TORONTO_DATASETS.get(dataset_key)


def get_dataset_url(dataset_key: str, resource_id: str) -> str:
    """
    Construct download URL for a dataset.
    
    Args:
        dataset_key: Dataset key from TORONTO_DATASETS
        resource_id: Resource ID from CKAN API
        
    Returns:
        Full download URL
    """
    return (
        f"{TORONTO_OPEN_DATA_BASE}/dataset/{dataset_key}/resource/"
        f"{resource_id}/download/{TORONTO_DATASETS[dataset_key]['filename']}"
    )

