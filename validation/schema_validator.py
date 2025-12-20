"""
Schema validation for crime data CSV files.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from loguru import logger


class CSVSchemaValidator:
    """Validates CSV structure and data types."""
    
    # Required columns with their expected types
    REQUIRED_COLUMNS = {
        "occurrence_date": ["occurrence_date", "Occurrence Date", "OCCURRENCE_DATE", "occurrenceyearmonth"],
        "lat": ["lat", "Lat", "LAT", "latitude", "Latitude", "LATITUDE"],
        "long": ["long", "Long", "LONG", "longitude", "Longitude", "LONGITUDE"],
        "major_crime_indicator": [
            "major_crime_indicator",
            "Major Crime Indicator",
            "MCI_CATEGORY",
            "mci_category",
            "crime_type",
            "offence",
        ],
    }
    
    # Optional columns (nice to have)
    OPTIONAL_COLUMNS = {
        "neighbourhood": ["neighbourhood", "Neighbourhood", "NEIGHBOURHOOD", "hood"],
        "premise_type": [
            "premises_type",
            "Premises Type",
            "PREMISES_TYPE",
            "premise_type",
            "location_type",
        ],
        "occurrence_time": [
            "occurrence_time",
            "Occurrence Time",
            "OCCURRENCE_TIME",
            "time",
        ],
    }
    
    def __init__(self, df: pd.DataFrame):
        """Initialize validator with DataFrame."""
        self.df = df
        self.column_mapping: Dict[str, str] = {}
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
    
    def find_column(self, required_key: str, possible_names: List[str]) -> Optional[str]:
        """Find column name from possible variations."""
        for col in self.df.columns:
            col_lower = col.lower().strip()
            for name in possible_names:
                if col_lower == name.lower().strip():
                    return col
        return None
    
    def validate_structure(self) -> Tuple[bool, List[str]]:
        """
        Validate that CSV has required columns.
        
        Returns:
            (is_valid, missing_columns)
        """
        missing = []
        
        # Check required columns
        for required_key, possible_names in self.REQUIRED_COLUMNS.items():
            found_col = self.find_column(required_key, possible_names)
            if found_col:
                self.column_mapping[required_key] = found_col
            else:
                missing.append(required_key)
        
        # Check optional columns (warnings only)
        for optional_key, possible_names in self.OPTIONAL_COLUMNS.items():
            found_col = self.find_column(optional_key, possible_names)
            if found_col:
                self.column_mapping[optional_key] = found_col
            else:
                self.validation_warnings.append(
                    f"Optional column '{optional_key}' not found"
                )
        
        if missing:
            self.validation_errors.append(
                f"Missing required columns: {', '.join(missing)}"
            )
            return False, missing
        
        return True, []
    
    def validate_data_types(self) -> bool:
        """Validate that data types are reasonable."""
        errors = []
        
        # Check latitude
        if "lat" in self.column_mapping:
            lat_col = self.column_mapping["lat"]
            try:
                lat_series = pd.to_numeric(self.df[lat_col], errors="coerce")
                # Toronto is approximately 43.6 to 43.8
                invalid = (lat_series < 43.0) | (lat_series > 44.0) | lat_series.isna()
                invalid_count = invalid.sum()
                if invalid_count > 0:
                    errors.append(
                        f"Found {invalid_count} invalid latitude values (should be ~43.6-43.8)"
                    )
            except Exception as e:
                errors.append(f"Error validating latitude: {e}")
        
        # Check longitude
        if "long" in self.column_mapping:
            long_col = self.column_mapping["long"]
            try:
                long_series = pd.to_numeric(self.df[long_col], errors="coerce")
                # Toronto is approximately -79.2 to -79.6
                invalid = (long_series > -79.0) | (long_series < -80.0) | long_series.isna()
                invalid_count = invalid.sum()
                if invalid_count > 0:
                    errors.append(
                        f"Found {invalid_count} invalid longitude values (should be ~-79.2 to -79.6)"
                    )
            except Exception as e:
                errors.append(f"Error validating longitude: {e}")
        
        if errors:
            self.validation_errors.extend(errors)
            return False
        
        return True
    
    def validate_completeness(self) -> bool:
        """Check for excessive missing values."""
        warnings = []
        
        for key, col_name in self.column_mapping.items():
            if key in self.REQUIRED_COLUMNS:
                missing_pct = self.df[col_name].isna().sum() / len(self.df) * 100
                if missing_pct > 10:  # More than 10% missing
                    warnings.append(
                        f"Column '{col_name}' has {missing_pct:.1f}% missing values"
                    )
        
        if warnings:
            self.validation_warnings.extend(warnings)
        
        return True  # Warnings don't fail validation
    
    def get_column_mapping(self) -> Dict[str, str]:
        """Get the mapping of standard names to actual column names."""
        return self.column_mapping.copy()
    
    def get_validation_report(self) -> Dict:
        """Get full validation report."""
        return {
            "is_valid": len(self.validation_errors) == 0,
            "errors": self.validation_errors,
            "warnings": self.validation_warnings,
            "column_mapping": self.column_mapping,
            "row_count": len(self.df),
            "column_count": len(self.df.columns),
        }


def validate_csv(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Convenience function to validate CSV.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        (is_valid, validation_report)
    """
    validator = CSVSchemaValidator(df)
    
    # Run validations
    structure_valid, missing = validator.validate_structure()
    if structure_valid:
        validator.validate_data_types()
        validator.validate_completeness()
    
    report = validator.get_validation_report()
    return report["is_valid"], report

