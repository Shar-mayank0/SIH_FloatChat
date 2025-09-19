import netCDF4 as nc
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta
import glob

class NetCDFParser:
    def __init__(self, netcdf_file_path):
        """
        Initialize the NetCDF parser with a file path.

        Args:
            netcdf_file_path (str): Path to the NetCDF file
        """
        self.file_path = netcdf_file_path
        self.dataset = None
        self.dimensions = {}
        self.variables = {}

    def open_dataset(self):
        """Open the NetCDF dataset and load basic information."""
        try:
            self.dataset = nc.Dataset(self.file_path, 'r')
            print(f"Successfully opened: {self.file_path}")
            return True
        except Exception as e:
            print(f"Error opening NetCDF file: {e}")
            return False

    def inspect_file_structure(self):
        """Inspect and display the structure of the NetCDF file."""
        if not self.dataset:
            print("Dataset not opened. Call open_dataset() first.")
            return

        print("\n" + "="*50)
        print("NetCDF File Structure Analysis")
        print("="*50)

        # Display dimensions
        print("\nDIMENSIONS:")
        print("-" * 20)
        for dim_name, dimension in self.dataset.dimensions.items():
            size = len(dimension)
            unlimited = "(unlimited)" if dimension.isunlimited() else ""
            print(f"  {dim_name}: {size} {unlimited}")
            self.dimensions[dim_name] = size

        # Display variables
        print("\nVARIABLES:")
        print("-" * 20)
        for var_name, variable in self.dataset.variables.items():
            dims = variable.dimensions
            shape = variable.shape
            dtype = variable.dtype
            print(f"  {var_name}:")
            print(f"    Dimensions: {dims}")
            print(f"    Shape: {shape}")
            print(f"    Data type: {dtype}")

            # Display attributes if available
            if hasattr(variable, 'long_name'):
                print(f"    Description: {variable.long_name}")
            if hasattr(variable, 'units'):
                print(f"    Units: {variable.units}")
            print()

            self.variables[var_name] = {
                'dimensions': dims,
                'shape': shape,
                'dtype': dtype,
                'variable': variable
            }

    def extract_coordinate_variables(self):
        """Extract coordinate variables (usually time, lat, lon, depth)."""
        coordinates = {}

        for var_name, var_info in self.variables.items():
            # Coordinate variables usually have the same name as their dimension
            if (len(var_info['dimensions']) == 1 and
                    var_info['dimensions'][0] == var_name):
                coordinates[var_name] = self.dataset.variables[var_name][:] # type: ignore
                print(f"Found coordinate variable: {var_name}")

        return coordinates

    def _safe_convert_value(self, value):
        """Safely convert a value to appropriate Python type."""
        if np.ma.is_masked(value):
            return None

        if isinstance(value, (np.bytes_, bytes)):
            return value.decode('utf-8', errors='ignore').strip('\x00')
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in ['S', 'U']:  # String types
                # Handle different string array formats
                try:
                    if value.ndim == 0:
                        # Scalar string
                        return str(value.item()).strip('\x00')
                    else:
                        # Character array - join bytes
                        if value.dtype.kind == 'S':
                            # Byte string array
                            joined = b''.join([item if isinstance(item, bytes) else str(item).encode() for item in value.flat])
                            return joined.decode('utf-8', errors='ignore').strip('\x00')
                        else:
                            # Unicode string array
                            return ''.join([str(item) for item in value.flat]).strip('\x00')
                except (ValueError, TypeError, UnicodeDecodeError):
                    return str(value).strip()
            else:
                # Non-string array
                if value.ndim == 0:
                    return self._safe_convert_value(value.item())
                else:
                    return str(value).strip()
        elif isinstance(value, (np.floating, float)):
            return float(value) if not np.isnan(value) else None
        elif isinstance(value, (np.integer, int)):
            return int(value)
        elif isinstance(value, str):
            return value.strip('\x00')
        else:
            try:
                return float(value) if str(value) != '' else None
            except (ValueError, TypeError):
                return str(value).strip()

    def _identify_parameter_types(self):
        """
        Dynamically identify different types of parameters in the dataset.
        Returns dictionaries categorizing variables by their purpose.
        """
        if not self.dataset:
            return {}, {}, {}, {}

        # Get dimensions info
        n_prof_exists = 'n_prof' in self.dataset.dimensions
        n_levels_exists = 'n_levels' in self.dataset.dimensions

        # Categories for different types of variables
        profile_metadata = {}  # Profile-level information (1D with n_prof)
        measurement_data = {}  # Scientific measurements (2D with n_prof, n_levels)
        quality_flags = {}     # Quality control flags
        global_attributes = {} # Global/scalar variables

        for var_name, var_info in self.variables.items():
            dims = var_info['dimensions']
            shape = var_info['shape']
            dtype = var_info['dtype']

            # Profile-level metadata (1D with n_prof dimension)
            if (len(dims) == 1 and n_prof_exists and 'n_prof' in dims) or \
               (len(dims) == 2 and n_prof_exists and 'n_prof' in dims and any('string' in dim for dim in dims)):
                profile_metadata[var_name] = {
                    'dimensions': dims,
                    'dtype': dtype,
                    'is_string': dtype.kind in ['S', 'U'] or any('string' in dim for dim in dims)
                }

            # Measurement data (2D with n_prof and n_levels)
            elif (len(dims) == 2 and n_prof_exists and n_levels_exists and
                  'n_prof' in dims and 'n_levels' in dims):
                if '_qc' in var_name.lower():
                    quality_flags[var_name] = {
                        'dimensions': dims,
                        'dtype': dtype,
                        'parameter': var_name.replace('_qc', '').replace('_QC', '')
                    }
                else:
                    measurement_data[var_name] = {
                        'dimensions': dims,
                        'dtype': dtype,
                        'is_adjusted': 'adjusted' in var_name.lower(),
                        'is_error': 'error' in var_name.lower()
                    }

            # Global attributes (scalar or 1D without n_prof)
            elif (len(dims) == 0) or \
                 (len(dims) == 1 and not (n_prof_exists and 'n_prof' in dims)):
                global_attributes[var_name] = {
                    'dimensions': dims,
                    'dtype': dtype,
                    'is_string': dtype.kind in ['S', 'U'] or any('string' in dim for dim in dims)
                }

        return profile_metadata, measurement_data, quality_flags, global_attributes

    def extract_argo_data(self, output_file=None):
        """
        Extract ARGO float data with dynamic parameter detection.
        Creates a structured CSV optimized for RAG applications.
        """
        if not self.dataset:
            print("Dataset not opened. Call open_dataset() first.")
            return

        if output_file is None:
            output_file = Path(self.file_path).stem + "_argo_data.csv"

        print("\nExtracting ARGO float data...")

        # Dynamically identify parameter types
        profile_metadata, measurement_data, quality_flags, global_attrs = self._identify_parameter_types()

        print(f"Detected parameter categories:")
        print(f"  Profile metadata: {len(profile_metadata)} variables")
        print(f"  Measurement data: {len(measurement_data)} variables")
        print(f"  Quality flags: {len(quality_flags)} variables")
        print(f"  Global attributes: {len(global_attrs)} variables")

        # Check required dimensions
        if 'n_prof' not in self.dataset.dimensions:
            print("Warning: n_prof dimension not found. This may not be standard ARGO data.")
            return None

        if 'n_levels' not in self.dataset.dimensions:
            print("Warning: n_levels dimension not found. This may not be standard ARGO data.")
            return None

        n_prof = self.dataset.dimensions['n_prof'].size
        n_levels = self.dataset.dimensions['n_levels'].size

        # Key variables to extract
        profile_data = []

        # Process each profile
        for prof_idx in range(n_prof):
            # Extract profile metadata dynamically
            profile_meta = {'profile_id': prof_idx}

            # Add all available profile metadata
            for var_name in profile_metadata:
                try:
                    if profile_metadata[var_name]['is_string']:
                        value = self._safe_convert_value(self.dataset.variables[var_name][prof_idx])
                    else:
                        value = self._safe_convert_value(self.dataset.variables[var_name][prof_idx])
                    profile_meta[var_name] = value
                except Exception as e:
                    print(f"Warning: Could not extract {var_name} for profile {prof_idx}: {e}")
                    profile_meta[var_name] = None # type: ignore

            # Convert Julian day to readable date if available
            if 'juld' in profile_meta and profile_meta['juld'] is not None:
                try:
                    ref_date = datetime(1950, 1, 1)
                    actual_date = ref_date + timedelta(days=float(profile_meta['juld']))
                    profile_meta['date'] = actual_date.strftime('%Y-%m-%d %H:%M:%S') # type: ignore
                except:
                    profile_meta['date'] = None # type: ignore

            # Extract measurements for this profile dynamically
            for level_idx in range(n_levels):
                # Check if this level has any valid measurement data
                has_valid_data = False
                level_data = {}

                # Extract all measurement variables
                for var_name in measurement_data:
                    try:
                        value = self.dataset.variables[var_name][prof_idx, level_idx]
                        converted_value = self._safe_convert_value(value)
                        level_data[var_name] = converted_value

                        if converted_value is not None and not np.ma.is_masked(value):
                            has_valid_data = True
                    except Exception as e:
                        print(f"Warning: Could not extract {var_name} for profile {prof_idx}, level {level_idx}: {e}")
                        level_data[var_name] = None

                # Extract quality flags for this level
                for qc_var_name in quality_flags:
                    try:
                        qc_value = self.dataset.variables[qc_var_name][prof_idx, level_idx]
                        level_data[qc_var_name] = self._safe_convert_value(qc_value)
                    except Exception as e:
                        print(f"Warning: Could not extract {qc_var_name} for profile {prof_idx}, level {level_idx}: {e}")
                        level_data[qc_var_name] = None

                # Only add row if there's valid measurement data
                if has_valid_data:
                    row = profile_meta.copy()
                    row['level'] = level_idx
                    row.update(level_data)
                    profile_data.append(row)

        # Create DataFrame and save
        if profile_data:
            df = pd.DataFrame(profile_data)

            # Dynamically identify string columns for cleaning
            string_columns = []
            for col in df.columns:
                if col in profile_metadata and profile_metadata[col]['is_string']:
                    string_columns.append(col)
                elif col in quality_flags:  # QC flags are usually strings
                    string_columns.append(col)
                elif df[col].dtype == 'object':  # Pandas object type often contains strings
                    string_columns.append(col)

            print(f"Identified string columns for cleaning: {string_columns}")

            # Clean up string columns
            for col in string_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()

            # Sort by available columns (flexible sorting)
            sort_columns = []
            for preferred_col in ['platform_number', 'cycle_number', 'pres', 'pressure']:
                if preferred_col in df.columns:
                    sort_columns.append(preferred_col)
                    break  # Use first available sorting column

            if sort_columns:
                df = df.sort_values(sort_columns)

            df.to_csv(output_file, index=False)
            print(f"\nARGO data successfully saved to: {output_file}")
            print(f"Total measurements: {len(df)}")
            print(f"Number of profiles: {n_prof}")
            print(f"Columns: {list(df.columns)}")

            # Show summary statistics (flexible based on available columns)
            print(f"\nData Summary:")
            if 'platform_number' in df.columns:
                print(f"  Platforms: {df['platform_number'].nunique()}")
            if 'date' in df.columns and df['date'].notna().any():
                print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            if 'latitude' in df.columns:
                print(f"  Latitude range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
            if 'longitude' in df.columns:
                print(f"  Longitude range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")

            # Check for pressure columns (flexible naming)
            pressure_col = None
            for col in ['pres', 'pressure', 'PRES']:
                if col in df.columns and df[col].notna().any():
                    pressure_col = col
                    break
            if pressure_col:
                print(f"  Pressure range: {df[pressure_col].min():.1f} to {df[pressure_col].max():.1f} dbar")

            # Check for temperature columns (flexible naming)
            temp_col = None
            for col in ['temp', 'temperature', 'TEMP']:
                if col in df.columns and df[col].notna().any():
                    temp_col = col
                    break
            if temp_col:
                print(f"  Temperature range: {df[temp_col].min():.2f} to {df[temp_col].max():.2f} °C")

            print(f"\nDetected measurement parameters:")
            for var_name in measurement_data:
                if var_name in df.columns:
                    non_null_count = df[var_name].notna().sum()
                    print(f"  {var_name}: {non_null_count} valid measurements")

            print(f"\nFirst few rows:")
            print(df.head())

            return df
        else:
            print("No data to save.")
            return None

    def flatten_data_to_csv(self, output_file=None, variables_to_extract=None):
        """
        Flatten multi-dimensional NetCDF data to CSV format.
        For ARGO data, use extract_argo_data() instead for better results.

        Args:
            output_file (str): Output CSV file path. If None, uses input filename.
            variables_to_extract (list): List of variable names to extract.
                                       If None, extracts scientific measurement variables only.
        """
        if not self.dataset:
            print("Dataset not opened. Call open_dataset() first.")
            return

        if output_file is None:
            output_file = Path(self.file_path).stem + "_extracted.csv"

        # For ARGO data, focus on scientific variables
        if variables_to_extract is None:
            # Focus on the key oceanographic measurements
            variables_to_extract = ['juld', 'latitude', 'longitude', 'pres', 'temp',
                                    'pres_adjusted', 'temp_adjusted', 'cycle_number']
            # Only include variables that exist in the dataset
            variables_to_extract = [var for var in variables_to_extract
                                    if var in self.variables.keys()]

        print(f"\nExtracting variables: {variables_to_extract}")
        print("Note: For ARGO data, consider using extract_argo_data() method for better formatting.")

        # Create a list to store all data rows
        data_rows = []

        # Process each data variable
        for var_name in variables_to_extract:
            if var_name not in self.variables:
                print(f"Warning: Variable '{var_name}' not found in dataset.")
                continue

            var_info = self.variables[var_name]
            var_data = self.dataset.variables[var_name][:]

            print(f"Processing variable: {var_name}")
            print(f"  Shape: {var_data.shape}")
            print(f"  Dimensions: {var_info['dimensions']}")
            print(f"  Data type: {var_info['dtype']}")

            # Handle different dimensionalities
            if len(var_data.shape) == 0:
                # Scalar variable
                value = self._safe_convert_value(var_data)
                row = {'variable': var_name, 'value': value}
                data_rows.append(row)

            elif len(var_data.shape) == 1:
                # 1D variable (e.g., profile-level data)
                dim_name = var_info['dimensions'][0]
                for i, value in enumerate(var_data):
                    converted_value = self._safe_convert_value(value)
                    if converted_value is not None:  # Skip None/masked values
                        row = {
                            'variable': var_name,
                            dim_name + '_index': i,
                            'value': converted_value
                        }
                        data_rows.append(row)

            elif len(var_data.shape) == 2:
                # 2D variable (e.g., profile x level data)
                dims = var_info['dimensions']
                for i in range(var_data.shape[0]):
                    for j in range(var_data.shape[1]):
                        converted_value = self._safe_convert_value(var_data[i, j])
                        if converted_value is not None:
                            row = {
                                'variable': var_name,
                                dims[0] + '_index': i,
                                dims[1] + '_index': j,
                                'value': converted_value
                            }
                            data_rows.append(row)

        # Convert to DataFrame and save
        if data_rows:
            df = pd.DataFrame(data_rows)
            df.to_csv(output_file, index=False)
            print(f"\nData successfully saved to: {output_file}")
            print(f"Total rows: {len(df)}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst few rows:")
            print(df.head())
        else:
            print("No data to save.")

    def create_separate_csv_per_variable(self, output_dir=None):
        """
        Create separate CSV files for each variable with all its dimensions.
        This approach preserves the structure better for some use cases.
        """
        if not self.dataset:
            print("Dataset not opened. Call open_dataset() first.")
            return

        if output_dir is None:
            output_dir = Path(self.file_path).stem + "_csv_files"

        os.makedirs(output_dir, exist_ok=True)

        coordinates = self.extract_coordinate_variables()

        for var_name, var_info in self.variables.items():
            if var_name in coordinates:
                continue  # Skip coordinate variables

            var_data = self.dataset.variables[var_name][:]
            dims = var_info['dimensions']

            print(f"Creating CSV for variable: {var_name}")

            # Create coordinate meshgrid for multi-dimensional data
            if len(dims) > 1:
                coord_arrays = []
                for dim in dims:
                    if dim in coordinates:
                        coord_arrays.append(coordinates[dim])
                    else:
                        coord_arrays.append(np.arange(self.dimensions[dim]))

                # Create meshgrid
                coord_mesh = np.meshgrid(*coord_arrays, indexing='ij')

                # Flatten everything
                rows_data = {}
                for i, dim in enumerate(dims):
                    rows_data[dim] = coord_mesh[i].flatten()

                rows_data[var_name] = var_data.flatten()

                # Remove masked values
                mask = ~np.ma.is_masked(var_data.flatten())
                for key in rows_data:
                    rows_data[key] = rows_data[key][mask]

                df = pd.DataFrame(rows_data)
            else:
                # 1D variable
                dim_name = dims[0]
                if dim_name in coordinates:
                    df = pd.DataFrame({
                        dim_name: coordinates[dim_name],
                        var_name: var_data
                    })
                else:
                    df = pd.DataFrame({
                        dim_name: np.arange(len(var_data)),
                        var_name: var_data
                    })

            # Save to CSV
            output_file = os.path.join(output_dir, f"{var_name}.csv")
            df.to_csv(output_file, index=False)
            print(f"  Saved: {output_file} ({len(df)} rows)")

    def close(self):
        """Close the NetCDF dataset."""
        if self.dataset:
            self.dataset.close()
            print("Dataset closed.")


def main():
    """
    Run the NetCDF parser for all files in ./data/raw
    and save extracted CSVs to ./data/csv
    """
    input_dir = Path("../data/raw")
    output_dir = Path("../data/csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    nc_files = list(input_dir.glob("*.nc"))

    if not nc_files:
        print(f"No NetCDF files found in {input_dir}")
        return

    print(f"Found {len(nc_files)} NetCDF files in {input_dir}")

    for nc_file in nc_files:
        print("\n" + "="*60)
        print(f"Processing file: {nc_file.name}")
        print("="*60)

        parser = NetCDFParser(nc_file)

        try:
            if parser.open_dataset():
                parser.inspect_file_structure()

                # Output CSV path in ./data/csv
                output_csv = output_dir / (nc_file.stem + "_argo_data.csv")

                # Extract ARGO data
                df = parser.extract_argo_data(output_file=output_csv)

                if df is not None:
                    print(f"✅ Saved CSV: {output_csv}")
                else:
                    print(f"⚠️ No data extracted for {nc_file.name}")
        finally:
            parser.close()


if __name__ == "__main__":
    main()
