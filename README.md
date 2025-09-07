# Oceanographic Data Analysis Chatbot: Technical Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Design](#architecture-design)
3. [Stage 1: Natural Language Processing](#stage-1-natural-language-processing)
4. [Stage 2: Data Fetching](#stage-2-data-fetching)
5. [Stage 3: TableLLM Analysis](#stage-3-tablellm-analysis)
6. [Stage 4: OceanGPT Interpretation](#stage-4-oceangpt-interpretation)
7. [Stage 5: Visualization](#stage-5-visualization)
8. [Orchestration and Security](#orchestration-and-security)
9. [API Documentation](#api-documentation)
10. [Implementation Guidelines](#implementation-guidelines)

---

## System Overview

The Oceanographic Data Analysis Chatbot implements a **five-stage agentic pipeline** that transforms natural language queries into comprehensive oceanographic analyses. The system processes user questions through specialized AI models, retrieves and analyzes Argo float data, and generates scientific interpretations with interactive visualizations.

### Core Pipeline Flow
```
Natural Language Query → SQL/Pandas Code → Data Retrieval → TableLLM Analysis → OceanGPT Interpretation → Visualizations
```

### Key Features
- **Multi-modal AI Integration**: Combines TableLLM, OceanGPT, and visualization-specific models
- **Comprehensive Data Access**: Supports multiple data sources from local databases to remote APIs
- **Scientific Rigor**: Implements quality control, provenance tracking, and verification loops
- **Interactive Outputs**: Generates both static and interactive visualizations with downloadable code

---

## Architecture Design

### High-Level Architecture

The system follows a **microservices approach** with five distinct processing stages:

1. **Natural Language Processing**: Intent classification and query translation
2. **Data Fetching**: Multi-source data retrieval and standardization
3. **Numeric Analysis**: TableLLM-powered statistical analysis
4. **Domain Interpretation**: OceanGPT oceanographic expertise
5. **Visualization**: Interactive plot generation and presentation

### Technology Stack

**Core AI Models:**
- **TableLLM**: Specialized tabular data analysis
- **OceanGPT**: Domain-specific oceanographic interpretation
- **GPT-4/Text-to-SQL**: Query generation and complex reasoning
- **Visualization LLM**: Plot code generation

**Data Infrastructure:**
- **PostgreSQL/PostGIS**: Primary geospatial database
- **Vector Database**: FAISS/Chroma for semantic search
- **NetCDF Storage**: Raw Argo float data archives
- **External APIs**: ERDDAP, OPeNDAP, Argovis, IFREMER GDAC

---

## Stage 1: Natural Language Processing

### Overview
Converts user natural language queries into executable database queries through a six-step process.

### Process Flow

#### 1.1 Intent Classification
**Purpose**: Categorize user queries into predefined oceanographic analysis types.

**Input**: Raw user natural language query
**Output**: Classified intent category

**Intent Categories:**
- `spatial`: Geographic/location-based queries
- `temporal`: Time-series and seasonal analyses
- `parameter`: Specific measurement queries (temperature, salinity, etc.)
- `compare`: Comparative analyses between regions/periods
- `trend`: Temporal trend and anomaly detection
- `viz`: Visualization-specific requests
- `meta`: Metadata and data availability queries

**Implementation Details:**
```python
# Example intent classification
def classify_intent(query):
    # Use NLP model to classify query intent
    # Return primary intent + confidence score
    return {
        "primary_intent": "temporal",
        "confidence": 0.92,
        "secondary_intents": ["parameter", "viz"]
    }
```

#### 1.2 Slot Extraction
**Purpose**: Extract structured parameters from natural language using specialized NLP tools.

**Tools Used:**
- **spaCy**: Named entity recognition and linguistic parsing
- **dateparser**: Flexible date/time extraction
- **geocoder**: Location name resolution and coordinate extraction
- **Custom synonyms**: Oceanographic parameter mapping

**Extracted Slots:**
- **Location**: Names, coordinates, or bounding boxes
- **Temporal**: Date ranges, seasons, specific periods
- **Spatial**: Depth ranges, geographic boundaries, radii
- **Parameters**: Temperature, salinity, pressure, derived variables
- **Aggregation**: Statistical operations (mean, min, max, percentiles)
- **Filters**: Quality control flags, instrument types, data modes
- **Output**: Sample sizes, visualization types, format preferences

**Example Extraction:**
```python
# Input: "Show me temperature trends in the North Atlantic from 2020 to 2023"
extracted_slots = {
    "location": {"name": "North Atlantic", "bbox": [-80, 0, -10, 70]},
    "date_range": {"start": "2020-01-01", "end": "2023-12-31"},
    "parameters": ["temperature"],
    "analysis_type": "trend",
    "viz_type": "timeseries"
}
```

#### 1.3 Canonicalization
**Purpose**: Standardize extracted parameters into system-compatible formats.

**Standardization Rules:**
- **Regions** → Bounding boxes with standard coordinate systems
- **Seasons** → Specific start/end date pairs
- **Parameters** → Canonical variable names (temp → sea_water_temperature)
- **Units** → Standard oceanographic units (°C, PSU, dbar)
- **Time zones** → UTC standardization

**Example Canonicalization:**
```python
# Before canonicalization
raw_slots = {
    "location": "Mediterranean Sea",
    "season": "winter",
    "parameter": "temp"
}

# After canonicalization
canonical_slots = {
    "bbox": [5.0, 30.0, 42.0, 47.0],
    "date_range": {"start": "2024-12-21", "end": "2025-03-19"},
    "parameters": ["sea_water_temperature"],
    "units": {"sea_water_temperature": "degree_Celsius"}
}
```

#### 1.4 Planning Decision
**Purpose**: Determine optimal data processing path based on query characteristics and data size.

**Decision Criteria:**
- **Small local subset** → Pandas processing path
- **Large/partitioned dataset** → PostgreSQL/PostGIS path
- **Geospatial queries** → PostGIS with spatial indices
- **Complex aggregations** → SQL with optimized queries

**Decision Logic:**
```python
def choose_processing_path(canonical_slots, estimated_data_size):
    if estimated_data_size < 10000 and not canonical_slots.get('geospatial_ops'):
        return 'pandas'
    elif canonical_slots.get('spatial_queries') or estimated_data_size > 100000:
        return 'sql'
    else:
        return 'hybrid'  # Combine approaches
```

#### 1.5 Translation
**Purpose**: Convert canonicalized parameters into executable code using table-aware AI models.

**Translation Models:**
- **TableLLM**: Specialized for complex tabular operations
- **Text-to-SQL LLM**: Optimized for database queries
- **Schema Injection**: Provide database schema and sample rows

**Output Format:**
- **Parameterized SQL**: Safe, injection-resistant queries
- **Pandas AST**: Abstract syntax tree for DataFrame operations
- **Metadata**: Execution parameters, limits, filters, provenance tokens

**Example Translation:**
```sql
-- Generated SQL with parameters
SELECT 
    profile_id, 
    lat, lon, date, 
    AVG(temperature) as mean_temp,
    COUNT(*) as sample_count
FROM argo_profiles 
WHERE 
    ST_Contains(ST_MakeEnvelope($1, $2, $3, $4, 4326), ST_Point(lon, lat))
    AND date BETWEEN $5 AND $6
    AND qc_flag IN ('1', '2')
GROUP BY profile_id, lat, lon, date
ORDER BY date;
```

#### 1.6 Validation
**Purpose**: Ensure generated queries are safe, syntactically correct, and semantically valid.

**Validation Checks:**
1. **Static Parsing**: AST analysis using sqlparse or pandas AST validation
2. **Dry Run**: EXPLAIN execution without data modification
3. **Safety Patterns**: Detect dangerous operations (DROP, DELETE, etc.)
4. **Resource Limits**: Validate query complexity and expected result size
5. **Schema Compliance**: Verify table and column references

**Validation Outcomes:**
- **PASS**: Query approved for execution
- **REJECT**: Query blocked due to safety or syntax issues
- **CLARIFY**: Request user clarification for ambiguous parameters

---

## Stage 2: Data Fetching

### Overview
Retrieves oceanographic data from multiple sources with intelligent source selection and comprehensive standardization.

### Data Source Priority

#### 2.1 Source Selection Strategy
**Priority Order (Ordered Preference):**

1. **Local PostgreSQL/PostGIS** (Primary)
   - Production-ready indexed data
   - Optimized for geospatial queries
   - Fastest access times

2. **Vector Database** (Secondary)
   - FAISS/Chroma for semantic similarity
   - Profile ID matching
   - Efficient for research queries

3. **NetCDF Object Store** (Tertiary)
   - Raw Argo float files
   - Complete measurement profiles
   - Required for specialized analysis

4. **External APIs** (Fallback)
   - ERDDAP: Real-time data access
   - OPeNDAP: Distributed data access
   - Argovis: Argo visualization service
   - IFREMER GDAC: Global Argo data center

#### 2.2 Execution Paths

**SQL Execution Path:**
```python
# For large datasets and geospatial queries
def execute_sql_query(parameterized_query, params):
    with database_connection() as conn:
        result = conn.execute(parameterized_query, params)
        return standardize_sql_result(result)
```

**Pandas Execution Path:**
```python
# For small subsets and complex processing
def execute_pandas_analysis(pandas_ast, netcdf_files):
    dataframes = []
    for file in netcdf_files:
        df = argopy.load_profile(file)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes)
    result = execute_ast(pandas_ast, combined_df)
    return standardize_pandas_result(result)
```

#### 2.3 Parallel Processing
- **Profile Fetching**: Concurrent retrieval of multiple Argo profiles
- **Chunked Operations**: Process large datasets in manageable chunks
- **Async I/O**: Non-blocking data access for external APIs

### 2.4 Data Standardization

#### Quality Control Filtering
```python
def apply_qc_filters(dataframe):
    """Apply Argo quality control standards."""
    # R/A/D mode filtering (Real-time/Adjusted/Delayed)
    df_filtered = dataframe[dataframe['mode'].isin(['A', 'D'])]
    
    # QC flag filtering (1=good, 2=probably good)
    df_filtered = df_filtered[df_filtered['qc_flag'].isin(['1', '2'])]
    
    return df_filtered
```

#### Unit Standardization
- **Temperature**: Celsius (°C)
- **Salinity**: Practical Salinity Units (PSU)
- **Pressure**: Decibars (dbar)
- **Depth**: Meters (m) calculated from pressure
- **Coordinates**: Decimal degrees (WGS84)
- **Time**: UTC ISO 8601 format

#### Provenance Addition
```python
def add_provenance(dataframe, source_info):
    """Add comprehensive data provenance."""
    dataframe['source_url'] = source_info['url']
    dataframe['file_name'] = source_info['filename']
    dataframe['profile_id'] = source_info['profile_id']
    dataframe['retrieval_time'] = datetime.utcnow().isoformat()
    dataframe['checksum'] = calculate_checksum(dataframe)
    return dataframe
```

### 2.5 Output Format

**Canonical DataFrame Structure:**
```python
canonical_columns = [
    'profile_id',      # Unique profile identifier
    'float_id',        # Argo float WMO number
    'lat',            # Latitude (decimal degrees)
    'lon',            # Longitude (decimal degrees)  
    'date',           # UTC timestamp
    'depth',          # Depth in meters
    'temperature',    # Sea water temperature (°C)
    'salinity',       # Practical salinity (PSU)
    'pressure',       # Sea water pressure (dbar)
    'qc_flags',       # Quality control flags
    'metadata'        # Additional measurement metadata
]
```

**Caching Strategy:**
- **Query hashing**: Generate unique hash for each query
- **Result caching**: Store processed results for common queries
- **TTL management**: Time-based cache expiration for data freshness

---

## Stage 3: TableLLM Analysis

### Overview
Performs comprehensive numeric analysis using specialized TableLLM with verification loops and fail-safe mechanisms.

### 3.1 Input Preparation

**TableLLM Input Package:**
```python
tablellm_input = {
    "schema": get_dataframe_schema(df),
    "data_head": df.head(100),  # Limited sample for LLM
    "shape": df.shape,
    "summary_stats": df.describe(),
    "user_intent": extracted_intent,
    "constraints": analysis_constraints,
    "required_metrics": specified_metrics,
    "few_shot_examples": domain_examples
}
```

### 3.2 Required TableLLM Outputs

**Structured JSON Output Format:**
```python
tablellm_output = {
    "analysis_plan": {
        "steps": [
            "Group by geographic region",
            "Calculate seasonal averages", 
            "Perform trend analysis",
            "Identify anomalies"
        ],
        "order": ["aggregation", "statistics", "trend_tests", "anomaly_detection"]
    },
    
    "executable_code": {
        "pandas_snippets": [...],  # Runnable pandas operations
        "sql_queries": [...],      # Alternative SQL approaches
        "verification_tests": [...] # Code to verify results
    },
    
    "numeric_results": {
        "aggregations": {...},     # Counts, means, medians, std devs
        "percentiles": {...},      # 5th, 25th, 50th, 75th, 95th percentiles
        "extremes": {...}          # Min/max values and locations
    },
    
    "diagnostics": {
        "nan_rates": {...},        # Missing data by parameter
        "qc_pass_rates": {...},    # Quality control statistics
        "sample_sizes": {...},     # Data coverage by region/time
        "data_quality_score": 0.85 # Overall quality assessment
    },
    
    "statistical_tests": {
        "trend_analysis": {
            "method": "Mann-Kendall",
            "slope": 0.02,           # Trend slope per year
            "p_value": 0.001,        # Statistical significance
            "confidence_interval": [0.015, 0.025]
        },
        "anomaly_detection": {
            "method": "z_score_threshold", 
            "threshold": 2.5,
            "anomalies_found": 15,
            "anomaly_locations": [...]
        }
    },
    
    "oceanographic_features": {
        "mixed_layer_depth": {...},     # MLD calculations
        "thermocline_depth": {...},     # Thermocline identification
        "vertical_gradients": {...},    # Temperature/salinity gradients
        "ts_clusters": {...}            # T-S diagram cluster analysis
    },
    
    "visualization_suggestions": [
        {
            "plot_type": "profile_comparison",
            "parameters": {"x": "temperature", "y": "depth", "color": "season"}
        },
        {
            "plot_type": "ts_diagram", 
            "parameters": {"x": "salinity", "y": "temperature", "color": "depth"}
        },
        {
            "plot_type": "geographic_map",
            "parameters": {"lat": "lat", "lon": "lon", "color": "mean_temperature"}
        }
    ]
}
```

### 3.3 Execution and Verification Loop

**Verification Process:**
```python
def verify_tablellm_results(llm_output, dataframe):
    """Verify LLM outputs against actual computation."""
    
    # Execute generated code
    computed_results = execute_code_snippets(
        llm_output['executable_code'], 
        dataframe
    )
    
    # Compare with LLM-reported results
    discrepancies = compare_results(
        computed_results, 
        llm_output['numeric_results'],
        tolerance=0.01
    )
    
    if discrepancies > tolerance_threshold:
        # Fail-safe: Use deterministic templates
        return execute_fallback_analysis(dataframe, llm_output['analysis_plan'])
    
    return computed_results
```

**Fail-Safe Mechanisms:**
1. **Code Execution Monitoring**: Track execution time and memory usage
2. **Result Validation**: Cross-check computed vs. reported values
3. **Fallback Templates**: Pre-built analysis functions for common operations
4. **Error Recovery**: Graceful degradation with partial results

### 3.4 Deliverables

**Stage 3 Outputs:**
- **Verified Numeric Tables**: Cross-validated statistical summaries
- **Generated Plots**: Preliminary visualizations (matplotlib objects)
- **Analysis Summary**: Concise LLM-written findings
- **Code Artifacts**: Reproducible analysis scripts
- **Quality Metrics**: Confidence scores and uncertainty estimates

---

## Stage 4: OceanGPT Interpretation

### Overview
Applies domain-specific oceanographic expertise to interpret numeric results and provide scientific context.

### 4.1 Input to OceanGPT

**Comprehensive Input Package:**
```python
oceangpt_input = {
    "numeric_summaries": verified_tablellm_results,
    "representative_profiles": {
        "temperature_profiles": sample_t_profiles,
        "salinity_profiles": sample_s_profiles, 
        "ts_relationships": ts_diagram_data
    },
    "spatial_temporal_context": {
        "location_metadata": geographic_info,
        "seasonal_context": temporal_info,
        "climatological_reference": historical_means
    },
    "quality_control_notes": {
        "data_flags": qc_summary,
        "instrument_info": float_metadata,
        "calibration_notes": adjustment_info
    },
    "provenance": data_source_tracking,
    "user_context": {
        "original_question": user_query,
        "follow_up_questions": tablellm_suggestions
    }
}
```

### 4.2 OceanGPT Analysis Tasks

#### Physical Interpretation
- **Water Mass Identification**: Classify T-S characteristics
- **Seasonal Cycle Analysis**: Interpret temporal patterns
- **Vertical Structure**: Analyze stratification and mixing
- **Circulation Patterns**: Identify currents and eddies

#### Anomaly Assessment
- **Climatological Comparison**: Compare against historical records
- **Statistical Significance**: Evaluate departure from normal
- **Physical Mechanisms**: Hypothesize causes of anomalies
- **Temporal Evolution**: Track changes over time

#### Quality Assessment
- **Sensor Drift Detection**: Identify potential calibration issues
- **Data Reliability**: Assess confidence in measurements
- **Spatial Representativeness**: Evaluate sampling adequacy
- **Temporal Consistency**: Check for systematic biases

#### Scientific Recommendations
- **Further Analysis**: Suggest additional investigations
- **Data Requirements**: Identify needed complementary datasets
- **Methodological Improvements**: Recommend analysis enhancements
- **Research Hypotheses**: Generate testable scientific questions

### 4.3 Structured Output Format

```python
oceangpt_output = {
    "executive_summary": "Concise 2-3 sentence overview of key findings",
    
    "physical_interpretation": {
        "water_masses": [
            {
                "name": "North Atlantic Deep Water",
                "characteristics": "T=2-4°C, S=34.9-35.0 PSU",
                "depth_range": "1500-4000m",
                "confidence": 0.85
            }
        ],
        "seasonal_patterns": {
            "description": "Strong seasonal thermocline development",
            "amplitude": "15°C surface temperature range",
            "timing": "Maximum stratification in September"
        },
        "circulation_features": {
            "identified": ["Gulf Stream meander", "Cold core eddy"],
            "evidence": "Temperature front at 40.5°N",
            "implications": "Enhanced mixing and productivity"
        }
    },
    
    "anomaly_assessment": {
        "detected_anomalies": [
            {
                "parameter": "surface_temperature",
                "magnitude": "+2.3°C above climatology", 
                "significance": "99.5% confidence",
                "spatial_extent": "Regional (>500km)",
                "duration": "Persistent (>3 months)"
            }
        ],
        "potential_causes": [
            "Marine heatwave event",
            "Altered atmospheric forcing",
            "Ocean circulation changes"
        ],
        "climatological_context": "Warmest recorded for this region/season"
    },
    
    "quality_assessment": {
        "data_reliability": 0.92,
        "potential_issues": [
            "Minor salinity calibration drift in float 1901234",
            "Sparse coverage in winter months"
        ],
        "recommendations": [
            "Cross-validate with nearby floats",
            "Apply delayed-mode calibration adjustments"
        ]
    },
    
    "scientific_recommendations": {
        "immediate_actions": [
            "Extend analysis to include dissolved oxygen",
            "Compare with satellite SST observations"
        ],
        "future_investigations": [
            "Multi-year trend analysis",
            "Biogeochemical impact assessment"
        ],
        "data_needs": [
            "Higher temporal resolution in target region",
            "Complementary BGC-Argo measurements"
        ]
    },
    
    "uncertainty_quantification": {
        "confidence_scores": {
            "water_mass_identification": 0.85,
            "anomaly_detection": 0.93,
            "trend_significance": 0.78
        },
        "limiting_factors": [
            "Sparse spatial coverage",
            "Short time series length",
            "Seasonal sampling bias"
        ]
    },
    
    "citations_and_references": [
        "Roemmich et al. (2019) - Argo climatology",
        "Holte et al. (2017) - Mixed layer algorithms",
        "Wong et al. (2020) - Argo quality control"
    ]
}
```

### 4.4 Iteration Mechanism

**Adaptive Analysis Loop:**
```python
def oceangpt_iteration_loop(initial_results, user_feedback):
    """Handle requests for additional analysis."""
    
    if oceangpt_output['confidence_scores']['overall'] < 0.7:
        # Request additional data slices
        additional_data = fetch_supporting_profiles(
            extended_region=True,
            temporal_expansion=True
        )
        
        # Re-run TableLLM analysis with expanded dataset
        enhanced_analysis = tablellm_analyze(additional_data)
        
        # Update OceanGPT interpretation
        updated_interpretation = oceangpt_reanalyze(
            enhanced_analysis, 
            initial_results
        )
        
        return updated_interpretation
    
    return initial_results
```

---

## Stage 5: Visualization

### Overview
Generates publication-quality visualizations with interactive capabilities and reproducible code artifacts.

### 5.1 Visualization Engine Selection

**Engine Decision Matrix:**
- **Plotly**: Interactive dashboards, web deployment
- **Matplotlib**: Static publication figures, precise control
- **Folium/Leaflet**: Geographic maps with interactivity
- **Bokeh**: Advanced interactive analytics

**Selection Criteria:**
```python
def select_visualization_engine(plot_requirements):
    if plot_requirements['interactive'] and plot_requirements['web_deploy']:
        return 'plotly'
    elif plot_requirements['publication_quality']:
        return 'matplotlib'
    elif plot_requirements['geographic'] and plot_requirements['interactive']:
        return 'folium'
    else:
        return 'bokeh'  # Advanced analytics
```

### 5.2 Visualization Parameter Determination

**Required Parameters (Explicit Specification):**
```python
visualization_params = {
    "plot_type": "profile_comparison",  # profile, overlay, TS, map, heatmap, timeseries
    "axes": {
        "x_field": "temperature",
        "y_field": "depth", 
        "z_field": None,           # For 3D plots
        "depth_inversion": True,    # Oceanographic convention
        "units": {"x": "°C", "y": "m"}
    },
    "styling": {
        "colormap": "thermal",      # Oceanography-appropriate
        "colorbar_label": "Temperature (°C)",
        "marker_size": "auto",      # Based on data density
        "line_width": 2.0
    },
    "interactivity": {
        "hover_fields": ["lat", "lon", "date", "qc_flag"],
        "zoom_controls": True,
        "selection_tools": ["box_select", "lasso_select"]
    },
    "geographic": {
        "coastline_provider": "naturalearthdata",
        "basemap": "ocean_focused",
        "projection": "PlateCarree"    # Geographic coordinates
    },
    "output": {
        "format": ["png", "svg", "html"], # Multiple formats
        "width": 1200,
        "height": 800,
        "dpi": 300                  # Publication quality
    }
}
```

### 5.3 Code Generation and Execution

**Visualization Generation Flow:**
```python
def generate_visualization(data, params, oceangpt_context):
    """Generate publication-quality visualizations."""
    
    # 1. Parameter validation and defaults
    validated_params = validate_viz_params(params)
    
    # 2. Code generation using visualization LLM
    viz_code = visualization_llm.generate_code(
        data_schema=data.dtypes,
        sample_data=data.head(10),
        plot_params=validated_params,
        domain_context=oceangpt_context
    )
    
    # 3. Code safety validation
    safe_code = validate_and_sanitize_code(viz_code)
    
    # 4. Sandboxed execution
    try:
        plot_result = execute_in_sandbox(safe_code, data)
        return plot_result
    except Exception as e:
        # Fallback to deterministic templates
        return execute_fallback_template(data, validated_params)
```

**Auto-Fix Mechanism:**
```python
def auto_fix_visualization(code, error, data):
    """Automatically fix common visualization errors."""
    
    common_fixes = {
        "MemoryError": lambda c: add_data_sampling(c, sample_rate=0.1),
        "KeyError": lambda c: fix_column_references(c, data.columns),
        "ValueError": lambda c: add_data_validation(c),
        "TimeoutError": lambda c: optimize_rendering(c)
    }
    
    if type(error).__name__ in common_fixes:
        fixed_code = common_fixes[type(error).__name__](code)
        return execute_in_sandbox(fixed_code, data)
    
    # Fall back to deterministic template
    return use_template_visualization(data)
```

### 5.4 Output Presentation

**Integrated Presentation Format:**
```python
presentation_output = {
    "visualizations": [
        {
            "id": "temp_profile_comparison",
            "type": "interactive_plot",
            "url": "https://plots.example.com/temp_profiles.html",
            "static_url": "https://plots.example.com/temp_profiles.png", 
            "caption": "Temperature profiles showing seasonal stratification",
            "code": "# Reproducible plotting code\nimport matplotlib.pyplot as plt...",
            "data_subset": "temp_profiles_sample.csv"
        }
    ],
    
    "numeric_tables": [
        {
            "title": "Regional Temperature Statistics",
            "data": tablellm_verified_results,
            "format": "html_table",
            "download_formats": ["csv", "xlsx", "json"]
        }
    ],
    
    "interpretive_text": {
        "executive_summary": oceangpt_output['executive_summary'],
        "detailed_analysis": oceangpt_output['physical_interpretation'],
        "confidence_notes": oceangpt_output['uncertainty_quantification']
    },
    
    "provenance": {
        "query_id": unique_identifier,
        "source_files": data_provenance,
        "processing_timestamp": datetime.utcnow().isoformat(),
        "reproducibility_package": "analysis_reproduction.zip"
    }
}
```

---

## Orchestration and Security

### Overview
Comprehensive system management ensuring security, performance, and reliability through multi-layered architecture.

### Agent Controller Architecture

**Tool-Based Framework Implementation:**
```python
from langchain.agents import Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory

# Define specialized tools
tools = [
    Tool(
        name="nl_parser",
        description="Parse natural language queries into structured intents",
        func=natural_language_parser
    ),
    Tool(
        name="schema_service", 
        description="Provide database schema and metadata",
        func=get_database_schema
    ),
    Tool(
        name="text2sql_translator",
        description="Convert natural language to SQL queries",
        func=text_to_sql_converter
    ),
    Tool(
        name="tablellm_analyzer",
        description="Perform comprehensive numeric analysis",
        func=tablellm_analysis_tool
    ),
    Tool(
        name="oceangpt_interpreter",
        description="Provide oceanographic domain interpretation", 
        func=oceangpt_interpretation_tool
    ),
    Tool(
        name="visualization_generator",
        description="Generate interactive oceanographic visualizations",
        func=visualization_generation_tool
    ),
    Tool(
        name="provenance_logger",
        description="Track data lineage and processing history",
        func=log_provenance
    )
]

# Initialize agent with memory and error handling
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=oceanographic_agent,
    tools=tools,
    memory=ConversationBufferMemory(return_messages=True),
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)
```

### Security Framework

#### Multi-Layer Security Architecture

**1. Query Security Layer:**
```python
def secure_query_execution(sql_query, parameters):
    """Ensure all queries are safe and parameterized."""
    
    # Validate SQL structure
    if not is_parameterized_query(sql_query):
        raise SecurityError("Only parameterized queries allowed")
    
    # Whitelist table and column access
    if not validate_table_access(sql_query, allowed_tables):
        raise SecurityError("Unauthorized table access attempt")
    
    # Parameter sanitization
    sanitized_params = sanitize_parameters(parameters)
    
    return execute_query(sql_query, sanitized_params)
```

**2. Code Execution Security:**
```python
class SecureCodeExecutor:
    """Sandboxed environment for LLM-generated code execution."""
    
    def __init__(self):
        self.allowed_modules = [
            'pandas', 'numpy', 'matplotlib', 'plotly', 
            'scipy', 'statsmodels', 'sklearn'
        ]
        self.forbidden_operations = [
            'exec', 'eval', 'open', '__import__', 
            'subprocess', 'os.system'
        ]
    
    def execute(self, code_string, data_context):
        """Execute code in restricted environment."""
        
        # Static analysis for forbidden patterns
        if self.contains_forbidden_ops(code_string):
            raise SecurityError("Forbidden operations detected")
        
        # Create isolated namespace
        safe_namespace = self.create_safe_namespace(data_context)
        
        # Execute with timeout and memory limits
        with resource_limits(timeout=30, memory="1GB"):
            result = exec(code_string, safe_namespace)
        
        return result
```

**3. Data Access Control:**
```python
def enforce_data_access_policy(user_context, requested_data):
    """Implement fine-grained data access controls."""
    
    # Check user permissions
    if not user_context.has_permission('oceanographic_data'):
        raise PermissionError("Insufficient data access rights")
    
    # Validate spatial/temporal bounds
    if requested_data['bbox'] not in user_context.allowed_regions:
        raise PermissionError("Geographic access restriction")
    
    # Apply data usage limits
    if requested_data['estimated_size'] > user_context.quota_remaining:
        raise ResourceError("Data quota exceeded")
    
    return True
```

### Performance Optimization

#### Database Performance Strategy
```python
# Spatial indexing for geographic queries
CREATE INDEX idx_argo_spatial ON argo_profiles 
USING GIST(ST_Point(longitude, latitude));

# Temporal partitioning for time-series efficiency  
CREATE TABLE argo_profiles_2024 PARTITION OF argo_profiles
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

# Composite indices for common query patterns
CREATE INDEX idx_argo_temp_spatial_temporal ON argo_profiles
(date, ST_Point(longitude, latitude)) 
WHERE temperature IS NOT NULL;
```

#### Caching Strategy Implementation
```python
import redis
import hashlib
import pickle

class IntelligentCache:
    """Multi-level caching for query results and computations."""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379)
        self.local_cache = {}
        
    def generate_cache_key(self, query_spec):
        """Generate deterministic cache key from query specification."""
        canonical_spec = self.canonicalize_query(query_spec)
        return hashlib.sha256(
            pickle.dumps(canonical_spec, protocol=4)
        ).hexdigest()
    
    def get_cached_result(self, cache_key):
        """Retrieve cached result with fallback hierarchy."""
        
        # Level 1: Local memory cache (fastest)
        if cache_key in self.local_cache:
            return self.local_cache[cache_key]
        
        # Level 2: Redis cache (fast)
        redis_result = self.redis_client.get(cache_key)
        if redis_result:
            result = pickle.loads(redis_result)
            self.local_cache[cache_key] = result  # Promote to L1
            return result
        
        return None  # Cache miss
    
    def store_result(self, cache_key, result, ttl=3600):
        """Store result with appropriate TTL."""
        
        # Store in both levels
        self.local_cache[cache_key] = result
        self.redis_client.setex(
            cache_key, 
            ttl, 
            pickle.dumps(result, protocol=4)
        )
```

### Quality Assurance Framework

#### Automated QA Gates
```python
class QualityAssuranceGates:
    """Automated quality checks throughout the pipeline."""
    
    def __init__(self):
        self.thresholds = {
            'numeric_tolerance': 0.01,
            'minimum_sample_size': 100,
            'qc_pass_rate': 0.8,
            'confidence_threshold': 0.7
        }
    
    def validate_numeric_results(self, computed, reported):
        """Ensure numeric consistency between computed and reported values."""
        
        for key in reported.keys():
            if key in computed:
                relative_error = abs(computed[key] - reported[key]) / abs(reported[key])
                if relative_error > self.thresholds['numeric_tolerance']:
                    return QAResult(
                        passed=False,
                        message=f"Numeric discrepancy in {key}: {relative_error:.3f}"
                    )
        
        return QAResult(passed=True, message="Numeric validation passed")
    
    def validate_sample_adequacy(self, data_summary):
        """Ensure sufficient data for reliable analysis."""
        
        total_samples = data_summary.get('total_profiles', 0)
        if total_samples < self.thresholds['minimum_sample_size']:
            return QAResult(
                passed=False,
                message=f"Insufficient data: {total_samples} < {self.thresholds['minimum_sample_size']}"
            )
        
        qc_pass_rate = data_summary.get('qc_pass_rate', 0.0)
        if qc_pass_rate < self.thresholds['qc_pass_rate']:
            return QAResult(
                passed=False,
                message=f"Poor data quality: {qc_pass_rate:.2f} < {self.thresholds['qc_pass_rate']}"
            )
        
        return QAResult(passed=True, message="Sample adequacy validated")
    
    def evaluate_confidence_levels(self, oceangpt_output):
        """Assess overall analysis confidence."""
        
        avg_confidence = sum(oceangpt_output['confidence_scores'].values()) / len(oceangpt_output['confidence_scores'])
        
        if avg_confidence < self.thresholds['confidence_threshold']:
            return QAResult(
                passed=False,
                message=f"Low confidence analysis: {avg_confidence:.2f}",
                recommendation="Human review recommended"
            )
        
        return QAResult(passed=True, message=f"High confidence analysis: {avg_confidence:.2f}")
```

#### Human Review Integration
```python
class HumanReviewSystem:
    """Integration point for human expert validation."""
    
    def trigger_human_review(self, analysis_result, trigger_reason):
        """Queue analysis for human expert review."""
        
        review_package = {
            'analysis_id': generate_unique_id(),
            'trigger_reason': trigger_reason,
            'confidence_scores': analysis_result['confidence_scores'],
            'qa_flags': analysis_result['qa_results'],
            'data_summary': analysis_result['data_provenance'],
            'review_priority': self.calculate_priority(analysis_result),
            'expert_domain': self.assign_expert_domain(analysis_result)
        }
        
        # Queue for expert review
        self.review_queue.add(review_package)
        
        # Notify relevant experts
        self.notify_experts(review_package)
        
        return review_package['analysis_id']
```

---

## API Documentation

### Overview
RESTful API providing programmatic access to the oceanographic analysis pipeline with comprehensive input/output contracts.

### Core API Contract

#### Request Format
```python
# POST /api/v1/analyze
{
    "user_text": "Show temperature trends in the North Atlantic from 2020 to 2023",
    "session_id": "unique-session-identifier", 
    "options": {
        "output_format": ["json", "html", "csv"],
        "visualization_types": ["interactive", "static"],
        "quality_threshold": 0.8,
        "max_processing_time": 300,
        "include_provenance": True
    }
}
```

#### Response Format
```python
{
    "status": "success",
    "processing_time": 45.3,
    "query_id": "unique-query-identifier",
    
    "query_specification": {
        "parsed_intent": "temporal_trend",
        "canonical_parameters": {
            "location": {"bbox": [-80, 0, -10, 70]},
            "temporal_range": {"start": "2020-01-01", "end": "2023-12-31"},
            "parameters": ["sea_water_temperature"],
            "aggregation": "monthly_mean"
        },
        "execution_plan": {
            "data_path": "sql",
            "estimated_records": 15420,
            "processing_strategy": "chunked_parallel"
        }
    },
    
    "verified_table_summary": {
        "total_profiles": 15420,
        "temporal_coverage": "2020-01-01 to 2023-12-31",
        "spatial_coverage": "North Atlantic Basin",
        "quality_metrics": {
            "qc_pass_rate": 0.92,
            "completeness": 0.87,
            "data_quality_score": 0.89
        },
        "statistical_summary": {
            "temperature": {
                "mean": 12.45,
                "std": 8.32,
                "min": -1.2,
                "max": 28.7,
                "trend_slope": 0.023,
                "trend_p_value": 0.001
            }
        }
    },
    
    "analysis_results": {
        "numeric_findings": {
            "significant_trends": [
                {
                    "parameter": "temperature",
                    "trend_magnitude": "+0.023°C/year",
                    "statistical_significance": 0.001,
                    "spatial_pattern": "Basin-wide warming"
                }
            ],
            "detected_anomalies": [
                {
                    "event": "Marine heatwave 2022",
                    "magnitude": "+3.2°C above climatology",
                    "duration": "June-September 2022",
                    "spatial_extent": "Gulf Stream region"
                }
            ]
        },
        "oceanographic_features": {
            "identified_patterns": [
                "Strengthening of seasonal thermocline",
                "Northward shift of thermal fronts",
                "Enhanced upper ocean stratification"
            ]
        }
    },
    
    "ocean_report": {
        "executive_summary": "Analysis reveals significant warming trend in North Atlantic surface waters with notable marine heatwave events.",
        "physical_interpretation": {
            "dominant_processes": [
                "Atmospheric forcing changes",
                "Gulf Stream variability", 
                "Arctic warming influence"
            ],
            "water_mass_changes": [
                {
                    "water_mass": "Subtropical Mode Water",
                    "observed_change": "2-3°C warming in formation region",
                    "implications": "Reduced oxygen solubility, ecosystem impacts"
                }
            ]
        },
        "confidence_assessment": {
            "overall_confidence": 0.87,
            "data_quality": 0.89,
            "spatial_representation": 0.85,
            "temporal_adequacy": 0.88
        },
        "recommendations": [
            "Extend analysis to include subsurface warming patterns", 
            "Investigate biogeochemical implications",
            "Compare with coupled climate model projections"
        ]
    },
    
    "visualizations": [
        {
            "id": "temp_trend_timeseries",
            "type": "interactive_timeseries",
            "title": "North Atlantic Temperature Trends 2020-2023",
            "url": "https://api.oceanbot.com/plots/temp_trend_12345.html",
            "static_url": "https://api.oceanbot.com/plots/temp_trend_12345.png",
            "download_code": "https://api.oceanbot.com/code/temp_trend_12345.py",
            "caption": "Monthly mean sea surface temperature anomalies showing warming trend",
            "metadata": {
                "plot_type": "time_series",
                "data_points": 48,
                "spatial_resolution": "2° × 2°",
                "temporal_resolution": "monthly"
            }
        },
        {
            "id": "spatial_warming_map", 
            "type": "interactive_map",
            "title": "Spatial Pattern of Temperature Trends",
            "url": "https://api.oceanbot.com/plots/spatial_map_12346.html",
            "static_url": "https://api.oceanbot.com/plots/spatial_map_12346.png",
            "download_code": "https://api.oceanbot.com/code/spatial_map_12346.py",
            "caption": "Geographic distribution of temperature trends (°C/decade)"
        }
    ],
    
    "provenance": {
        "data_sources": [
            {
                "source": "Argo Global Ocean Observatory",
                "profiles_used": 15420,
                "date_range": "2020-01-01 to 2023-12-31",
                "quality_control": "Real-time and delayed-mode",
                "access_method": "Local PostgreSQL database"
            }
        ],
        "processing_pipeline": [
            {
                "stage": "natural_language_processing",
                "model": "GPT-4",
                "processing_time": 2.1
            },
            {
                "stage": "data_retrieval", 
                "source": "PostgreSQL",
                "records_retrieved": 15420,
                "processing_time": 8.7
            },
            {
                "stage": "numeric_analysis",
                "model": "TableLLM-8B",
                "processing_time": 18.3
            },
            {
                "stage": "domain_interpretation",
                "model": "OceanGPT-7B",
                "processing_time": 12.4
            },
            {
                "stage": "visualization",
                "engine": "Plotly",
                "plots_generated": 2,
                "processing_time": 3.8
            }
        ],
        "reproducibility": {
            "query_hash": "sha256:abc123def456...",
            "data_checksum": "md5:789ghi012jkl...",
            "code_archive": "https://api.oceanbot.com/reproduce/12345.zip",
            "environment": "oceanbot-v1.2.3"
        }
    }
}
```

#### Error Response Format
```python
{
    "status": "error",
    "error_code": "INSUFFICIENT_DATA",
    "message": "Insufficient Argo profiles found for reliable analysis",
    "details": {
        "requested_region": "North Atlantic", 
        "requested_period": "2020-2023",
        "profiles_found": 23,
        "minimum_required": 100
    },
    "suggestions": [
        "Expand geographic search region",
        "Extend temporal search window", 
        "Reduce spatial/temporal resolution requirements"
    ],
    "query_id": "failed-query-identifier"
}
```

### Streaming API for Long-Running Analyses

```python
# WebSocket connection for real-time progress updates
ws://api.oceanbot.com/stream/analyze

# Progress messages
{
    "type": "progress",
    "stage": "data_retrieval", 
    "progress": 0.45,
    "message": "Retrieved 6,890 of 15,420 profiles",
    "estimated_completion": "2024-01-15T14:32:00Z"
}

{
    "type": "partial_result",
    "stage": "tablellm_analysis",
    "preliminary_findings": {
        "trend_detected": True,
        "trend_magnitude": "+0.021°C/year",
        "confidence": 0.83
    }
}

{
    "type": "completion",
    "stage": "final",
    "result_url": "https://api.oceanbot.com/results/12345",
    "processing_summary": {
        "total_time": 67.4,
        "stages_completed": 5,
        "quality_score": 0.89
    }
}
```

---

## Implementation Guidelines

### Development Phases

#### Phase 1: Core Infrastructure (Weeks 1-4)
1. **Database Setup**: PostgreSQL/PostGIS with Argo data ingestion
2. **Basic NLP Pipeline**: Intent classification and slot extraction  
3. **Data Retrieval**: Multi-source data access with standardization
4. **Security Framework**: Parameterized queries and access controls

#### Phase 2: AI Integration (Weeks 5-8)
1. **TableLLM Integration**: Numeric analysis with verification loops
2. **Basic Visualization**: Static plot generation with matplotlib
3. **Quality Assurance**: Automated validation and error handling
4. **API Development**: RESTful interface with basic functionality

#### Phase 3: Advanced Features (Weeks 9-12)
1. **OceanGPT Integration**: Domain-specific interpretation
2. **Interactive Visualizations**: Plotly and web-based outputs
3. **Performance Optimization**: Caching and parallel processing
4. **Human Review System**: Expert validation workflows

#### Phase 4: Production Deployment (Weeks 13-16)
1. **Load Testing**: Performance validation under realistic loads
2. **Monitoring**: Comprehensive logging and alerting
3. **Documentation**: User guides and API documentation
4. **Training**: Expert user onboarding and feedback integration

### Technical Requirements

#### Hardware Specifications
```yaml
Minimum Production Environment:
  CPU: 32 cores (64 vCPUs)
  RAM: 128 GB
  Storage: 2 TB NVMe SSD
  GPU: NVIDIA A100 40GB (for LLM inference)
  Network: 10 Gbps connection

Database Server:
  CPU: 16 cores (32 vCPUs) 
  RAM: 64 GB
  Storage: 5 TB SSD (RAID 10)
  Network: 10 Gbps connection
```

#### Software Dependencies
```yaml
Core Runtime:
  - Python 3.9+
  - PostgreSQL 14+ with PostGIS 3.2+
  - Redis 6.0+ (caching)
  - Docker 20.10+ (containerization)

Python Libraries:
  - pandas >= 2.0.0
  - numpy >= 1.24.0
  - xarray >= 2023.1.0
  - argopy >= 0.1.13
  - sqlalchemy >= 2.0.0
  - plotly >= 5.14.0
  - matplotlib >= 3.7.0
  - scikit-learn >= 1.2.0
  - langchain >= 0.0.200
  - transformers >= 4.30.0

AI Model Dependencies:
  - torch >= 2.0.0
  - transformers >= 4.30.0
  - sentence-transformers >= 2.2.0
  - faiss-cpu >= 1.7.4
```

### Deployment Architecture

```yaml
Production Deployment (Kubernetes):
  
  api-gateway:
    replicas: 3
    resources:
      cpu: "2"
      memory: "4Gi"
    
  nlp-service:
    replicas: 2  
    resources:
      cpu: "4"
      memory: "8Gi"
      
  tablellm-service:
    replicas: 1
    resources:
      cpu: "8"
      memory: "16Gi" 
      gpu: "1"
      
  oceangpt-service:
    replicas: 1
    resources:
      cpu: "8"
      memory: "16Gi"
      gpu: "1"
      
  visualization-service:
    replicas: 2
    resources:
      cpu: "4"
      memory: "8Gi"
      
  data-service:
    replicas: 3
    resources:
      cpu: "2"
      memory: "8Gi"

  postgresql:
    replicas: 1
    resources:
      cpu: "16"
      memory: "64Gi"
      storage: "5Ti"
      
  redis:
    replicas: 1  
    resources:
      cpu: "2"
      memory: "8Gi"
      storage: "100Gi"
```

### Monitoring and Observability

```python
# Comprehensive monitoring configuration
monitoring_config = {
    "performance_metrics": [
        "request_latency_p95",
        "throughput_requests_per_second", 
        "error_rate_percentage",
        "ai_model_inference_time",
        "database_query_time", 
        "cache_hit_ratio"
    ],
    
    "business_metrics": [
        "successful_analyses_per_day",
        "user_satisfaction_score",
        "data_quality_score", 
        "expert_review_rate",
        "cost_per_analysis"
    ],
    
    "alerts": [
        {
            "metric": "error_rate",
            "threshold": 0.05,
            "action": "page_on_call_engineer"
        },
        {
            "metric": "latency_p95", 
            "threshold": 120,
            "action": "scale_services"
        },
        {
            "metric": "data_quality_score",
            "threshold": 0.8,
            "action": "trigger_expert_review"
        }
    ]
}
```

This comprehensive documentation provides the foundation for implementing a robust, scalable oceanographic data analysis chatbot that combines cutting-edge AI capabilities with rigorous scientific standards and operational excellence.