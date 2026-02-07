# Data Directory

Place your CSV files here:

## Required Files

### 1. shipments.csv

Expected columns:
- data, trip_creation_time, route_schedule_uuid, route_type, trip_uuid
- source_center, source_name, destination_center, destination_name
- od_start_time, od_end_time, start_scan_to_end_scan
- is_cutoff, cutoff_factor, cutoff_timestamp
- actual_distance_to_destination, actual_time
- osrm_time, osrm_distance, factor
- segment_actual_time, segment_osrm_time, segment_osrm_distance, segment_factor

### 2. weather.csv

Expected columns:
- city_id, name, state, country, lon, lat
- temperature, pressure, humidity, temp_min, temp_max
- visibility, wind_speed, wind_deg, wind_gust
- clouds_all, weather_main, weather_description
- rain_1h, rain_3h, snow_1h, snow_3h, timezone

## Notes

- Make sure CSV files are properly formatted
- Column names should match exactly as specified
- Missing values will be handled by the backend
