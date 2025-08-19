from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union
import threading
import uuid
import bisect
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import sys

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Create custom logging handler that supports Unicode on Windows
class UnicodeStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        if sys.platform == 'win32' and stream is None:
            # Use UTF-8 encoding for Windows console
            self.stream = sys.stdout
            if hasattr(self.stream, 'reconfigure'):
                self.stream.reconfigure(encoding='utf-8')

# Alternative: Use simpler emojis that work better on Windows
def safe_log_message(message: str) -> str:
    """Replace problematic Unicode characters with ASCII alternatives for Windows."""
    replacements = {
        '❌': '[X]',
        '✅': '[OK]',
        '📊': '[STATS]',
        '🎯': '[TARGET]',
        '⚠️': '[WARNING]',
        '💡': '[TIP]',
        '🚗': '[CAR]',
        '📂': '[FOLDER]',
        '🔗': '[LINK]',
        '🔍': '[SEARCH]',
        '🌐': '[WEB]'
    }
    
    for unicode_char, ascii_replacement in replacements.items():
        message = message.replace(unicode_char, ascii_replacement)
    
    return message

# Setup logging with Unicode support for Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        UnicodeStreamHandler(),  # Unicode-safe console output
        logging.FileHandler('diagnostic.log', encoding='utf-8')  # UTF-8 file output
    ]
)
logger = logging.getLogger(__name__)

# Simple progress tracking (stores in memory)
progress_store = {}

def json_serializable(obj):
    """Convert numpy/pandas types to Python native types for JSON serialization."""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, (pd.Int64Dtype, pd.Float64Dtype)):
        return int(obj) if 'Int' in str(type(obj)) else float(obj)
    return obj

class WebDiagnosticProcessor:
    """
    Processor that captures diagnostic info for both console AND web display.
    """
    
    def __init__(self, task_id, max_workers=None):
        self.task_id = task_id
        self.stats = {
            'total_entries': 0,
            'processed_entries': 0,
            'activities_retained': 0,
            'visits_retained': 0,
            'position_fixes_retained': 0,
            'entries_filtered_out': 0,
            'accuracy_filtered': 0,
            'outliers_removed': 0
        }
        self.debug_stats = {
            'activities_found': 0,
            'activities_distance_filtered': 0,
            'visits_found': 0,
            'visits_duration_filtered': 0,
            'visits_probability_filtered': 0,
            'timeline_paths_found': 0,
            'coordinate_parse_failures': 0,
            'timestamp_parse_failures': 0,
            'processing_errors': 0
        }
        self.modes_seen = set()
        self.progress_callback = None
        self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
        
        # Store diagnostic messages for web display
        self.diagnostic_log = []
    
    def log_diagnostic(self, message: str, level: str = "INFO"):
        """Log message to both console and web diagnostic log."""
        # Make message Windows-safe
        safe_message = safe_log_message(message)
        
        # Log to console with safe message
        if level == "INFO":
            logger.info(safe_message)
        elif level == "WARNING":
            logger.warning(safe_message)
        elif level == "ERROR":
            logger.error(safe_message)
        
        # Store original message for web display (browsers handle Unicode fine)
        self.diagnostic_log.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'level': level,
            'message': message  # Keep original Unicode for web
        })
        
        # Update progress store with latest diagnostics
        if self.task_id in progress_store:
            progress_store[self.task_id]['diagnostics'] = self.diagnostic_log[-20:]  # Keep last 20 messages
    
    def set_progress_callback(self, callback):
        """Set callback function for progress updates."""
        self.progress_callback = callback
    
    def update_progress(self, message: str, percentage: float = None):
        """Update progress with optional percentage."""
        if self.progress_callback:
            self.progress_callback(message, percentage)
        self.log_diagnostic(f"PROGRESS: {message}")
    
    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points in meters."""
        if any(pd.isna([lat1, lon1, lat2, lon2])):
            return 0.0
            
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Earth radius in kilometers
        return c * r * 1000  # meters
    
    @staticmethod
    def preserve_timestamp_format(timestamp: str) -> str:
        """Preserve the original timestamp format for analyzer compatibility."""
        return timestamp
    
    def parse_coordinates(self, coord_string: str) -> Optional[Tuple[float, float]]:
        """Parse coordinate string with validation and debugging."""
        try:
            if not coord_string or not coord_string.startswith('geo:'):
                self.debug_stats['coordinate_parse_failures'] += 1
                return None
            coords = coord_string.replace('geo:', '').split(',')
            if len(coords) != 2:
                self.debug_stats['coordinate_parse_failures'] += 1
                return None
            lat, lon = float(coords[0]), float(coords[1])
            
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                self.debug_stats['coordinate_parse_failures'] += 1
                return None
            return lat, lon
        except (ValueError, IndexError):
            self.debug_stats['coordinate_parse_failures'] += 1
            return None
    
    def analyze_sample_data(self, full_data: List[Dict]) -> Dict:
        """Analyze a sample of the data to understand structure."""
        sample_size = min(1000, len(full_data) // 10)
        sample_indices = list(range(0, len(full_data), max(1, len(full_data) // sample_size)))
        
        analysis = {
            'entry_types': {'activity': 0, 'visit': 0, 'timelinePath': 0, 'other': 0},
            'date_range': {'first': None, 'last': None},
            'sample_activities': [],
            'sample_visits': [],
            'modes_found': set()
        }
        
        dates_found = []
        
        for idx in sample_indices:
            entry = full_data[idx]
            if not entry or not isinstance(entry, dict):
                continue
            
            # Count entry types and collect samples
            if 'activity' in entry:
                analysis['entry_types']['activity'] += 1
                if len(analysis['sample_activities']) < 3:
                    activity = entry['activity']
                    distance = activity.get('distanceMeters', 0)
                    activity_type = activity.get('topCandidate', {}).get('type', 'unknown')
                    analysis['sample_activities'].append({
                        'distance': distance,
                        'type': activity_type,
                        'start': activity.get('start', ''),
                        'end': activity.get('end', '')
                    })
                    analysis['modes_found'].add(activity_type)
                    
            elif 'visit' in entry:
                analysis['entry_types']['visit'] += 1
                if len(analysis['sample_visits']) < 3:
                    visit = entry['visit']
                    # FIX: Convert probability to float before storing
                    try:
                        probability = float(visit.get('probability', 0))
                    except (ValueError, TypeError):
                        probability = 0.0
                    
                    # Calculate duration
                    try:
                        start_dt = pd.to_datetime(entry['startTime'], utc=True)
                        end_dt = pd.to_datetime(entry['endTime'], utc=True)
                        duration = (end_dt - start_dt).total_seconds()
                    except:
                        duration = 0
                    
                    analysis['sample_visits'].append({
                        'duration': duration,
                        'probability': probability,  # Now guaranteed to be float
                        'location': visit.get('topCandidate', {}).get('placeLocation', '')
                    })
                    
            elif 'timelinePath' in entry:
                analysis['entry_types']['timelinePath'] += 1
            else:
                analysis['entry_types']['other'] += 1
            
            # Collect dates
            start_time_str = entry.get('startTime')
            if start_time_str:
                try:
                    entry_dt = pd.to_datetime(start_time_str, utc=True)
                    dates_found.append(entry_dt)
                except:
                    pass
        
        if dates_found:
            dates_found.sort()
            analysis['date_range']['first'] = dates_found[0]
            analysis['date_range']['last'] = dates_found[-1]
        
        return analysis
    
    def process_file_web_diagnostic(self, input_file: str, settings: Dict) -> Dict:
        """Process file with streamlined web-visible diagnostics focused on key metrics."""
        try:
            self.update_progress("Loading file...", 5)
            
            # Load file
            with open(input_file, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
            
            total_entries = len(full_data)
            self.stats['total_entries'] = total_entries
            self.log_diagnostic(f"📊 Loaded {total_entries:,} total entries")
            
            # Quick date range filtering
            self.update_progress("Filtering by date range...", 20)
            from_dt = pd.to_datetime(settings['from_date'], utc=True)
            to_dt = pd.to_datetime(settings['to_date'], utc=True) + pd.Timedelta(days=1)
            
            self.log_diagnostic(f"🎯 Target date range: {from_dt.date()} to {to_dt.date()}")
            
            relevant_data = []
            for entry in full_data:
                if not entry or not isinstance(entry, dict):
                    continue
                
                start_time_str = entry.get('startTime')
                end_time_str = entry.get('endTime')
                
                if not (start_time_str and end_time_str):
                    continue
                
                try:
                    start_dt = pd.to_datetime(start_time_str, utc=True)
                    end_dt = pd.to_datetime(end_time_str, utc=True)
                    
                    if start_dt < to_dt and end_dt >= from_dt:
                        relevant_data.append(entry)
                except:
                    continue
            
            date_filtered_count = len(relevant_data)
            self.log_diagnostic(f"📅 Entries in date range: {date_filtered_count:,}")
            
            if not relevant_data:
                self.log_diagnostic("❌ NO ENTRIES FOUND IN DATE RANGE!", "ERROR")
                self.log_diagnostic("💡 Try expanding your date range", "WARNING")
                return {'error': 'No entries found in specified date range'}
            
            del full_data  # Free memory
            
            # Process entries with focused diagnostics
            self.update_progress("Processing entries...", 50)
            activities, visits, position_fixes = self.process_entries_streamlined(relevant_data, settings)
            
            # Key metrics logging
            self.log_diagnostic("=== PROCESSING RESULTS ===")
            self.log_diagnostic(f"📊 Raw activities found: {self.debug_stats['activities_found']}")
            self.log_diagnostic(f"📊 Raw visits found: {self.debug_stats['visits_found']}")
            self.log_diagnostic(f"📊 Timeline paths found: {self.debug_stats['timeline_paths_found']}")
            
            # Filtering analysis with actionable insights
            self.log_diagnostic("=== FILTERING ANALYSIS ===")
            
            # Activities analysis
            if self.debug_stats['activities_found'] > 0:
                kept_activities = len(activities)
                filtered_activities = self.debug_stats['activities_found'] - kept_activities
                keep_rate = (kept_activities / self.debug_stats['activities_found']) * 100
                
                self.log_diagnostic(f"🚗 Activities: {kept_activities}/{self.debug_stats['activities_found']} kept ({keep_rate:.1f}%)")
                
                if filtered_activities > 0:
                    self.log_diagnostic(f"❌ {filtered_activities} activities filtered (distance < {settings['distance_threshold']}m)")
                    if keep_rate < 50:
                        self.log_diagnostic(f"💡 LOW ACTIVITY RETENTION: Try lowering distance_threshold to 50m", "WARNING")
                    elif keep_rate < 80:
                        self.log_diagnostic(f"💡 Consider lowering distance_threshold to {int(settings['distance_threshold']/2)}m", "WARNING")
            else:
                self.log_diagnostic("⚠️ No activities found in date range", "WARNING")
            
            # Visits analysis
            if self.debug_stats['visits_found'] > 0:
                kept_visits = len(visits)
                total_visit_filtered = self.debug_stats['visits_duration_filtered'] + self.debug_stats['visits_probability_filtered']
                keep_rate = (kept_visits / self.debug_stats['visits_found']) * 100
                
                self.log_diagnostic(f"🏠 Visits: {kept_visits}/{self.debug_stats['visits_found']} kept ({keep_rate:.1f}%)")
                
                if self.debug_stats['visits_duration_filtered'] > 0:
                    self.log_diagnostic(f"❌ {self.debug_stats['visits_duration_filtered']} visits filtered (duration < {settings['duration_threshold']}s)")
                    if self.debug_stats['visits_duration_filtered'] > kept_visits:
                        self.log_diagnostic(f"💡 HIGH DURATION FILTERING: Try lowering duration_threshold to 300s", "WARNING")
                
                if self.debug_stats['visits_probability_filtered'] > 0:
                    self.log_diagnostic(f"❌ {self.debug_stats['visits_probability_filtered']} visits filtered (probability < {settings['probability_threshold']})")
                    if self.debug_stats['visits_probability_filtered'] > kept_visits:
                        self.log_diagnostic(f"💡 HIGH PROBABILITY FILTERING: Try lowering probability_threshold to 0.05", "WARNING")
            else:
                self.log_diagnostic("⚠️ No visits found in date range", "WARNING")
            
            # Apply filters and build output
            self.update_progress("Building output...", 80)
            position_fixes = self.apply_filters_streamlined(position_fixes, settings)
            rebuilt_data = self.build_output_json_web(activities, visits, position_fixes)
            
            # Final summary
            self.log_diagnostic("=== FINAL SUMMARY ===")
            self.log_diagnostic(f"✅ Total output entries: {len(rebuilt_data)}")
            self.log_diagnostic(f"✅ Activities retained: {len(activities)}")
            self.log_diagnostic(f"✅ Visits retained: {len(visits)}")
            self.log_diagnostic(f"✅ Position fixes: {len(position_fixes)}")
            
            # Performance recommendations
            recommendations = self.generate_streamlined_recommendations(settings)
            if recommendations:
                self.log_diagnostic("=== RECOMMENDATIONS ===")
                for rec in recommendations:
                    self.log_diagnostic(f"💡 {rec}", "WARNING")
            
            self.update_progress("Complete!", 100)
            
            # Combine all stats
            all_stats = {**self.stats, **self.debug_stats}
            all_stats['date_filtered_count'] = date_filtered_count
            all_stats['final_output_count'] = len(rebuilt_data)
            
            clean_stats = {}
            for key, value in all_stats.items():
                clean_stats[key] = json_serializable(value)
            
            return {
                'success': True,
                'data': rebuilt_data,
                'stats': clean_stats,
                'modes': sorted(list(self.modes_seen)),
                'diagnostics': self.diagnostic_log,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.log_diagnostic(f"❌ PROCESSING FAILED: {str(e)}", "ERROR")
            return {'error': str(e)}
    
    def process_entries_streamlined(self, relevant_data: List[Dict], settings: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Streamlined processing focused on key metrics."""
        activities = []
        visits = []
        all_position_fixes = []
        
        from_dt = pd.to_datetime(settings['from_date'], utc=True) if settings.get('from_date') else None
        to_dt = pd.to_datetime(settings['to_date'], utc=True) + pd.Timedelta(days=1) if settings.get('to_date') else None
        
        for entry in relevant_data:
            try:
                if 'activity' in entry:
                    self.debug_stats['activities_found'] += 1
                    activity = self.process_activity_web_diagnostic(entry, settings)
                    if activity:
                        activities.append(activity)
                        all_position_fixes.extend(self.activity_to_position_fixes(activity, from_dt, to_dt))
                
                elif 'visit' in entry:
                    self.debug_stats['visits_found'] += 1
                    visit = self.process_visit_web_diagnostic(entry, settings)
                    if visit:
                        visits.append(visit)
                        all_position_fixes.extend(self.visit_to_position_fixes(visit, from_dt, to_dt))
                
                elif 'timelinePath' in entry:
                    self.debug_stats['timeline_paths_found'] += 1
                    timeline_fixes = self.process_timeline_path(entry, from_dt, to_dt)
                    all_position_fixes.extend(timeline_fixes)
            
            except Exception as e:
                self.debug_stats['processing_errors'] += 1
                continue
        
        self.stats['processed_entries'] = len(relevant_data)
        self.stats['activities_retained'] = len(activities)
        self.stats['visits_retained'] = len(visits)
        
        return activities, visits, all_position_fixes
    
    def apply_filters_streamlined(self, position_fixes: List[Dict], settings: Dict) -> List[Dict]:
        """Streamlined filtering."""
        if not position_fixes:
            return []
        
        result = []
        for fix in position_fixes:
            try:
                result.append({
                    'tracked_at': fix['tracked_at'].isoformat() if hasattr(fix['tracked_at'], 'isoformat') else str(fix['tracked_at']),
                    'latitude': float(fix['latitude']),
                    'longitude': float(fix['longitude']),
                    'user_id': str(fix['user_id']),
                    'mode': str(fix['mode'])
                })
            except:
                continue
        
        self.stats['position_fixes_retained'] = len(result)
        return result
    
    def generate_streamlined_recommendations(self, settings: Dict) -> List[str]:
        """Generate focused, actionable recommendations."""
        recommendations = []
        
        # Activity recommendations
        if self.debug_stats['activities_found'] > 0:
            kept_activities = self.stats.get('activities_retained', 0)
            activity_keep_rate = (kept_activities / self.debug_stats['activities_found']) * 100
            
            if activity_keep_rate < 30:
                recommendations.append(f"Very low activity retention ({activity_keep_rate:.0f}%) - try distance_threshold = 50m")
            elif activity_keep_rate < 60:
                recommendations.append(f"Low activity retention ({activity_keep_rate:.0f}%) - consider distance_threshold = {int(settings['distance_threshold']/2)}m")
        
        # Visit recommendations  
        if self.debug_stats['visits_found'] > 0:
            kept_visits = self.stats.get('visits_retained', 0)
            visit_keep_rate = (kept_visits / self.debug_stats['visits_found']) * 100
            
            if visit_keep_rate < 30:
                if self.debug_stats['visits_duration_filtered'] > self.debug_stats['visits_probability_filtered']:
                    recommendations.append(f"Very low visit retention ({visit_keep_rate:.0f}%) - try duration_threshold = 300s")
                else:
                    recommendations.append(f"Very low visit retention ({visit_keep_rate:.0f}%) - try probability_threshold = 0.05")
            elif visit_keep_rate < 60:
                if self.debug_stats['visits_duration_filtered'] > kept_visits:
                    recommendations.append(f"Many visits filtered by duration - consider duration_threshold = {int(settings['duration_threshold']/2)}s")
                if self.debug_stats['visits_probability_filtered'] > kept_visits:
                    recommendations.append(f"Many visits filtered by probability - consider probability_threshold = {settings['probability_threshold']/2:.2f}")
        
        # Data availability
        if self.debug_stats['activities_found'] == 0 and self.debug_stats['visits_found'] == 0:
            recommendations.append("No location data found in date range - try expanding the date range")
        
        return recommendations
    
    def generate_recommendations(self, settings: Dict) -> List[str]:
        """Generate specific recommendations based on diagnostic results."""
        recommendations = []
        
        if self.debug_stats['activities_found'] == 0:
            recommendations.append("No activities found - check if your date range contains activity data")
        elif self.stats.get('activities_retained', 0) == 0 and self.debug_stats['activities_found'] > 0:
            recommendations.append(f"All {self.debug_stats['activities_found']} activities were filtered out - try lowering distance_threshold from {settings['distance_threshold']}m to 50m")
        
        if self.debug_stats['visits_found'] == 0:
            recommendations.append("No visits found - check if your date range contains visit data")
        elif self.stats.get('visits_retained', 0) == 0 and self.debug_stats['visits_found'] > 0:
            if self.debug_stats['visits_duration_filtered'] > 0:
                recommendations.append(f"Visits filtered by duration - try lowering duration_threshold from {settings['duration_threshold']}s to 300s")
            if self.debug_stats['visits_probability_filtered'] > 0:
                recommendations.append(f"Visits filtered by probability - try lowering probability_threshold from {settings['probability_threshold']} to 0.05")
        
        if not recommendations:
            recommendations.append("Processing completed successfully with current settings!")
        
        return recommendations
    
    # Include simplified versions of processing methods for diagnostics
    def process_entries_web_diagnostic(self, relevant_data: List[Dict], settings: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Process entries with web diagnostic logging."""
        activities = []
        visits = []
        all_position_fixes = []
        
        from_dt = pd.to_datetime(settings['from_date'], utc=True) if settings.get('from_date') else None
        to_dt = pd.to_datetime(settings['to_date'], utc=True) + pd.Timedelta(days=1) if settings.get('to_date') else None
        
        for entry in relevant_data:
            try:
                if 'activity' in entry:
                    self.debug_stats['activities_found'] += 1
                    activity = self.process_activity_web_diagnostic(entry, settings)
                    if activity:
                        activities.append(activity)
                        all_position_fixes.extend(self.activity_to_position_fixes(activity, from_dt, to_dt))
                    else:
                        self.debug_stats['activities_distance_filtered'] += 1
                
                elif 'visit' in entry:
                    self.debug_stats['visits_found'] += 1
                    visit = self.process_visit_web_diagnostic(entry, settings)
                    if visit:
                        visits.append(visit)
                        all_position_fixes.extend(self.visit_to_position_fixes(visit, from_dt, to_dt))
                
                elif 'timelinePath' in entry:
                    self.debug_stats['timeline_paths_found'] += 1
                    timeline_fixes = self.process_timeline_path(entry, from_dt, to_dt)
                    all_position_fixes.extend(timeline_fixes)
            
            except Exception as e:
                self.debug_stats['processing_errors'] += 1
                continue
        
        self.stats['processed_entries'] = len(relevant_data)
        self.stats['activities_retained'] = len(activities)
        self.stats['visits_retained'] = len(visits)
        
        return activities, visits, all_position_fixes
    
    def process_activity_web_diagnostic(self, entry: Dict, settings: Dict) -> Optional[Dict]:
        """Process activity with web diagnostic logging."""
        try:
            activity = entry['activity']
            
            start_coords = self.parse_coordinates(activity.get('start', ''))
            end_coords = self.parse_coordinates(activity.get('end', ''))
            
            if not start_coords or not end_coords:
                return None
            
            distance = float(activity.get('distanceMeters', 0))
            distance_threshold = settings.get('distance_threshold', 200)
            
            if distance < distance_threshold:
                return None
            
            top_candidate = activity.get('topCandidate', {})
            activity_type = top_candidate.get('type', 'unknown').lower()
            self.modes_seen.add(activity_type)
            
            return {
                'startTime': self.preserve_timestamp_format(entry['startTime']),
                'endTime': self.preserve_timestamp_format(entry['endTime']),
                'start_lat': start_coords[0],
                'start_lon': start_coords[1],
                'end_lat': end_coords[0],
                'end_lon': end_coords[1],
                'distanceMeters': distance,
                'type': activity_type,
                'probability': float(top_candidate.get('probability', 0.0)),
                'activity_probability': float(activity.get('probability', 0.0))
            }
        except:
            return None
    
    def process_visit_web_diagnostic(self, entry: Dict, settings: Dict) -> Optional[Dict]:
        """Process visit with web diagnostic logging."""
        try:
            visit = entry['visit']
            top_candidate = visit.get('topCandidate', {})
            coords = self.parse_coordinates(top_candidate.get('placeLocation', ''))
            if not coords:
                return None
            
            start_dt = pd.to_datetime(entry['startTime'], utc=True)
            end_dt = pd.to_datetime(entry['endTime'], utc=True)
            duration = (end_dt - start_dt).total_seconds()
            duration_threshold = settings.get('duration_threshold', 600)
            
            if duration < duration_threshold:
                self.debug_stats['visits_duration_filtered'] += 1
                return None
            
            probability = float(visit.get('probability', 0.0))
            probability_threshold = settings.get('probability_threshold', 0.1)
            
            if probability < probability_threshold:
                self.debug_stats['visits_probability_filtered'] += 1
                return None
            
            return {
                'startTime': self.preserve_timestamp_format(entry['startTime']),
                'endTime': self.preserve_timestamp_format(entry['endTime']),
                'latitude': coords[0],
                'longitude': coords[1],
                'placeID': top_candidate.get('placeID', ''),
                'semanticType': top_candidate.get('semanticType', 'UNKNOWN'),
                'probability': probability,
                'topCandidate_probability': float(top_candidate.get('probability', 0.0))
            }
        except:
            return None
    
    # Include other necessary methods (simplified for space)
    def process_timeline_path(self, entry: Dict, from_dt=None, to_dt=None) -> List[Dict]:
        """Process timeline path entry."""
        position_fixes = []
        try:
            start_dt = pd.to_datetime(entry['startTime'], utc=True)
            timeline_points = entry.get('timelinePath', [])
            
            for point in timeline_points:
                coords = self.parse_coordinates(point.get('point', ''))
                if not coords:
                    continue
                
                try:
                    offset_min = float(point.get('durationMinutesOffsetFromStartTime', 0))
                    point_dt = start_dt + pd.Timedelta(minutes=offset_min)
                    mode = point.get('mode', point.get('type', 'unknown')).lower()
                    
                    self.modes_seen.add(mode)
                    
                    if from_dt and point_dt < from_dt:
                        continue
                    if to_dt and point_dt >= to_dt:
                        continue
                    
                    position_fixes.append({
                        'tracked_at': point_dt,
                        'latitude': coords[0],
                        'longitude': coords[1],
                        'user_id': 'user_1',
                        'mode': mode
                    })
                except:
                    continue
        except:
            pass
        
        return position_fixes
    
    def activity_to_position_fixes(self, activity: Dict, from_dt=None, to_dt=None) -> List[Dict]:
        """Convert activity to position fixes."""
        fixes = []
        start_dt = pd.to_datetime(activity['startTime'], utc=True)
        end_dt = pd.to_datetime(activity['endTime'], utc=True)
        
        if (not from_dt or start_dt >= from_dt) and (not to_dt or start_dt < to_dt):
            fixes.append({
                'tracked_at': start_dt,
                'latitude': activity['start_lat'],
                'longitude': activity['start_lon'],
                'user_id': 'user_1',
                'mode': activity['type']
            })
        
        if (not from_dt or end_dt >= from_dt) and (not to_dt or end_dt < to_dt):
            fixes.append({
                'tracked_at': end_dt,
                'latitude': activity['end_lat'],
                'longitude': activity['end_lon'],
                'user_id': 'user_1',
                'mode': activity['type']
            })
        
        return fixes
    
    def visit_to_position_fixes(self, visit: Dict, from_dt=None, to_dt=None) -> List[Dict]:
        """Convert visit to position fixes."""
        fixes = []
        start_dt = pd.to_datetime(visit['startTime'], utc=True)
        
        if (not from_dt or start_dt >= from_dt) and (not to_dt or start_dt < to_dt):
            fixes.append({
                'tracked_at': start_dt,
                'latitude': visit['latitude'],
                'longitude': visit['longitude'],
                'user_id': 'user_1',
                'mode': 'stationary'
            })
        
        return fixes
    
    def apply_filters_web_diagnostic(self, position_fixes: List[Dict], settings: Dict) -> List[Dict]:
        """Apply filters with web diagnostic logging."""
        if not position_fixes:
            self.log_diagnostic("No position fixes to filter")
            return []
        
        self.log_diagnostic(f"Starting with {len(position_fixes)} position fixes")
        
        # Simple filtering for diagnostic purposes
        result = []
        for fix in position_fixes:
            try:
                result.append({
                    'tracked_at': fix['tracked_at'].isoformat() if hasattr(fix['tracked_at'], 'isoformat') else str(fix['tracked_at']),
                    'latitude': float(fix['latitude']),
                    'longitude': float(fix['longitude']),
                    'user_id': str(fix['user_id']),
                    'mode': str(fix['mode'])
                })
            except:
                continue
        
        self.stats['position_fixes_retained'] = len(result)
        self.log_diagnostic(f"Final position fixes: {len(result)}")
        
        return result
    
    def build_output_json_web(self, activities: List[Dict], visits: List[Dict], position_fixes: List[Dict]) -> List[Dict]:
        """Build output JSON with web diagnostic logging."""
        rebuilt_data = []
        
        # Add activities
        for activity in activities:
            rebuilt_data.append({
                'startTime': activity['startTime'],
                'endTime': activity['endTime'],
                'activity': {
                    'start': f"geo:{activity['start_lat']},{activity['start_lon']}",
                    'end': f"geo:{activity['end_lat']},{activity['end_lon']}",
                    'distanceMeters': str(float(activity['distanceMeters'])),
                    'topCandidate': {
                        'type': str(activity['type']),
                        'probability': str(float(activity['probability']))
                    },
                    'probability': str(float(activity['activity_probability']))
                }
            })
        
        # Add visits
        for visit in visits:
            rebuilt_data.append({
                'startTime': visit['startTime'],
                'endTime': visit['endTime'],
                'visit': {
                    'topCandidate': {
                        'placeLocation': f"geo:{visit['latitude']},{visit['longitude']}",
                        'placeID': str(visit['placeID']),
                        'semanticType': str(visit['semanticType']),
                        'probability': str(float(visit['topCandidate_probability']))
                    },
                    'probability': str(float(visit['probability']))
                }
            })
        
        # Add timeline paths
        if position_fixes:
            daily_fixes = {}
            for fix in position_fixes:
                try:
                    if isinstance(fix['tracked_at'], str):
                        dt = pd.to_datetime(fix['tracked_at'])
                    else:
                        dt = fix['tracked_at']
                    
                    date_key = dt.date()
                    if date_key not in daily_fixes:
                        daily_fixes[date_key] = []
                    daily_fixes[date_key].append({
                        'tracked_at': dt,
                        'latitude': float(fix['latitude']),
                        'longitude': float(fix['longitude']),
                        'mode': str(fix['mode'])
                    })
                except:
                    continue
            
            for date, fixes in daily_fixes.items():
                if not fixes:
                    continue
                    
                fixes.sort(key=lambda x: x['tracked_at'])
                start_time = fixes[0]['tracked_at'].isoformat()
                end_time = fixes[-1]['tracked_at'].isoformat()
                start_dt = fixes[0]['tracked_at']
                
                timeline_points = []
                for fix in fixes:
                    offset_minutes = (fix['tracked_at'] - start_dt).total_seconds() / 60
                    timeline_points.append({
                        'point': f"geo:{fix['latitude']},{fix['longitude']}",
                        'durationMinutesOffsetFromStartTime': str(float(offset_minutes)),
                        'mode': fix['mode']
                    })
                
                rebuilt_data.append({
                    'startTime': start_time,
                    'endTime': end_time,
                    'timelinePath': timeline_points
                })
        
        return rebuilt_data


# ===== WEB APP ROUTES WITH WEB DIAGNOSTICS =====

def update_progress(task_id: str, message: str, percentage: float = None):
    """Store progress update for web interface."""
    if task_id not in progress_store:
        progress_store[task_id] = {}
    
    progress_store[task_id].update({
        'message': message,
        'percentage': percentage or 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/')
def index():
    """Main page with enhanced diagnostic display."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start WEB DIAGNOSTIC processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.json'):
        return jsonify({'error': 'Please upload a JSON file'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    task_id = str(uuid.uuid4())
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
    file.save(upload_path)
    
    # Get settings from form
    settings = {
        'from_date': request.form.get('from_date'),
        'to_date': request.form.get('to_date'),
        'distance_threshold': float(request.form.get('distance_threshold', 200)),
        'probability_threshold': float(request.form.get('probability_threshold', 0.1)),
        'duration_threshold': int(request.form.get('duration_threshold', 600)),
        'speed_threshold_kmh': float(request.form.get('speed_threshold_kmh', 120)),
        'accuracy_threshold': float(request.form.get('accuracy_threshold', 0)),
        'outlier_std_multiplier': float(request.form.get('outlier_std_multiplier', 3.0))
    }
    
    # Initialize progress tracking
    progress_store[task_id] = {
        'status': 'PENDING', 
        'message': 'Starting WEB DIAGNOSTIC processing...', 
        'percentage': 0,
        'diagnostics': []
    }
    
    # Start WEB DIAGNOSTIC background processing
    def process_in_background():
        processor = WebDiagnosticProcessor(task_id, max_workers=4)
        processor.set_progress_callback(lambda msg, pct: update_progress(task_id, msg, pct))
        
        try:
            # Use the WEB DIAGNOSTIC processing method
            result = processor.process_file_web_diagnostic(upload_path, settings)
            
            if result.get('success'):
                # Save processed data
                output_file = os.path.join(app.config['PROCESSED_FOLDER'], f"{task_id}_output.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result['data'], f, indent=2)
                
                progress_store[task_id]['result'] = result
                progress_store[task_id]['status'] = 'SUCCESS'
                update_progress(task_id, 'WEB DIAGNOSTIC processing completed!', 100)
            else:
                progress_store[task_id]['status'] = 'FAILURE'
                progress_store[task_id]['error'] = result.get('error', 'Unknown error')
                update_progress(task_id, f"Error: {result.get('error', 'Unknown error')}", 0)
                
        except Exception as e:
            logger.error(f"WEB DIAGNOSTIC processing failed: {e}")
            progress_store[task_id]['status'] = 'FAILURE'
            progress_store[task_id]['error'] = str(e)
            update_progress(task_id, f"Error: {str(e)}", 0)
        
        # Clean up uploaded file after processing
        try:
            os.remove(upload_path)
        except:
            pass
    
    # Start processing in background thread
    thread = threading.Thread(target=process_in_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'task_id': task_id,
        'message': 'WEB DIAGNOSTIC processing started - watch the diagnostics panel!',
        'status': 'PENDING'
    })

@app.route('/progress/<task_id>')
def get_progress(task_id):
    """Get processing progress with web diagnostics."""
    if task_id not in progress_store:
        return jsonify({'state': 'PENDING', 'message': 'Task not found'})
    
    progress = progress_store[task_id]
    
    # Ensure all values are JSON serializable
    result = progress.get('result', {})
    if result:
        for key, value in result.items():
            if hasattr(value, 'item') or hasattr(value, 'tolist'):
                result[key] = json_serializable(value)
    
    return jsonify({
        'state': progress.get('status', 'PENDING'),
        'message': progress.get('message', ''),
        'percentage': progress.get('percentage', 0),
        'result': result,
        'diagnostics': progress.get('diagnostics', [])
    })

@app.route('/download/<task_id>')
def download_result(task_id):
    """Download the processed file with date-based filename."""
    output_file = os.path.join(app.config['PROCESSED_FOLDER'], f"{task_id}_output.json")
    
    if not os.path.exists(output_file):
        return jsonify({'error': 'File not found'}), 404
    
    # Try to get the date range from the task's progress store for filename
    download_name = f"processed_location_data_{task_id[:8]}.json"
    if task_id in progress_store:
        result = progress_store[task_id].get('result', {})
        # The frontend will handle the filename, so we just need to send the file
        # The frontend sets the download attribute with the date-based name
    
    return send_file(
        output_file,
        as_attachment=True,
        download_name=download_name,
        mimetype='application/json'
    )

@app.route('/cleanup/<task_id>', methods=['POST'])
def cleanup_files(task_id):
    """Clean up temporary files after download."""
    try:
        upload_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(task_id)]
        for f in upload_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
            if os.path.exists(file_path):
                os.remove(file_path)
        
        output_file = os.path.join(app.config['PROCESSED_FOLDER'], f"{task_id}_output.json")
        if os.path.exists(output_file):
            os.remove(output_file)
        
        if task_id in progress_store:
            del progress_store[task_id]
        
        return jsonify({'message': 'Files cleaned up successfully'})
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check with web diagnostic info."""
    return jsonify({
        'status': 'healthy',
        'version': '5.0.0-web-diagnostic',
        'active_tasks': len(progress_store),
        'cpu_count': multiprocessing.cpu_count(),
        'features': [
            'Real-time web diagnostics',
            'Filtering analysis',
            'Smart recommendations',
            'Detailed logging'
        ]
    })

if __name__ == '__main__':
    print("[SEARCH] WEB DIAGNOSTIC Location Data Processor")
    print("=" * 60)
    print("[WEB] WEB DIAGNOSTIC Features:")
    print("   • Real-time diagnostics visible in browser")
    print("   • Detailed filtering analysis")
    print("   • Smart recommendations for settings")
    print("   • No need to watch console - everything in web!")
    print("=" * 60)
    print("[FOLDER] Upload folder:", app.config['UPLOAD_FOLDER'])
    print("[FOLDER] Processed folder:", app.config['PROCESSED_FOLDER'])
    print("[LINK] Open in browser: http://localhost:5000")
    print("=" * 60)
    print("[SEARCH] ALL DIAGNOSTICS NOW VISIBLE IN WEB BROWSER!")
    print("   No need to watch console - diagnostics panel shows everything")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)