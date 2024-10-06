from future.builtins import *  # NOQA
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn.client import Client
import os

SECONDS_PER_DAY=3600.*24

user=os.environ['IRIS_USER']
auth_password=os.environ['IRIS_PASSWORD']

def download_seismogram_per_month(start_year, end_year):
    """Download and view seismograms month by month."""
    network = 'XA'
    station = '*'
    channel = 'MHZ'
    location = '*'
    
    client = Client("IRIS", user=user, password=auth_password)
    
    for year in range(start_year, end_year):
        for month in range(1, 13):
            # Define start time for the month
            starttime = UTCDateTime(f'{year}-{month:02d}-01T00:00:00.0')
            
            # Define end time as start of the next month
            if month == 12:
                # If December, next month is January of the next year
                endtime = UTCDateTime(f'{year+1}-01-01T00:00:00.0')
            else:
                # Otherwise, it's the next month in the same year
                endtime = UTCDateTime(f'{year}-{month+1:02d}-01T00:00:00.0')
            
            try:
                # Request and download the data for the month
                print(f"Downloading data for {year}-{month:02d}")
                stream = client.get_waveforms(network=network, station=station, channel=channel, location=location, starttime=starttime, endtime=endtime)
                print(f"Downloaded data for {year}-{month:02d}")
                # Process or save the stream as needed

                for trace in stream:
                    # Set file name based on trace details
                    filename = f"{trace.stats.network}_{trace.stats.station}_{trace.stats.channel}_{trace.stats.starttime}.mseed"
                    # Ensure filename is valid
                    filename = filename.replace(':', '-')
                    # Write the trace to a MiniSEED file

                    trace.write('data2/' + filename, format='MSEED')
                    print(f"Saved {filename}")

            except Exception as e:
                print(f"Error downloading data for {year}-{month:02d}: {e}")

# Example usage
download_seismogram_per_month(1975, 1980)
