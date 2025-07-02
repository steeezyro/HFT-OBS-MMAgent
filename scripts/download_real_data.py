#!/usr/bin/env python3
"""
Download real NASDAQ ITCH 5.0 data for performance testing.
This script downloads actual market data from NASDAQ's public FTP.
"""

import os
import sys
import urllib.request
import gzip
import shutil
from datetime import datetime, timedelta
import argparse

# NASDAQ FTP base URL for ITCH data
NASDAQ_FTP_BASE = "ftp://emi.nasdaq.com/ITCH/"

def get_available_dates():
    """Get recent trading dates for which ITCH data might be available."""
    dates = []
    today = datetime.now()
    
    # Look back 30 days for available data
    for i in range(30):
        date = today - timedelta(days=i)
        # Skip weekends
        if date.weekday() < 5:  # Monday = 0, Friday = 4
            dates.append(date.strftime("%m%d%Y"))
    
    return dates

def download_itch_file(date_str, output_dir="data"):
    """
    Download ITCH file for a specific date.
    
    Args:
        date_str: Date in MMDDYYYY format (e.g., "01302019")
        output_dir: Directory to save the file
    
    Returns:
        Path to downloaded file or None if failed
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # NASDAQ ITCH filename format
    filename = f"{date_str}.NASDAQ_ITCH50.gz"
    url = NASDAQ_FTP_BASE + filename
    local_gz_path = os.path.join(output_dir, filename)
    local_path = os.path.join(output_dir, filename.replace('.gz', ''))
    
    print(f"Downloading {url}...")
    
    try:
        # Download compressed file
        urllib.request.urlretrieve(url, local_gz_path)
        print(f"Downloaded {local_gz_path}")
        
        # Decompress
        print(f"Decompressing to {local_path}...")
        with gzip.open(local_gz_path, 'rb') as f_in:
            with open(local_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove compressed file to save space
        os.remove(local_gz_path)
        
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"✅ Successfully downloaded and decompressed: {local_path}")
        print(f"   File size: {file_size_mb:.1f} MB")
        
        return local_path
        
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        # Clean up partial downloads
        for path in [local_gz_path, local_path]:
            if os.path.exists(path):
                os.remove(path)
        return None

def get_recent_data():
    """Try to download the most recent available ITCH data."""
    dates = get_available_dates()
    
    print("Attempting to download recent NASDAQ ITCH data...")
    print("Note: NASDAQ typically makes data available with a delay")
    
    for date_str in dates:
        print(f"\nTrying date: {date_str}")
        result = download_itch_file(date_str)
        if result:
            return result
        print(f"No data available for {date_str}")
    
    print("\n❌ No recent ITCH data available from NASDAQ FTP")
    print("This is normal - NASDAQ has delays in publishing free historical data")
    return None

def create_sample_data():
    """Create sample ITCH-like data for testing."""
    print("\n📝 Creating sample ITCH data for testing...")
    
    import struct
    import random
    import time
    
    os.makedirs("data", exist_ok=True)
    
    def create_sample_itch():
        messages = []
        timestamp = int(time.time() * 1e9)  # Nanoseconds
        
        # Create 1000 sample messages
        for i in range(1000):
            # Add Order message (type 'A')
            msg_type = b'A'
            ts = struct.pack('>Q', timestamp + i * 1000000)  # 1ms apart
            order_id = struct.pack('>Q', i + 1)
            side = b'B' if random.random() > 0.5 else b'S'
            shares = struct.pack('>L', random.randint(100, 1000))
            stock = b'AAPL    '  # 8 bytes, padded
            price = struct.pack('>L', random.randint(1500000, 1600000))  # $150-160 in 1/10000
            
            # Message length (2 bytes) + message
            msg = msg_type + ts + order_id + side + shares + stock + price
            msg_length = struct.pack('>H', len(msg))
            messages.append(msg_length + msg)
        
        return b''.join(messages)

    # Write sample data
    with open('data/sample.itch', 'wb') as f:
        f.write(create_sample_itch())

    print("✅ Created data/sample.itch with 1000 synthetic messages")
    return "data/sample.itch"

def main():
    parser = argparse.ArgumentParser(description='Download real NASDAQ ITCH data')
    parser.add_argument('--date', help='Specific date in MMDDYYYY format')
    parser.add_argument('--sample', action='store_true', help='Create sample data instead')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    
    args = parser.parse_args()
    
    if args.sample:
        result = create_sample_data()
    elif args.date:
        result = download_itch_file(args.date, args.output_dir)
    else:
        result = get_recent_data()
    
    if result:
        print(f"\n🎯 Data ready: {result}")
        print(f"\nNext steps:")
        print(f"1. Build the C++ system: cmake -B build && cmake --build build")
        print(f"2. Run performance benchmark: ./build/bench_orderbook")
        print(f"3. Test market replay: ./build/replay -f {result} -s 1.0")
        return 0
    else:
        print(f"\n⚠️  No data available. Use --sample to create test data.")
        return 1

if __name__ == "__main__":
    sys.exit(main())