import os
import csv
import ftplib
import pandas as pd
import hashlib
import logging
from datetime import datetime
from google.cloud import storage
from flask import jsonify
import re
from urllib.parse import quote

# ----------------------------
# LOGGING CONFIGURATION
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# CONFIGURATION & CREDENTIALS
# ----------------------------

# Environment variables
bucket_name = os.environ.get("BUCKET_NAME")
bucket_folder = os.environ.get("BUCKET_FOLDER")
local_output_directory = os.environ.get("LOCAL_OUTPUT_DIRECTORY", "/tmp/output")
ftp_download_dir = os.environ.get("FTP_DOWNLOAD_DIR", "/tmp/ftp")

# Create directories if they don't exist
os.makedirs(local_output_directory, exist_ok=True)
os.makedirs(ftp_download_dir, exist_ok=True)

# FTP Server Details
FTP_SERVER = "ftp.nivoda.net"
FTP_PORT = 21
FTP_USERNAME = "leeladiamondscorporate@gmail.com"
FTP_PASSWORD = "1yH£lG4n0Mq"

# Enhanced product configuration
ftp_files = {
    "natural": {
        "remote_filename": "Leela Diamond_natural.csv",
        "local_path": os.path.join(ftp_download_dir, "Natural.csv"),
        "priority": 1
    },
    "lab_grown": {
        "remote_filename": "Leela Diamond_labgrown.csv",
        "local_path": os.path.join(ftp_download_dir, "Labgrown.csv"),
        "priority": 2
    },
    "gemstone": {
        "remote_filename": "Leela Diamond_gemstones.csv",
        "local_path": os.path.join(ftp_download_dir, "gemstones.csv"),
        "priority": 3
    }
}

# Enhanced shape mapping with SEO-friendly attributes
SHAPE_CONFIG = {
    "ASSCHER": {
        "url": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/ASSCHER.jpg",
        "seo_name": "Asscher Cut",
        "category_path": "Jewelry & Watches > Fine Jewelry > Diamonds > Asscher",
        "keywords": ["asscher", "square", "emerald", "step cut"]
    },
    "CUSHION": {
        "url": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/CUSHION.jpg",
        "seo_name": "Cushion Cut",
        "category_path": "Jewelry & Watches > Fine Jewelry > Diamonds > Cushion",
        "keywords": ["cushion", "pillow", "romantic", "vintage"]
    },
    "ROUND": {
        "url": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/ROUND.png",
        "seo_name": "Round Brilliant",
        "category_path": "Jewelry & Watches > Fine Jewelry > Diamonds > Round",
        "keywords": ["round", "brilliant", "classic", "sparkle"]
    },
    "OVAL": {
        "url": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/OVAL.webp",
        "seo_name": "Oval Cut",
        "category_path": "Jewelry & Watches > Fine Jewelry > Diamonds > Oval",
        "keywords": ["oval", "elongated", "elegant", "modern"]
    },
    "PRINCESS": {
        "url": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/PRINCESS.jpg",
        "seo_name": "Princess Cut",
        "category_path": "Jewelry & Watches > Fine Jewelry > Diamonds > Princess",
        "keywords": ["princess", "square", "brilliant", "contemporary"]
    },
    "EMERALD": {
        "url": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/EMERALD.jpg",
        "seo_name": "Emerald Cut",
        "category_path": "Jewelry & Watches > Fine Jewelry > Diamonds > Emerald",
        "keywords": ["emerald", "rectangular", "step cut", "sophisticated"]
    },
    "PEAR": {
        "url": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/PEAR.jpg",
        "seo_name": "Pear Cut",
        "category_path": "Jewelry & Watches > Fine Jewelry > Diamonds > Pear",
        "keywords": ["pear", "teardrop", "unique", "elegant"]
    },
    "MARQUISE": {
        "url": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/MARQUISE.jpg",
        "seo_name": "Marquise Cut",
        "category_path": "Jewelry & Watches > Fine Jewelry > Diamonds > Marquise",
        "keywords": ["marquise", "boat", "elongated", "regal"]
    },
    "HEART": {
        "url": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/HEART.png",
        "seo_name": "Heart Cut",
        "category_path": "Jewelry & Watches > Fine Jewelry > Diamonds > Heart",
        "keywords": ["heart", "romantic", "love", "special"]
    },
    "RADIANT": {
        "url": "https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/RADIANT.jpg",
        "seo_name": "Radiant Cut",
        "category_path": "Jewelry & Watches > Fine Jewelry > Diamonds > Radiant",
        "keywords": ["radiant", "rectangular", "brilliant", "fire"]
    }
    # Add more shapes as needed
}

# ----------------------------
# UTILITY FUNCTIONS
# ----------------------------

def generate_unique_id(row, product_type):
    """Generate a unique, consistent ID to prevent duplicates."""
    # Create a hash from key identifying fields
    id_string = f"{row.get('ReportNo', '')}-{product_type}-{row.get('carats', '')}-{row.get('shape', '')}"
    return hashlib.md5(id_string.encode()).hexdigest()[:12].upper()

def sanitize_text(text):
    """Clean and sanitize text for SEO."""
    if pd.isna(text) or text == '':
        return ''
    # Remove extra spaces, special characters that might cause issues
    text = re.sub(r'\s+', ' ', str(text).strip())
    text = re.sub(r'[^\w\s\-\.\,\(\)%]', '', text)
    return text

def format_price(price_value, apply_markup_func, convert_to_cad_func):
    """Format price with proper currency and markup."""
    try:
        price_numeric = pd.to_numeric(price_value, errors='coerce')
        if pd.isna(price_numeric) or price_numeric <= 0:
            return "0.00 CAD"
        
        marked_up = apply_markup_func(price_numeric)
        cad_price = convert_to_cad_func(marked_up)
        return f"{cad_price:.2f} CAD"
    except Exception as e:
        logger.error(f"Error formatting price {price_value}: {e}")
        return "0.00 CAD"

# ----------------------------
# FTP DOWNLOAD FUNCTION
# ----------------------------

def explore_ftp_structure():
    """Explore FTP server structure to find files."""
    try:
        with ftplib.FTP() as ftp:
            ftp.connect(FTP_SERVER, FTP_PORT, timeout=30)
            ftp.login(FTP_USERNAME, FTP_PASSWORD)
            
            logger.info(f"Connected to FTP server: {ftp.getwelcome()}")
            logger.info(f"Current directory: {ftp.pwd()}")
            
            # List all files in current directory
            file_list = []
            ftp.retrlines('LIST', file_list.append)
            
            logger.info("FTP Directory contents:")
            for item in file_list:
                logger.info(f"  {item}")
            
            # Get simple file list using LIST instead of NLST to avoid the directory error
            files = []
            for line in file_list:
                # Parse the LIST output to extract filenames
                parts = line.split()
                if len(parts) >= 9:  # Valid file listing
                    filename = parts[-1]  # Last part is the filename
                    files.append(filename)
            
            return files
                
    except Exception as e:
        logger.error(f"Error exploring FTP structure: {e}")
        return []

def find_diamond_files():
    """Find diamond files on FTP server with flexible naming."""
    try:
        with ftplib.FTP() as ftp:
            ftp.connect(FTP_SERVER, FTP_PORT, timeout=30)
            ftp.login(FTP_USERNAME, FTP_PASSWORD)
            
            # Get all files
            all_files = ftp.nlst()
            logger.info(f"Total files on server: {len(all_files)}")
            
            # Look for files containing diamond-related keywords
            found_files = {}
            
            for file in all_files:
                file_lower = file.lower()
                
                # Check for natural diamonds
                if any(keyword in file_lower for keyword in ['natural', 'mined']) and 'diamond' in file_lower:
                    found_files['natural'] = file
                    logger.info(f"Found natural diamond file: {file}")
                
                # Check for lab grown diamonds
                elif any(keyword in file_lower for keyword in ['lab', 'labgrown', 'lab-grown', 'synthetic']) and 'diamond' in file_lower:
                    found_files['lab_grown'] = file
                    logger.info(f"Found lab grown diamond file: {file}")
                
                # Check for gemstones
                elif any(keyword in file_lower for keyword in ['gemstone', 'gem', 'colored']) and not 'diamond' in file_lower:
                    found_files['gemstone'] = file
                    logger.info(f"Found gemstone file: {file}")
            
            return found_files
            
    except Exception as e:
        logger.error(f"Error finding diamond files: {e}")
        return {}

def download_file_from_ftp(remote_filename, local_path):
    """Download a file from the FTP server with enhanced error handling."""
    try:
        logger.info(f"Attempting to download: {remote_filename}")
        
        with ftplib.FTP() as ftp:
            # Set passive mode (helps with firewall issues)
            ftp.set_pasv(True)
            
            ftp.connect(FTP_SERVER, FTP_PORT, timeout=30)
            ftp.login(FTP_USERNAME, FTP_PASSWORD)
            
            # Verify file exists and get size
            try:
                file_size = ftp.size(remote_filename)
                logger.info(f"File {remote_filename} found, size: {file_size} bytes")
            except ftplib.error_perm as e:
                if "550" in str(e):
                    logger.error(f"File not found: {remote_filename}")
                    
                    # Try to find similar files
                    logger.info("Searching for similar files...")
                    files = ftp.nlst()
                    similar_files = [f for f in files if any(word in f.lower() for word in remote_filename.lower().split('_'))]
                    
                    if similar_files:
                        logger.info(f"Similar files found: {similar_files}")
                    else:
                        logger.info("No similar files found")
                    
                    return False
                else:
                    logger.warning(f"Could not get file size: {e}")
            
            # Download the file
            with open(local_path, 'wb') as f:
                ftp.retrbinary(f"RETR {remote_filename}", f.write)
            
            # Verify download success
            if os.path.exists(local_path):
                downloaded_size = os.path.getsize(local_path)
                logger.info(f"Successfully downloaded {remote_filename}: {downloaded_size} bytes")
                return True
            else:
                logger.error(f"Download failed: {remote_filename}")
                return False
                
    except ftplib.error_perm as e:
        logger.error(f"FTP Permission error downloading {remote_filename}: {e}")
        return False
    except ftplib.error_temp as e:
        logger.error(f"FTP Temporary error downloading {remote_filename}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {remote_filename}: {e}")
        return False

def download_all_files():
    """Download all defined files from the FTP server with smart file discovery."""
    logger.info("Starting file download process...")
    
    # First, explore the FTP structure
    all_files = explore_ftp_structure()
    if not all_files:
        logger.error("Could not retrieve FTP file list")
        return 0
    
    logger.info(f"Found {len(all_files)} files on FTP server:")
    for file in all_files:
        logger.info(f"  {file}")
    
    # Download all available diamond files
    success_count = 0
    for product_type, file_info in ftp_files.items():
        remote_filename = file_info["remote_filename"]
        local_path = file_info["local_path"]
        
        if remote_filename in all_files:
            logger.info(f"Downloading {product_type}: {remote_filename}")
            if download_file_from_ftp(remote_filename, local_path):
                success_count += 1
                logger.info(f"✅ Downloaded {product_type}: {remote_filename}")
            else:
                logger.error(f"❌ Failed to download {product_type}: {remote_filename}")
        else:
            logger.warning(f"File not found on server: {remote_filename}")
    
    logger.info(f"Downloaded {success_count}/{len(ftp_files)} files successfully")
    return success_count

# ----------------------------
# PRICING FUNCTIONS
# ----------------------------

def convert_to_cad(price_usd):
    """Convert price from USD to CAD using current exchange rate."""
    cad_rate = 1.46  # Consider making this dynamic from an API
    try:
        return round(float(price_usd) * cad_rate, 2)
    except (ValueError, TypeError) as e:
        logger.error(f"Error in currency conversion for {price_usd}: {e}")
        return 0.0

def apply_markup(price):
    """Apply tiered markup strategy."""
    try:
        price = float(price)
        base = price * 1.05 * 1.13  # Base markup + tax
        
        # Tiered additional markup
        additional = (
            210 if price <= 500 else
            375 if price <= 1000 else
            500 if price <= 1500 else
            700 if price <= 2000 else
            900 if price <= 2500 else
            1100 if price <= 3000 else
            1200 if price <= 5000 else
            1500 if price <= 100000 else
            2000  # For very high-end products
        )
        return round(base + additional, 2)
    except (ValueError, TypeError) as e:
        logger.error(f"Error applying markup to {price}: {e}")
        return price

# ----------------------------
# SEO OPTIMIZATION FUNCTIONS
# ----------------------------

def generate_seo_title(row, product_type):
    """Generate SEO-optimized titles."""
    try:
        shape = sanitize_text(row.get('shape', '')).title()
        carats = sanitize_text(row.get('carats', ''))
        color = sanitize_text(row.get('col', ''))
        clarity = sanitize_text(row.get('clar', ''))
        lab = sanitize_text(row.get('lab', ''))
        
        shape_config = SHAPE_CONFIG.get(shape.upper(), {})
        shape_name = shape_config.get('seo_name', shape)
        
        if product_type == "natural":
            title = f"{shape_name} Diamond {carats}ct {color} {clarity} {lab} Certified - Natural"
        elif product_type == "lab_grown":
            title = f"{shape_name} Lab Diamond {carats}ct {color} {clarity} {lab} Certified - Sustainable"
        else:  # gemstone
            gem_type = sanitize_text(row.get('gemType', ''))
            gem_color = sanitize_text(row.get('Color', ''))
            title = f"{shape_name} {gem_type} {carats}ct {gem_color} {clarity} - Premium Gemstone"
        
        # Ensure title is within Google's recommended length
        return title[:150]
    except Exception as e:
        logger.error(f"Error generating title for row {row}: {e}")
        return f"Premium Diamond - {product_type}"

def generate_seo_description(row, product_type):
    """Generate detailed, SEO-optimized descriptions."""
    try:
        shape = sanitize_text(row.get('shape', '')).title()
        carats = sanitize_text(row.get('carats', ''))
        color = sanitize_text(row.get('col', ''))
        clarity = sanitize_text(row.get('clar', ''))
        cut = sanitize_text(row.get('cut', ''))
        polish = sanitize_text(row.get('pol', ''))
        symmetry = sanitize_text(row.get('symm', ''))
        
        shape_config = SHAPE_CONFIG.get(shape.upper(), {})
        keywords = ', '.join(shape_config.get('keywords', [shape.lower()]))
        
        if product_type == "natural":
            desc = f"Exquisite {shape.lower()} natural diamond featuring {carats} carats of brilliance. Premium {color} color grade with {clarity} clarity ensures exceptional sparkle. Professional {cut} cut with {polish} polish and {symmetry} symmetry. Certified authenticity with detailed grading report. Perfect for engagement rings, fine jewelry. Keywords: {keywords}, natural diamond, certified, premium quality."
        elif product_type == "lab_grown":
            desc = f"Sustainable {shape.lower()} lab-grown diamond with identical properties to natural stones. {carats} carats of eco-conscious luxury featuring {color} color and {clarity} clarity. Superior {cut} cut maximizes brilliance. Environmentally responsible choice without compromising quality. Keywords: {keywords}, lab grown, sustainable, eco-friendly, certified."
        else:  # gemstone
            gem_type = sanitize_text(row.get('gemType', ''))
            desc = f"Stunning {shape.lower()} {gem_type.lower()} gemstone showcasing {carats} carats of natural beauty. Rich {color} coloration with {clarity} clarity. Expert cutting enhances natural fire and brilliance. Perfect for custom jewelry, collectors. Authentic certification included."
        
        return desc[:5000]  # Google's description limit
    except Exception as e:
        logger.error(f"Error generating description: {e}")
        return f"Premium quality {product_type} with certified authenticity and exceptional brilliance."

# ----------------------------
# DATA PROCESSING FUNCTIONS
# ----------------------------

def process_files_to_cad(files_to_load, output_file):
    """Process input CSV files with advanced deduplication and SEO optimization."""
    try:
        all_data = []
        seen_ids = set()
        duplicate_count = 0
        
        # Sort by priority to handle duplicates consistently
        sorted_files = sorted(files_to_load.items(), 
                            key=lambda x: ftp_files[x[0]].get('priority', 999))
        
        for product_type, file_info in sorted_files:
            input_file = file_info["file_path"]
            
            if not os.path.exists(input_file):
                logger.warning(f"File not found: {input_file}")
                continue
                
            logger.info(f"Processing {product_type} file: {input_file}")
            
            # Load CSV with better error handling
            try:
                df = pd.read_csv(input_file, dtype=str, encoding='utf-8-sig')
                df = df.fillna('')
                logger.info(f"Loaded {len(df)} rows from {product_type}")
            except Exception as e:
                logger.error(f"Error reading {input_file}: {e}")
                continue
            
            # Data cleaning and validation
            if 'shape' in df.columns:
                df['shape'] = df['shape'].str.strip().str.upper()
            
            # Generate unique IDs and remove duplicates
            df['unique_id'] = df.apply(lambda row: generate_unique_id(row, product_type), axis=1)
            
            # Remove duplicates within this file
            initial_count = len(df)
            df = df.drop_duplicates(subset=['unique_id'], keep='first')
            logger.info(f"Removed {initial_count - len(df)} duplicates within {product_type}")
            
            # Remove global duplicates (across all files)
            mask = ~df['unique_id'].isin(seen_ids)
            df = df[mask]
            duplicate_count += (~mask).sum()
            
            # Add processed IDs to global set
            seen_ids.update(df['unique_id'].tolist())
            
            # Price processing
            if 'price' in df.columns:
                df['price_numeric'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
                df['price'] = df['price_numeric'].apply(
                    lambda x: format_price(x, apply_markup, convert_to_cad)
                )
            
            # Enhanced image handling
            if 'image' in df.columns:
                df['image'] = df['image'].apply(lambda x: 
                    re.search(r'(https?://.*\.(jpg|jpeg|png|webp))', str(x)).group(1) 
                    if re.search(r'(https?://.*\.(jpg|jpeg|png|webp))', str(x)) else ''
                )
            
            # Smart image assignment with fallbacks
            df['image_link'] = df.apply(
                lambda row: (
                    row.get('image', '') if row.get('image', '') else
                    SHAPE_CONFIG.get(row.get('shape', ''), {}).get('url', 
                        'https://storage.googleapis.com/sitemaps.leeladiamond.com/shapes/default.jpg')
                ), axis=1
            )
            
            # Generate SEO-optimized content
            df['title'] = df.apply(lambda row: generate_seo_title(row, product_type), axis=1)
            df['description'] = df.apply(lambda row: generate_seo_description(row, product_type), axis=1)
            
            # Enhanced product URLs with tracking
            timestamp = datetime.now().strftime("%Y%m%d")
            df['link'] = df.apply(
                lambda row: f"https://leeladiamond.com/pages/{product_type.replace('_', '-')}-catalog"
                           f"?id={quote(str(row.get('ReportNo', '')))}"
                           f"&shape={quote(str(row.get('shape', '')))}"
                           f"&utm_source=google&utm_medium=merchant&utm_campaign={timestamp}", 
                axis=1
            )
            
            # Google Merchant Center required fields
            df['id'] = df['unique_id']
            df['availability'] = 'in_stock'
            df['brand'] = 'Leela Diamonds'
            df['mpn'] = df['unique_id']
            df['gtin'] = ''
            df['condition'] = 'new'
            df['age_group'] = 'adult'
            df['gender'] = 'unisex'
            df['material'] = 'Diamond' if product_type != 'gemstone' else 'Gemstone'
            
            # Enhanced Google product categorization
            df['google_product_category'] = df.apply(
                lambda row: SHAPE_CONFIG.get(
                    row.get('shape', ''), {}
                ).get('category_path', 'Jewelry & Watches > Fine Jewelry > Diamonds'),
                axis=1
            )
            
            # Additional Google Shopping fields
            df['product_type'] = df.apply(
                lambda row: f"{product_type.title()} > {row.get('shape', '').title()} > {row.get('carats', '')}ct",
                axis=1
            )
            
            # Custom labels for Google Ads optimization
            df['custom_label_0'] = product_type
            df['custom_label_1'] = df.apply(
                lambda row: 'premium' if pd.to_numeric(row.get('price_numeric', 0), errors='coerce') > 5000 else 'standard',
                axis=1
            )
            df['custom_label_2'] = df['shape']
            df['custom_label_3'] = df.apply(
                lambda row: f"{row.get('carats', '0')[:3]}ct" if row.get('carats') else '',
                axis=1
            )
            df['custom_label_4'] = df.apply(
                lambda row: f"{row.get('col', '')}-{row.get('clar', '')}",
                axis=1
            )
            
            # Shipping information
            df['shipping'] = 'CA::Standard:0.00 CAD,CA::Express:25.00 CAD'
            df['shipping_weight'] = '0.01 kg'
            
            # Select and order columns for final output
            final_columns = [
                'id', 'title', 'description', 'link', 'image_link', 'availability', 
                'price', 'brand', 'gtin', 'mpn', 'condition', 'google_product_category',
                'product_type', 'age_group', 'gender', 'material', 'shipping',
                'shipping_weight', 'custom_label_0', 'custom_label_1', 'custom_label_2',
                'custom_label_3', 'custom_label_4'
            ]
            
            # Ensure all required columns exist
            for col in final_columns:
                if col not in df.columns:
                    df[col] = ''
            
            df = df[final_columns]
            all_data.append(df)
            logger.info(f"Processed {len(df)} products from {product_type}")
        
        # Combine all data
        if not all_data:
            logger.error("No data to process")
            return
            
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} total products, removed {duplicate_count} cross-file duplicates")
        
        # Final data validation and cleaning
        combined_df = combined_df[combined_df['price'] != '0.00 CAD']  # Remove zero-price items
        combined_df = combined_df[combined_df['title'].str.len() > 10]  # Remove incomplete titles
        
        # Save to CSV with optimized settings
        combined_df.to_csv(
            output_file,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            escapechar='\\',
            encoding='utf-8'
        )
        
        # Generate summary report
        summary = {
            'total_products': len(combined_df),
            'by_type': combined_df['custom_label_0'].value_counts().to_dict(),
            'price_ranges': {
                'under_1000': len(combined_df[pd.to_numeric(combined_df['price'].str.replace(' CAD', ''), errors='coerce') < 1000]),
                '1000_5000': len(combined_df[
                    (pd.to_numeric(combined_df['price'].str.replace(' CAD', ''), errors='coerce') >= 1000) &
                    (pd.to_numeric(combined_df['price'].str.replace(' CAD', ''), errors='coerce') < 5000)
                ]),
                'over_5000': len(combined_df[pd.to_numeric(combined_df['price'].str.replace(' CAD', ''), errors='coerce') >= 5000])
            }
        }
        
        logger.info(f"Processing complete: {summary}")
        
        # Save summary report
        with open(os.path.join(local_output_directory, 'processing_summary.txt'), 'w') as f:
            f.write(f"Processing Summary - {datetime.now()}\n")
            f.write(f"Total products: {summary['total_products']}\n")
            f.write(f"By type: {summary['by_type']}\n")
            f.write(f"Price distribution: {summary['price_ranges']}\n")
        
        logger.info(f"Combined data saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in processing files: {e}")
       combined_df = combined_df[combined_df['price'] != '0.00 CAD']  # Remove zero-price items
        
        # Save to CSV with optimized settings
        combined_df.to_csv(
            output_file,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            escapechar='\\',
            encoding='utf-8'
        )
        
        # Generate summary report
        summary = {
            'total_products': len(combined_df),
            'by_type': combined_df['custom_label_0'].value_counts().to_dict(),
            'price_ranges': {
                'under_1000': len(combined_df[pd.to_numeric(combined_df['price'].str.replace(' CAD', ''), errors='coerce') < 1000]),
                '1000_5000': len(combined_df[
                    (pd.to_numeric(combined_df['price'].str.replace(' CAD', ''), errors='coerce') >= 1000) &
                    (pd.to_numeric(combined_df['price'].str.replace(' CAD', ''), errors='coerce') < 5000)
                ]),
                'over_5000': len(combined_df[pd.to_numeric(combined_df['price'].str.replace(' CAD', ''), errors='coerce') >= 5000])
            }
        }
        
        logger.info(f"Processing complete: {summary}")
        
        # Save summary report
        with open(os.path.join(local_output_directory, 'processing_summary.txt'), 'w') as f:
            f.write(f"Processing Summary - {datetime.now()}\n")
            f.write(f"Total products: {summary['total_products']}\n")
            f.write(f"By type: {summary['by_type']}\n")
            f.write(f"Price distribution: {summary['price_ranges']}\n")
        
        logger.info(f"Combined data saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in processing files: {e}")
        raise00 CAD']  # Remove zero-price items
        combined_df = combined_df[combined_df['title'].str.len() > 10]  # Remove incomplete titles
        
        # Save to CSV with optimized settings
        combined_df.to_csv(
            output_file,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            escapechar='\\',
            encoding='utf-8'
        )
        
        # Generate summary report
        summary = {
            'total_products': len(combined_df),
            'by_type': combined_df['custom_label_0'].value_counts().to_dict(),
            'price_ranges': {
                'under_1000': len(combined_df[pd.to_numeric(combined_df['price'].str.replace(' CAD', ''), errors='coerce') < 1000]),
                '1000_5000': len(combined_df[
                    (pd.to_numeric(combined_df['price'].str.replace(' CAD', ''), errors='coerce') >= 1000) &
                    (pd.to_numeric(combined_df['price'].str.replace(' CAD', ''), errors='coerce') < 5000)
                ]),
                'over_5000': len(combined_df[pd.to_numeric(combined_df['price'].str.replace(' CAD', ''), errors='coerce') >= 5000])
            }
        }
        
        logger.info(f"Processing complete: {summary}")
        
        # Save summary report
        with open(os.path.join(local_output_directory, 'processing_summary.txt'), 'w') as f:
            f.write(f"Processing Summary - {datetime.now()}\n")
            f.write(f"Total products: {summary['total_products']}\n")
            f.write(f"By type: {summary['by_type']}\n")
            f.write(f"Price distribution: {summary['price_ranges']}\n")
        
        logger.info(f"Combined data saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in processing files: {e}")
        raise

# ----------------------------
# GOOGLE CLOUD UPLOAD FUNCTION
# ----------------------------

def upload_files_to_bucket(bucket_name, bucket_folder, local_directory):
    """Upload files to GCS with metadata and caching optimization."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        uploaded_files = []
        for file_name in os.listdir(local_directory):
            file_path = os.path.join(local_directory, file_name)
            if os.path.isfile(file_path):
                destination_blob_name = f"{bucket_folder}/{file_name}"
                blob = bucket.blob(destination_blob_name)
                
                # Set metadata for better performance
                if file_name.endswith('.csv'):
                    blob.content_type = 'text/csv'
                    blob.cache_control = 'public, max-age=3600'  # 1 hour cache
                
                blob.upload_from_filename(file_path)
                uploaded_files.append(destination_blob_name)
                logger.info(f"Uploaded {file_name} to {destination_blob_name}")
        
        logger.info(f"Successfully uploaded {len(uploaded_files)} files")
        return uploaded_files
        
    except Exception as e:
        logger.error(f"Error during upload: {e}")
        raise

# ----------------------------
# MAIN AUTOMATION WORKFLOW
# ----------------------------

def run_workflow():
    """Execute the complete workflow with error handling and logging."""
    start_time = datetime.now()
    logger.info(f"Starting workflow at {start_time}")
    
    try:
        # Step 1: Download files from FTP
        logger.info("Step 1: Downloading files from FTP")
        success_count = download_all_files()
        if success_count == 0:
            raise Exception("No files downloaded successfully")
        
        # Step 2: Process files
        logger.info("Step 2: Processing files")
        files_to_load = {
            "natural": {"file_path": os.path.join(ftp_download_dir, "Natural.csv")},
            "lab_grown": {"file_path": os.path.join(ftp_download_dir, "Labgrown.csv")},
            "gemstone": {"file_path": os.path.join(ftp_download_dir, "gemstones.csv")}
        }
        
        output_file = os.path.join(local_output_directory, "combined_google_merchant_feed.csv")
        process_files_to_cad(files_to_load, output_file)
        
        # Step 3: Upload to Google Cloud Storage
        logger.info("Step 3: Uploading to Google Cloud Storage")
        uploaded_files = upload_files_to_bucket(bucket_name, bucket_folder, local_output_directory)
        
        # Step 4: Cleanup temporary files
        logger.info("Step 4: Cleaning up temporary files")
        for product_type, file_info in ftp_files.items():
            try:
                if os.path.exists(file_info["local_path"]):
                    os.remove(file_info["local_path"])
            except Exception as e:
                logger.warning(f"Could not clean up {file_info['local_path']}: {e}")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        result = {
            "status": "success",
            "message": "Workflow completed successfully",
            "duration": str(duration),
            "files_processed": success_count,
            "files_uploaded": len(uploaded_files),
            "timestamp": end_time.isoformat()
        }
        
        logger.info(f"Workflow completed successfully in {duration}")
        return result
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ----------------------------
# CLOUD FUNCTION ENTRY POINT
# ----------------------------

def cloud_function_entry(request):
    """HTTP Cloud Function entry point with enhanced error handling."""
    try:
        result = run_workflow()
        status_code = 200 if result["status"] == "success" else 500
        return jsonify(result), status_code
    except Exception as e:
        logger.error(f"Cloud function error: {e}")
        return jsonify({
            "status": "error", 
            "message": f"Unexpected error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

# ----------------------------
# MAIN EXECUTION
# ----------------------------

if __name__ == "__main__":
    # For local testing
    result = run_workflow()
    print(f"Result: {result}")

