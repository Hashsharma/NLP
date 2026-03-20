"""
WordNet Vocabulary Extractor for English Vocabulary Quiz App
Extracts words, definitions, examples, synonyms, and difficulty levels
"""

import requests
import tarfile
import io
import os
import re
import json
import csv
import sqlite3
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import statistics
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WordNetExtractor:
    """Extracts and processes WordNet data for vocabulary quiz application"""
    
    def __init__(self, output_dir: str = "vocabulary_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # POS mapping
        self.POS_MAP = {
            'n': 'noun',
            'v': 'verb',
            'a': 'adjective',
            's': 'adjective',  # satellite adjective
            'r': 'adverb'
        }
        
        # Reverse mapping for database
        self.POS_ID_MAP = {
            'noun': 1,
            'verb': 2,
            'adjective': 3,
            'adverb': 4,
            'preposition': 5,
            'conjunction': 6,
            'pronoun': 7,
            'interjection': 8
        }
        
        # Difficulty levels mapping (CEFR)
        self.DIFFICULTY_LEVELS = {
            'A1': 1,  # Beginner
            'A2': 2,  # Elementary
            'B1': 3,  # Intermediate
            'B2': 4,  # Upper Intermediate
            'C1': 5,  # Advanced
            'C2': 6   # Proficient
        }
        
        # Common word frequency list (for difficulty estimation)
        self.common_words = self._load_common_words()
        
    def _load_common_words(self) -> Set[str]:
        """Load common English words for frequency reference"""
        common_words = set()
        try:
            # Basic common words list
            basic_words = {
                'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
                'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
                'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
                'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there',
                'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
                'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no',
                'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
                'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
                'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
                'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
                'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
                'give', 'day', 'most', 'us'
            }
            common_words.update(basic_words)
        except Exception as e:
            logger.warning(f"Could not load frequency list: {e}")
        
        return common_words
    
    def download_wordnet(self) -> bool:
        """Download and extract WordNet 3.0"""
        url = "http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz"
        
        try:
            logger.info("Downloading WordNet 3.0...")
            response = requests.get(url, stream=True, timeout=60)
            
            if response.status_code == 200:
                # Create extraction directory
                extract_dir = self.output_dir / "wordnet_raw"
                extract_dir.mkdir(exist_ok=True)
                
                # Extract the archive
                logger.info("Extracting WordNet files...")
                with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:gz') as tar:
                    tar.extractall(extract_dir)
                
                logger.info("WordNet downloaded and extracted successfully")
                return True
            else:
                logger.error(f"Failed to download WordNet. Status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading WordNet: {e}")
            return False
    
    def parse_wordnet_data(self) -> Dict:
        """Parse WordNet data files and extract vocabulary"""
        
        wordnet_dir = self.output_dir / "wordnet_raw" / "WordNet-3.0" / "dict"
        if not wordnet_dir.exists():
            logger.error(f"WordNet directory not found: {wordnet_dir}")
            return {}
        
        vocabulary = {}
        
        # Parse each data file
        data_files = [
            ('n', wordnet_dir / "data.noun"),
            ('v', wordnet_dir / "data.verb"),
            ('a', wordnet_dir / "data.adj"),
            ('r', wordnet_dir / "data.adv")
        ]
        
        for pos_code, data_file in data_files:
            if not data_file.exists():
                logger.warning(f"Data file not found: {data_file}")
                continue
            
            pos_name = self.POS_MAP.get(pos_code, pos_code)
            logger.info(f"Processing {pos_name}s from {data_file.name}...")
            
            with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if line.startswith('  '):
                        continue
                    
                    try:
                        word_entry = self._parse_synset_line(line, pos_code)
                        if word_entry:
                            for word in word_entry['words']:
                                if word not in vocabulary:
                                    vocabulary[word] = {
                                        'word': word,
                                        'definitions': [],
                                        'examples': [],
                                        'synonyms': defaultdict(set),
                                        'part_of_speech': pos_name,
                                        'difficulty': self._estimate_difficulty(word),
                                        'frequency': 0,
                                        'wordnet_offset': word_entry['offset']
                                    }
                                
                                # Add definition if unique
                                if word_entry['definition'] and word_entry['definition'] not in vocabulary[word]['definitions']:
                                    vocabulary[word]['definitions'].append(word_entry['definition'])
                                
                                # Add example if exists
                                if word_entry['example'] and word_entry['example'] not in vocabulary[word]['examples']:
                                    vocabulary[word]['examples'].append(word_entry['example'])
                                
                                # Add synonyms (other words in this synset)
                                for synonym in word_entry['words']:
                                    if synonym != word:
                                        vocabulary[word]['synonyms'][pos_name].add(synonym)
                    
                    except Exception as e:
                        logger.debug(f"Error parsing line {line_num}: {e}")
                        continue
            
            logger.info(f"  Processed {len([w for w in vocabulary.values() if w['part_of_speech'] == pos_name])} {pos_name}s")
        
        logger.info(f"Total unique words extracted: {len(vocabulary)}")
        return vocabulary
    
    def _parse_synset_line(self, line: str, pos_code: str) -> Optional[Dict]:
        """Parse a single synset line from WordNet data file"""
        line = line.strip()
        if not line:
            return None
        
        parts = line.split()
        if len(parts) < 5:
            return None
        
        # Parse synset header
        synset_offset = parts[0]
        lex_filenum = parts[1]
        ss_type = parts[2]  # Should match pos_code
        
        # Number of words in synset (in hex)
        word_count = int(parts[3], 16)
        
        # Extract words
        words = []
        for i in range(word_count):
            word_idx = 4 + i * 2
            if word_idx < len(parts):
                word = parts[word_idx].replace('_', ' ').lower().strip()
                if word:  # Skip empty words
                    words.append(word)
        
        if not words:
            return None
        
        # Extract gloss (definition and example)
        gloss = ""
        if '|' in line:
            gloss = line.split('|', 1)[1].strip()
        
        definition, example = self._extract_definition_and_example(gloss)
        
        return {
            'offset': synset_offset,
            'type': ss_type,
            'words': words,
            'definition': definition,
            'example': example,
            'raw_line': line
        }
    
    def _extract_definition_and_example(self, gloss: str) -> Tuple[str, str]:
        """Extract definition and example from gloss"""
        definition = ""
        example = ""
        
        if not gloss:
            return definition, example
        
        # Clean up the gloss
        gloss = re.sub(r'\s+', ' ', gloss).strip()
        
        # Try to split by semicolon or quotation marks
        if ';' in gloss:
            parts = gloss.split(';', 1)
            definition = parts[0].strip()
            if len(parts) > 1:
                example = parts[1].strip()
                # Clean up example
                example = re.sub(r'^"|"$', '', example)  # Remove quotes
                example = re.sub(r'^\'', '', example)  # Remove single quotes
        elif '"' in gloss:
            # Definition might be in quotes
            match = re.search(r'"([^"]+)"', gloss)
            if match:
                example = match.group(1)
                definition = re.sub(r'"([^"]+)"', '', gloss).strip()
        else:
            definition = gloss
        
        # Clean up definition
        definition = definition.strip(';.,"\' ')
        
        return definition, example
    
    def _estimate_difficulty(self, word: str) -> str:
        """Estimate CEFR difficulty level for a word"""
        # Simple heuristics based on word characteristics
        word = word.lower()
        
        # Check if it's a very common word
        if word in self.common_words or len(word) <= 3:
            return 'A1'
        
        # Check word length
        if len(word) <= 5:
            return 'A2'
        elif len(word) <= 7:
            # Check for common prefixes/suffixes
            common_patterns = ['un', 're', 'dis', 'ing', 'ed', 'ly', 'er', 'est']
            if any(word.startswith(p) or word.endswith(p) for p in common_patterns):
                return 'B1'
            return 'B2'
        elif len(word) <= 9:
            return 'B2'
        elif len(word) <= 11:
            return 'C1'
        else:
            return 'C2'
    
    def filter_and_clean_vocabulary(self, vocabulary: Dict, min_definitions: int = 1) -> List[Dict]:
        """Filter and clean vocabulary data"""
        cleaned_vocabulary = []
        
        logger.info("Filtering and cleaning vocabulary...")
        
        for word, data in vocabulary.items():
            # Skip words that don't meet criteria
            if not self._should_include_word(word, data):
                continue
            
            # Skip words without definitions
            if len(data['definitions']) < min_definitions:
                continue
            
            # Get best definition (first one)
            best_definition = data['definitions'][0] if data['definitions'] else ""
            
            # Get best example (first one)
            best_example = data['examples'][0] if data['examples'] else ""
            
            # Get all synonyms
            all_synonyms = []
            for pos_synonyms in data['synonyms'].values():
                all_synonyms.extend(list(pos_synonyms))
            all_synonyms = list(set(all_synonyms))  # Remove duplicates
            
            # Create cleaned entry
            entry = {
                'word': word,
                'part_of_speech': data['part_of_speech'],
                'definition': best_definition,
                'example': best_example,
                'difficulty_level': data['difficulty'],
                'synonyms': all_synonyms,
                'wordnet_offset': data['wordnet_offset'],
                'definition_count': len(data['definitions']),
                'example_count': len(data['examples'])
            }
            
            cleaned_vocabulary.append(entry)
        
        # Sort by word
        cleaned_vocabulary.sort(key=lambda x: x['word'])
        
        logger.info(f"After filtering: {len(cleaned_vocabulary)} words")
        return cleaned_vocabulary
    
    def _should_include_word(self, word: str, data: Dict) -> bool:
        """Determine if a word should be included in the vocabulary"""
        # Skip words with special characters (except hyphens in compound words)
        if re.search(r'[^a-zA-Z\- ]', word):
            return False
        
        # Skip single letters (except 'a' and 'i')
        if len(word) == 1 and word not in ['a', 'i']:
            return False
        
        # Skip proper nouns (capitalized in WordNet)
        if word[0].isupper():
            return False
        
        # Skip words that are just numbers
        if word.replace('-', '').replace(' ', '').isdigit():
            return False
        
        # Skip very obscure or technical words
        if len(word) > 20:
            return False
        
        # Skip words that are actually phrases (more than 2 words)
        if len(word.split()) > 2:
            return False
        
        return True
    
    def export_to_csv(self, vocabulary: List[Dict], filename: str = "vocabulary.csv"):
        """Export vocabulary to CSV file"""
        csv_path = self.output_dir / filename
        
        # Define CSV headers - only fields we want to export
        fieldnames = [
            'word', 'part_of_speech', 'definition', 'example', 
            'difficulty_level', 'synonyms', 'wordnet_offset'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in vocabulary:
                # Create a clean dictionary with only the fields we want
                csv_entry = {
                    'word': entry['word'],
                    'part_of_speech': entry['part_of_speech'],
                    'definition': entry['definition'][:500] if entry['definition'] else '',  # Limit length
                    'example': entry['example'][:500] if entry['example'] else '',  # Limit length
                    'difficulty_level': entry['difficulty_level'],
                    'synonyms': '; '.join(entry['synonyms'][:10]),  # Join and limit
                    'wordnet_offset': entry['wordnet_offset']
                }
                writer.writerow(csv_entry)
        
        logger.info(f"Exported {len(vocabulary)} words to {csv_path}")
    
    def export_to_sql(self, vocabulary: List[Dict], db_name: str = "vocabulary.db"):
        """Export vocabulary to SQLite database"""
        db_path = self.output_dir / db_name
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS parts_of_speech (
                    pos_id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    abbreviation TEXT,
                    description TEXT,
                    sort_order INTEGER DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS difficulty_levels (
                    level_id INTEGER PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    sort_order INTEGER DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS words (
                    word_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT UNIQUE NOT NULL,
                    part_of_speech_id INTEGER,
                    definition TEXT NOT NULL,
                    example_sentence TEXT,
                    phonetic_spelling TEXT,
                    difficulty_level_id INTEGER DEFAULT 1,
                    wordnet_offset TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (part_of_speech_id) REFERENCES parts_of_speech(pos_id),
                    FOREIGN KEY (difficulty_level_id) REFERENCES difficulty_levels(level_id)
                )
            ''')
            
            # Create synonyms table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS synonyms (
                    synonym_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word_id INTEGER NOT NULL,
                    synonym TEXT NOT NULL,
                    FOREIGN KEY (word_id) REFERENCES words(word_id) ON DELETE CASCADE,
                    UNIQUE(word_id, synonym)
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_word ON words(word)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_difficulty ON words(difficulty_level_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_synonyms_word ON synonyms(word_id)')
            
            # Insert parts of speech
            for pos_name, pos_id in self.POS_ID_MAP.items():
                cursor.execute('''
                    INSERT OR IGNORE INTO parts_of_speech (pos_id, name) 
                    VALUES (?, ?)
                ''', (pos_id, pos_name))
            
            # Insert difficulty levels
            for level_name, level_id in self.DIFFICULTY_LEVELS.items():
                cursor.execute('''
                    INSERT OR IGNORE INTO difficulty_levels (level_id, name) 
                    VALUES (?, ?)
                ''', (level_id, level_name))
            
            # Insert words and synonyms
            for entry in vocabulary:
                pos_id = self.POS_ID_MAP.get(entry['part_of_speech'], 1)
                difficulty_id = self.DIFFICULTY_LEVELS.get(entry['difficulty_level'], 3)
                
                # Insert word
                cursor.execute('''
                    INSERT OR REPLACE INTO words 
                    (word, part_of_speech_id, definition, example_sentence, 
                     difficulty_level_id, wordnet_offset)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    entry['word'],
                    pos_id,
                    entry['definition'][:1000] if entry['definition'] else '',
                    entry['example'][:500] if entry['example'] else '',
                    difficulty_id,
                    entry['wordnet_offset']
                ))
                
                # Get the word_id
                word_id = cursor.lastrowid
                if not word_id:
                    cursor.execute('SELECT word_id FROM words WHERE word = ?', (entry['word'],))
                    result = cursor.fetchone()
                    word_id = result[0] if result else None
                
                # Insert synonyms
                if word_id and entry['synonyms']:
                    for synonym in entry['synonyms'][:20]:  # Limit to 20 synonyms
                        try:
                            cursor.execute('''
                                INSERT OR IGNORE INTO synonyms (word_id, synonym)
                                VALUES (?, ?)
                            ''', (word_id, synonym))
                        except sqlite3.Error as e:
                            logger.debug(f"Could not insert synonym: {e}")
            
            conn.commit()
            logger.info(f"Exported {len(vocabulary)} words to SQLite database: {db_path}")
            
            # Show statistics
            cursor.execute("SELECT COUNT(*) FROM words")
            word_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM synonyms")
            synonym_count = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT dl.name, COUNT(*) 
                FROM words w 
                JOIN difficulty_levels dl ON w.difficulty_level_id = dl.level_id 
                GROUP BY dl.name
                ORDER BY dl.sort_order
            ''')
            difficulty_stats = cursor.fetchall()
            
            logger.info(f"Database contains {word_count} words and {synonym_count} synonyms")
            logger.info("Difficulty distribution:")
            for level, count in difficulty_stats:
                logger.info(f"  {level}: {count} words")
            
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
    
    def export_to_json(self, vocabulary: List[Dict], filename: str = "vocabulary.json"):
        """Export vocabulary to JSON file"""
        json_path = self.output_dir / filename
        
        # Create a simplified version for JSON
        json_data = []
        for entry in vocabulary:
            json_entry = {
                'word': entry['word'],
                'part_of_speech': entry['part_of_speech'],
                'definition': entry['definition'],
                'example': entry['example'],
                'difficulty_level': entry['difficulty_level'],
                'synonyms': entry['synonyms'][:15],  # Limit synonyms
                'wordnet_offset': entry['wordnet_offset']
            }
            json_data.append(json_entry)
        
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported {len(vocabulary)} words to {json_path}")
    
    def generate_statistics_report(self, vocabulary: List[Dict]):
        """Generate statistics report about the vocabulary"""
        report_path = self.output_dir / "vocabulary_stats.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("VOCABULARY DATABASE STATISTICS\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic counts
            f.write(f"Total words: {len(vocabulary)}\n\n")
            
            # POS distribution
            pos_counts = Counter(item['part_of_speech'] for item in vocabulary)
            f.write("Parts of Speech Distribution:\n")
            f.write("-" * 40 + "\n")
            for pos, count in pos_counts.most_common():
                percentage = (count / len(vocabulary)) * 100
                f.write(f"{pos.title():15} {count:6} ({percentage:5.1f}%)\n")
            f.write("\n")
            
            # Difficulty distribution
            diff_counts = Counter(item['difficulty_level'] for item in vocabulary)
            f.write("Difficulty Level Distribution:\n")
            f.write("-" * 40 + "\n")
            difficulty_order = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
            for diff in difficulty_order:
                if diff in diff_counts:
                    count = diff_counts[diff]
                    percentage = (count / len(vocabulary)) * 100
                    f.write(f"{diff:15} {count:6} ({percentage:5.1f}%)\n")
            f.write("\n")
            
            # Word length statistics
            word_lengths = [len(item['word']) for item in vocabulary]
            f.write("Word Length Statistics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average length: {statistics.mean(word_lengths):.1f} characters\n")
            f.write(f"Minimum length: {min(word_lengths)} characters\n")
            f.write(f"Maximum length: {max(word_lengths)} characters\n")
            f.write(f"Most common length: {Counter(word_lengths).most_common(1)[0][0]} characters\n")
            f.write("\n")
            
            # Top 10 longest words
            longest_words = sorted(vocabulary, key=lambda x: len(x['word']), reverse=True)[:10]
            f.write("Top 10 Longest Words:\n")
            f.write("-" * 40 + "\n")
            for item in longest_words:
                f.write(f"{len(item['word']):3} chars: {item['word']}\n")
            
        logger.info(f"Statistics report saved to {report_path}")
    
    def run_full_extraction(self, export_formats: List[str] = ['csv', 'sql', 'json']):
        """Run complete extraction pipeline"""
        logger.info("Starting WordNet vocabulary extraction...")
        
        # Step 1: Download WordNet
        if not self.download_wordnet():
            logger.error("Failed to download WordNet. Exiting.")
            return
        
        # Step 2: Parse WordNet data
        raw_vocabulary = self.parse_wordnet_data()
        if not raw_vocabulary:
            logger.error("No vocabulary extracted. Exiting.")
            return
        
        # Step 3: Filter and clean
        cleaned_vocabulary = self.filter_and_clean_vocabulary(raw_vocabulary)
        
        # Step 4: Export to various formats
        if 'csv' in export_formats:
            self.export_to_csv(cleaned_vocabulary)
        
        if 'sql' in export_formats:
            self.export_to_sql(cleaned_vocabulary)
        
        if 'json' in export_formats:
            self.export_to_json(cleaned_vocabulary)
        
        # Step 5: Generate statistics
        self.generate_statistics_report(cleaned_vocabulary)
        
        logger.info("Vocabulary extraction completed successfully!")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        
        return cleaned_vocabulary


def main():
    """Main execution function"""
    print("=" * 60)
    print("WORDNET VOCABULARY EXTRACTOR")
    print("For English Vocabulary Quiz Application")
    print("=" * 60)
    
    # Create extractor instance
    extractor = WordNetExtractor(output_dir="vocabulary_data")
    
    # Run full extraction pipeline (you can customize export formats)
    vocabulary = extractor.run_full_extraction(export_formats=['csv', 'json'])
    
    # Show output files
    print("\n" + "=" * 60)
    print("OUTPUT FILES GENERATED:")
    print("=" * 60)
    
    output_dir = Path("vocabulary_data")
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        files.sort(key=lambda x: x.stat().st_size, reverse=True)
        
        for file in files:
            if file.is_file():
                size = file.stat().st_size
                size_str = f"{size:,} bytes" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
                print(f"  {file.name:30} {size_str:>15}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Import vocabulary.csv or vocabulary.db into your application")
    print("2. Check vocabulary_stats.txt for database statistics")
    print("3. Use vocabulary.json for web/mobile applications")
    print("4. Customize difficulty levels as needed")
    print("5. Add more words from other sources if required")
    print("=" * 60)


if __name__ == "__main__":
    main()