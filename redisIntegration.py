import redis
import hashlib
import json
import binascii
from audioHash import AudioHash

class RedisAudioHashSearch:
    """
    Redis integration for audio fingerprint search using AudioHash
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        """Initialize with Redis connection"""
        self.redis = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.audio_hash = AudioHash()
        self._load_lua_scripts()
    
    def _load_lua_scripts(self):
        """Load and register Lua scripts with Redis"""
        # Script for decoding binary hash to peaks
        decode_script = """
        local function decode_peaks(binary_data)
            if not binary_data or #binary_data == 0 then
                return {}
            end
            
            -- Convert binary data to bit representation
            local bits = {}
            for i = 1, #binary_data do
                local byte = string.byte(binary_data, i)
                for j = 7, 0, -1 do
                    bits[#bits + 1] = bit.band(bit.rshift(byte, j), 1)
                end
            end
            
            -- Extract version (1 byte)
            local version = 0
            for i = 1, 8 do
                version = bit.lshift(version, 1) + bits[i]
            end
            
            if version ~= 3 then
                return error("Unsupported version: " .. version)
            end
            
            -- Extract number of peaks (2 bytes)
            local num_peaks = 0
            for i = 9, 24 do
                num_peaks = bit.lshift(num_peaks, 1) + bits[i]
            end
            
            -- Constants for decoding (must match Python values)
            local FREQ_BITS = 10
            local MINUTE_BITS = 6
            local SECOND_BITS = 12
            local AMP_BITS = 10
            local BITS_PER_PEAK = FREQ_BITS + MINUTE_BITS + SECOND_BITS + AMP_BITS
            
            local FREQ_SCALE = 1000
            local SECOND_SCALE = 68.31
            local AMP_SCALE = 1000
            
            -- Extract peaks
            local peaks = {}
            local bit_pos = 25  -- Start position after header
            
            for p = 1, num_peaks do
                -- Extract frequency (10 bits)
                local freq_int = 0
                for i = 0, FREQ_BITS - 1 do
                    freq_int = bit.lshift(freq_int, 1) + bits[bit_pos + i]
                end
                
                -- Extract minutes (6 bits)
                local minutes_int = 0
                for i = FREQ_BITS, FREQ_BITS + MINUTE_BITS - 1 do
                    minutes_int = bit.lshift(minutes_int, 1) + bits[bit_pos + i]
                end
                
                -- Extract seconds (12 bits)
                local seconds_int = 0
                local offset = FREQ_BITS + MINUTE_BITS
                for i = offset, offset + SECOND_BITS - 1 do
                    seconds_int = bit.lshift(seconds_int, 1) + bits[bit_pos + i]
                end
                
                -- Extract amplitude (10 bits)
                local amp_int = 0
                offset = FREQ_BITS + MINUTE_BITS + SECOND_BITS
                for i = offset, offset + AMP_BITS - 1 do
                    amp_int = bit.lshift(amp_int, 1) + bits[bit_pos + i]
                end
                
                -- Convert back to float values
                local freq = freq_int / FREQ_SCALE
                local seconds = seconds_int / SECOND_SCALE
                local time_seconds = minutes_int * 60 + seconds
                local amp = amp_int / AMP_SCALE
                
                -- Add peak to list
                peaks[p] = {freq, time_seconds, amp}
                bit_pos = bit_pos + BITS_PER_PEAK
            end
            
            return cjson.encode(peaks)
        end
        
        return decode_peaks(ARGV[1])
        """
        
        # Updated compare script to return hash keys directly
        compare_script = """
        local function decode_peaks(binary_data)
            if not binary_data or #binary_data == 0 then
                return {}
            end
            
            -- Convert binary data to bit representation
            local bits = {}
            for i = 1, #binary_data do
                local byte = string.byte(binary_data, i)
                for j = 7, 0, -1 do
                    bits[#bits + 1] = bit.band(bit.rshift(byte, j), 1)
                end
            end
            
            -- Extract version (1 byte)
            local version = 0
            for i = 1, 8 do
                version = bit.lshift(version, 1) + bits[i]
            end
            
            if version ~= 3 then
                return error("Unsupported version: " .. version)
            end
            
            -- Extract number of peaks (2 bytes)
            local num_peaks = 0
            for i = 9, 24 do
                num_peaks = bit.lshift(num_peaks, 1) + bits[i]
            end
            
            -- Constants for decoding (must match Python values)
            local FREQ_BITS = 10
            local MINUTE_BITS = 6
            local SECOND_BITS = 12
            local AMP_BITS = 10
            local BITS_PER_PEAK = FREQ_BITS + MINUTE_BITS + SECOND_BITS + AMP_BITS
            
            local FREQ_SCALE = 1000
            local SECOND_SCALE = 68.31
            local AMP_SCALE = 1000
            
            -- Extract peaks
            local peaks = {}
            local bit_pos = 25  -- Start position after header
            
            for p = 1, num_peaks do
                -- Extract frequency (10 bits)
                local freq_int = 0
                for i = 0, FREQ_BITS - 1 do
                    freq_int = bit.lshift(freq_int, 1) + bits[bit_pos + i]
                end
                
                -- Extract minutes (6 bits)
                local minutes_int = 0
                for i = FREQ_BITS, FREQ_BITS + MINUTE_BITS - 1 do
                    minutes_int = bit.lshift(minutes_int, 1) + bits[bit_pos + i]
                end
                
                -- Extract seconds (12 bits)
                local seconds_int = 0
                local offset = FREQ_BITS + MINUTE_BITS
                for i = offset, offset + SECOND_BITS - 1 do
                    seconds_int = bit.lshift(seconds_int, 1) + bits[bit_pos + i]
                end
                
                -- Extract amplitude (10 bits)
                local amp_int = 0
                offset = FREQ_BITS + MINUTE_BITS + SECOND_BITS
                for i = offset, offset + AMP_BITS - 1 do
                    amp_int = bit.lshift(amp_int, 1) + bits[bit_pos + i]
                end
                
                -- Convert back to float values
                local freq = freq_int / FREQ_SCALE
                local seconds = seconds_int / SECOND_SCALE
                local time_seconds = minutes_int * 60 + seconds
                local amp = amp_int / AMP_SCALE
                
                -- Add peak to list
                peaks[p] = {freq, time_seconds, amp}
                bit_pos = bit_pos + BITS_PER_PEAK
            end
            
            return peaks
        end
        
        -- Linear assignment (simplified Hungarian algorithm)
        local function linear_assignment(cost_matrix, n)
            -- This is a simplified version of the Hungarian algorithm
            -- For a complete implementation, you might want to use a more optimized approach
            
            local row_ind = {}
            local col_ind = {}
            
            -- Greedy assignment (not optimal but simpler for Lua)
            local used_cols = {}
            
            for i = 1, n do
                local min_val = math.huge
                local min_col = 0
                
                for j = 1, n do
                    if not used_cols[j] and cost_matrix[i][j] < min_val then
                        min_val = cost_matrix[i][j]
                        min_col = j
                    end
                end
                
                if min_col > 0 then
                    row_ind[#row_ind + 1] = i
                    col_ind[#col_ind + 1] = min_col
                    used_cols[min_col] = true
                end
            end
            
            return row_ind, col_ind
        end
        
        local function compare_peaks(query_binary_data, stored_binary_data, time_tolerance, freq_tolerance, amp_tolerance)
            -- Decode both binary data to get peaks
            local peaks1 = decode_peaks(query_binary_data)
            local peaks2 = decode_peaks(stored_binary_data)
            
            -- Handle edge cases
            if #peaks1 == 0 and #peaks2 == 0 then
                return 1.0
            end
            if #peaks1 == 0 or #peaks2 == 0 then
                return 0.0
            end
            if #peaks1 ~= #peaks2 then
                return 0.0
            end
            
            local n = #peaks1
            -- Create cost matrix
            local cost_matrix = {}
            
            -- Fill cost matrix with manhattan distances
            for i = 1, n do
                cost_matrix[i] = {}
                local p1 = peaks1[i]
                
                for j = 1, n do
                    local p2 = peaks2[j]
                    
                    -- Calculate manhattan distance using all three dimensions
                    local full_distance = 
                        math.abs(p1[2] - p2[2])/time_tolerance +  -- time difference
                        math.abs(p1[1] - p2[1])/freq_tolerance +  -- frequency difference
                        math.abs(p1[3] - p2[3])/amp_tolerance    -- amplitude difference
                    
                    cost_matrix[i][j] = full_distance
                end
            end
            
            -- Use assignment algorithm to find optimal matching
            local row_ind, col_ind = linear_assignment(cost_matrix, n)
            
            -- Calculate similarity from matched pairs
            local total_similarity = 0.0
            
            for k = 1, #row_ind do
                local i, j = row_ind[k], col_ind[k]
                local distance = cost_matrix[i][j]
                
                -- Convert distance to similarity score
                local similarity = math.max(0.0, 1.0 - (distance / 3.0))
                total_similarity = total_similarity + similarity
            end
            
            return total_similarity / n
        end
        
        -- Get parameters
        local query_hash = ARGV[1]
        local time_tolerance = tonumber(ARGV[2]) or 0.01
        local freq_tolerance = tonumber(ARGV[3]) or 0.01
        local amp_tolerance = tonumber(ARGV[4]) or 0.05
        local threshold = tonumber(ARGV[5]) or 0.7
        local max_results = tonumber(ARGV[6]) or 5
        
        -- Get all binary hashes from Redis (use hex keys)
        local all_keys = redis.call('KEYS', 'audio_hash:*')
        local results = {}
        
        for i, key in ipairs(all_keys) do
            -- Get the binary hash from the mapping
            local stored_hash = redis.call('GET', key)
            -- Get the value associated with this hash
            local hash_value_key = string.gsub(key, "audio_hash:", "audio_hash_val:")
            local hash_value = redis.call('GET', hash_value_key)
            
            local similarity = compare_peaks(query_hash, stored_hash, time_tolerance, freq_tolerance, amp_tolerance)
            
            if similarity >= threshold then
                -- Return both the similarity and the stored value
                table.insert(results, {key, similarity, hash_value})
            end
        end
        
        -- Sort results by similarity (descending)
        table.sort(results, function(a, b) return a[2] > b[2] end)
        
        -- Return top results
        local top_results = {}
        for i = 1, math.min(max_results, #results) do
            table.insert(top_results, {
                results[i][1],  -- hash key
                results[i][2],  -- similarity score
                results[i][3]   -- stored value
            })
        end
        
        return cjson.encode(top_results)
        """
        
        # Register scripts with Redis
        self.decode_script_sha = self.redis.script_load(decode_script)
        self.compare_script_sha = self.redis.script_load(compare_script)
    
    def store_hash(self, audio_path, value, num_peaks=256, hash_id=None):
        """
        Process audio file and store its hash in Redis with associated value
        
        Args:
            audio_path: Path to the audio file
            value: The value to associate with this audio hash
            num_peaks: Number of peaks to use for hashing
            
        Returns:
            The key under which the hash was stored
        """
        # Generate hash
        peaks = self.audio_hash.create_time_sensitive_hash(audio_path, num_peaks)
        binary_hash = self.audio_hash.encode_peaks(peaks)
        
        # Create a unique ID for the hash
        hash_id = hash_id if hash_id else hashlib.md5(binary_hash).hexdigest()
        
        # Store both the binary hash and the associated value
        hash_key = f"audio_hash:{hash_id}"
        value_key = f"audio_hash_val:{hash_id}"
        
        # Store the binary hash and its value
        self.redis.set(hash_key, binary_hash)
        self.redis.set(value_key, value)
        
        print("Redis Set done", self.redis.get(value_key))
        
        return hash_key
    
    def store_binary_hash(self, binary_hash, value):
        """
        Store a pre-computed binary hash in Redis with associated value
        
        Args:
            binary_hash: The binary hash to store
            value: The value to associate with this hash
            
        Returns:
            The key under which the hash was stored
        """
        # Create a unique ID for the hash
        hash_id = hashlib.md5(binary_hash).hexdigest()
        
        # Store both the binary hash and the associated value
        hash_key = f"audio_hash:{hash_id}"
        value_key = f"audio_hash_val:{hash_id}"
        
        # Store the binary hash and its value
        self.redis.set(hash_key, binary_hash)
        self.redis.set(value_key, value)
        
        return hash_key
    
    def find_similar(self, audio_path=None, binary_hash=None, num_peaks=256, 
                   time_tolerance=0.01, freq_tolerance=0.01, amp_tolerance=0.05,
                   threshold=0.7, max_results=5):
        """
        Find similar audio files in Redis and return their associated values
        
        Args:
            audio_path: Path to the query audio file (optional if binary_hash is provided)
            binary_hash: Binary hash to query (optional if audio_path is provided)
            num_peaks: Number of peaks to use for hashing (only if audio_path is provided)
            time_tolerance: Tolerance for time differences
            freq_tolerance: Tolerance for frequency differences
            amp_tolerance: Tolerance for amplitude differences
            threshold: Minimum similarity score to include in results
            max_results: Maximum number of results to return
            
        Returns:
            A list of dictionaries with 'key', 'similarity', and 'value' for matching audio files
        """
        # Get query hash
        if binary_hash is None:
            if audio_path is None:
                raise ValueError("Either audio_path or binary_hash must be provided")
            peaks = self.audio_hash.create_time_sensitive_hash(audio_path, num_peaks)
            binary_hash = self.audio_hash.encode_peaks(peaks)
        
        # Call Lua script to find similar hashes
        result = self.redis.evalsha(
            self.compare_script_sha,
            0,  # No keys used in script
            binary_hash,
            time_tolerance,
            freq_tolerance,
            amp_tolerance,
            threshold,
            max_results
        )
        
        # Parse results
        matches = json.loads(result)
        print("Matches: ", matches)
        # Return results with values
        results = []
        for key, similarity, value in matches:
            # The value is already returned from the Lua script
            results.append({
                'key': key,
                'similarity': similarity,
                'value': value.decode('utf-8') if isinstance(value, bytes) else value
            })
        
        return results
    
    def decode_hash(self, binary_hash):
        """
        Decode a binary hash using Redis Lua script
        
        Args:
            binary_hash: The binary hash to decode
            
        Returns:
            List of peaks
        """
        result = self.redis.evalsha(
            self.decode_script_sha,
            0,  # No keys used in script
            binary_hash
        )
        
        return json.loads(result)

# Example usage
if __name__ == "__main__":
    # Initialize with Redis connection
    redis_hash = RedisAudioHashSearch(redis_host='localhost', redis_port=6379)
    
    # Store a reference audio file with a value
    key = redis_hash.store_hash('example.mp3', value='{"id": 123, "title": "Example Song"}')
    print(f"Stored reference audio under key: {key}")
    
    # Search for similar audio
    matches = redis_hash.find_similar('query.mp3', threshold=0.8)
    for match in matches:
        print(f"Match found: {match['key']} with similarity {match['similarity']}")
        print(f"Value: {match['value']}") 