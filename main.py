from audioHash import AudioHash
from redisIntegration import RedisAudioHashSearch
import time

audio_path = "./audio/h_audio1.mp3"
audio_path2 = "./audio/h_audio_cut2.wav"
audio_path3 = "./audio/rec_1.mp3"

def test_python():

    ahash = AudioHash()

    start_peak = time.time()
    peaks = ahash.create_time_sensitive_hash(audio_path,visualize=True)
    end_peak = time.time()

    start_encode = time.time()
    binary_hash = ahash.encode_peaks(peaks)
    end_encode = time.time()

    start_decode = time.time()
    decoded_peak = ahash.decode_peaks(binary_hash)
    end_decode = time.time()

    hash2 = ahash.encode_peaks(ahash.create_time_sensitive_hash(audio_path2))
    start_compare = time.time()
    similarity = ahash.compare_hungarian(binary_hash, hash2)
    end_compare = time.time()

    peak_time =  end_peak - start_peak
    encode_time = end_encode - start_encode
    decode_time = end_decode - start_decode 
    compare_time = end_compare  - start_compare


    print(f"Peak generation took {peak_time:.5f} seconds")
    print(f"Encoding took {encode_time:.5f} seconds")
    print(f"Decoding took {decode_time:.5f} seconds")
    print(f"Comparing took {compare_time:.5f} seconds")

    print(similarity)

    ahash.compare_visually(binary_hash, hash2)

def test_redis():
    # Initialize with Redis connection
    redis_hash = RedisAudioHashSearch(redis_host='localhost', redis_port=6379)

    start_set = time.time()
    # Store audio fingerprints with values
    for i in range(1000):
        value = f'{{"id": {i}, "title": "Rec"}}'
        key = redis_hash.store_hash(audio_path3, value=value, hash_id=str(i+1))
        if i==500:
            value = '{"id": 1000, "title": "Match song"}'
            key = redis_hash.store_hash(audio_path, value=value, hash_id=str(1000))
    
    end_set = time.time()
    # key2 = redis_hash.store_hash(audio_path2, value='{"id": 2, "title": "Song Two"}')

    
    # Find similar audio and get the values
    start_find = time.time()
    matches = redis_hash.find_similar(audio_path2, threshold=0.2, max_results=1)
    end_find = time.time()
    for match in matches:
        print(f"Match found with similarity: {match['similarity']:.2f}")
        print(f"Value: {match['value']}")

    print("Set Time:" , end_set - start_set)
    print("Find Time:", end_find - start_find)
    # # You can also store hash directly if you already have it
    # peaks = redis_hash.audio_hash.create_time_sensitive_hash('song3.mp3')
    # binary_hash = redis_hash.audio_hash.encode_peaks(peaks)
    # key3 = redis_hash.store_binary_hash(binary_hash, value='{"id": 3, "title": "Song Three"}')

if __name__ == "__main__":
    test_python()