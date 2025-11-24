import numpy as np
import soundfile as sf
import argparse
import os
import tempfile

# Effekt-Definitionen
AVAILABLE_EFFECTS = {
    'tremolo': {
        'description': 'Amplitude modulation',
        'params': {'rate': (2, 8), 'depth': (0.3, 0.8)}
    },
    'vibrato': {
        'description': 'Frequency modulation',
        'params': {'rate': (3, 7), 'depth': (0.02, 0.1)}
    },
    'phaser': {
        'description': 'Sweeping frequency notches',
        'params': {'rate': (0.5, 2), 'depth': (0.3, 0.7)}
    },
    'distortion': {
        'description': 'Harmonic distortion',
        'params': {'amount': (1.5, 4)}
    },
    'chorus': {
        'description': 'Multiple delayed copies',
        'params': {'voices': (2, 4), 'depth': (0.01, 0.03)}
    },
    'stutter': {
        'description': 'Rhythmic repeats',
        'params': {'rate': (4, 16), 'probability': (0.1, 0.4)}
    },
    'filter_sweep': {
        'description': 'Moving filter cutoff',
        'params': {'rate': (0.3, 1.5), 'range': (0.2, 0.8)}
    },
    'bitcrush': {
        'description': 'Reduce bit depth',
        'params': {'bits': (4, 10)}
    }
}

COMPOSITION_PARTS = [
    {
        'code': 'T',
        'name': "Tease",
        'description': "A slight tease to get you worked up.",
        'frequencies': (0.6, 1.0),
        'amplitude': (-6, -2, 3),
        'continuity': (0.5, 0.7),  # Range statt fester Wert
        'waveform': ['sine', 'triangle'],  # Mehrere mögliche Waveforms
        'randomness': (0.6, 0.8),  # Range für mehr Variation
        'possible_effects': ['tremolo', 'vibrato', 'chorus'],
        'effect_probability': 0.6,
        'max_effects': 2,
        'jumping': {'enabled': True, 'probability': (0.2, 0.4)}  # Range
    },
    {
        'code': 'G',
        'name': "Gentle",
        'description': "A gentle, soothing pattern.",
        'frequencies': (0.5, 0.9),
        'amplitude': (-4, -1, 2),
        'continuity': (0.7, 0.9),
        'waveform': ['sine'],
        'randomness': (0.2, 0.4),
        'possible_effects': ['tremolo', 'chorus', 'phaser'],
        'effect_probability': 0.4,
        'max_effects': 1,
        'jumping': {'enabled': False, 'probability': (0.0, 0.0)}
    },
    {
        'code': 'H',
        'name': "Hard",
        'description': "A strong, intense pattern.",
        'frequencies': (0.4, 0.7),
        'amplitude': (-3, -1, 1),
        'continuity': (0.7, 0.9),
        'waveform': ['sine', 'sawtooth'],
        'randomness': (0.3, 0.5),
        'possible_effects': ['distortion', 'tremolo', 'filter_sweep'],
        'effect_probability': 0.7,
        'max_effects': 2,
        'jumping': {'enabled': True, 'probability': (0.3, 0.5)}
    },
    {
        'code': 'P',
        'name': "Spicy",
        'description': "A pattern on the verge between pain and pleasure.",
        'frequencies': (0.35, 0.6),
        'amplitude': (-3, -1, 2),
        'continuity': (0.4, 0.6),
        'waveform': ['sawtooth', 'triangle'],
        'randomness': (0.7, 0.9),
        'possible_effects': ['distortion', 'stutter', 'bitcrush', 'phaser'],
        'effect_probability': 0.8,
        'max_effects': 3,
        'jumping': {'enabled': True, 'probability': (0.5, 0.7)}
    },
    {
        'code': 'C',
        'name': "Chaos",
        'description': "Unpredictable.",
        'frequencies': (0.1, 0.9),
        'amplitude': (-6, -2, 5),
        'continuity': (0.5, 0.7),
        'waveform': ['sawtooth', 'triangle'],
        'randomness': (0.9, 1.0),
        'possible_effects': ['stutter', 'bitcrush', 'distortion', 'vibrato', 'filter_sweep'],
        'effect_probability': 0.9,
        'max_effects': 4,
        'jumping': {'enabled': True, 'probability': (0.7, 0.9)}
    },
    {
        'code': 'S',
        'name': "Shock",
        'description': "For masochists, only.",
        'frequencies': (0.15, 0.3),
        'amplitude': (-2, 0, 2),
        'continuity': (0.01, 0.03),
        'waveform': ['sawtooth'],
        'randomness': (0.7, 0.9),
        'possible_effects': ['bitcrush', 'distortion', 'stutter'],
        'effect_probability': 0.5,
        'max_effects': 2,
        'jumping': {'enabled': True, 'probability': (0.4, 0.6)}
    },
    {
        'code': 'X',
        'name': "Extreme Shock",
        'description': "At this point for sadistic dommes only.",
        'frequencies': (0.0, 0.25),
        'amplitude': (-1, 0, 1),
        'continuity': (0.04, 0.08),
        'waveform': ['sawtooth'],
        'randomness': (0.5, 0.7),
        'possible_effects': ['bitcrush', 'distortion'],
        'effect_probability': 0.4,
        'max_effects': 2,
        'jumping': {'enabled': True, 'probability': (0.6, 0.8)}
    },
    {
        'code': 'O',
        'name': "Orgasm",
        'description': "The peak of pleasure.",
        'frequencies': (0.4, 1.0),
        'amplitude': (-2, -1, 2),
        'continuity': (0.95, 1.0),
        'waveform': ['triangle', 'sine'],
        'randomness': (0.5, 0.7),
        'possible_effects': ['tremolo', 'vibrato', 'chorus', 'phaser'],
        'effect_probability': 0.9,
        'max_effects': 3,
        'jumping': {'enabled': True, 'probability': (0.3, 0.5)}
    },
]

PARTS_MAP = {part['code']: part for part in COMPOSITION_PARTS}

def get_parts_help_text():
    lines = ["Available composition codes:"]
    for part in COMPOSITION_PARTS:
        lines.append(f"  {part['code']} - {part['name']}: {part['description']}")
    return "\n".join(lines)

def db_to_amplitude(db):
    return 10 ** (db / 20)

def generate_waveform(waveform_type, frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    if waveform_type == 'sine':
        return np.sin(2 * np.pi * frequency * t)
    elif waveform_type == 'sawtooth':
        return 2 * (t * frequency - np.floor(t * frequency + 0.5))
    elif waveform_type == 'triangle':
        return 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1
    else:
        raise ValueError(f"Unknown waveform type: {waveform_type}")

def apply_hardcut_filter(audio, sample_rate, freq_min, freq_max):
    """
    Schneidet alle Frequenzen unter freq_min und über freq_max ab.
    Verwendet FFT für präzises Frequenz-Cutting.
    """
    # FFT durchführen
    fft = np.fft.rfft(audio)
    frequencies = np.fft.rfftfreq(len(audio), 1/sample_rate)
    
    # Maske erstellen: nur Frequenzen innerhalb der Range behalten
    mask = (frequencies >= freq_min) & (frequencies <= freq_max)
    
    # Frequenzen außerhalb der Range auf 0 setzen
    fft_filtered = fft * mask
    
    # Zurück in Zeit-Domain
    audio_filtered = np.fft.irfft(fft_filtered, n=len(audio))
    
    return audio_filtered

def apply_tremolo(audio, sample_rate, rate, depth):
    """Amplitude modulation"""
    t = np.linspace(0, len(audio) / sample_rate, len(audio))
    modulator = 1 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))
    return audio * modulator

def apply_vibrato(audio, sample_rate, rate, depth):
    """Frequency modulation durch variable Verzögerung"""
    t = np.linspace(0, len(audio) / sample_rate, len(audio))
    delay_samples = depth * sample_rate * np.sin(2 * np.pi * rate * t)
    
    output = np.zeros_like(audio)
    for i in range(len(audio)):
        delay_idx = i + int(delay_samples[i])
        if 0 <= delay_idx < len(audio):
            output[i] = audio[delay_idx]
    return output

def apply_phaser(audio, sample_rate, rate, depth):
    """Phaser-Effekt mit beweglichem Notch-Filter"""
    t = np.linspace(0, len(audio) / sample_rate, len(audio))
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * rate * t)
    
    delay = (1 + depth * lfo) * 0.001 * sample_rate
    output = audio.copy()
    
    for i in range(len(audio)):
        delay_idx = i - int(delay[i])
        if delay_idx >= 0:
            output[i] = audio[i] + 0.5 * audio[delay_idx]
    
    return output * 0.7

def apply_distortion(audio, amount):
    """Soft-Clipping Distortion"""
    return np.tanh(audio * amount) / np.tanh(amount)

def apply_chorus(audio, sample_rate, voices, depth):
    """Chorus durch mehrere verzögerte Kopien"""
    output = audio.copy()
    
    for v in range(int(voices)):
        delay_ms = (v + 1) * depth * 1000
        delay_samples = int(delay_ms * sample_rate / 1000)
        
        if delay_samples < len(audio):
            delayed = np.pad(audio[:-delay_samples], (delay_samples, 0), 'constant')
            output += delayed * (0.5 / voices)
    
    return output / (1 + voices * 0.3)

def apply_stutter(audio, sample_rate, rate, probability):
    """Rhythmisches Stottern"""
    chunk_size = int(sample_rate / rate)
    output = audio.copy()
    
    for i in range(0, len(audio) - chunk_size, chunk_size):
        if np.random.random() < probability:
            repeat_count = np.random.randint(2, 5)
            chunk = audio[i:i + chunk_size // repeat_count]
            for r in range(repeat_count):
                start = i + r * len(chunk)
                end = start + len(chunk)
                if end <= len(output):
                    output[start:end] = chunk
    
    return output

def apply_filter_sweep(audio, sample_rate, rate, range_param):
    """Beweglicher Low-Pass-Filter"""
    t = np.linspace(0, len(audio) / sample_rate, len(audio))
    cutoff = 200 + (2000 - 200) * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t)) * range_param
    
    output = audio.copy()
    
    for i in range(len(audio)):
        w = max(3, int(sample_rate / cutoff[i] * 2))
        if w % 2 == 0:
            w += 1
        start = max(0, i - w // 2)
        end = min(len(audio), i + w // 2 + 1)
        output[i] = np.mean(audio[start:end])
    
    return output

def apply_bitcrush(audio, bits):
    """Bit-Depth Reduktion"""
    levels = 2 ** int(bits)
    normalized = (audio + 1) / 2
    crushed = np.round(normalized * levels) / levels
    return crushed * 2 - 1

def apply_effects(audio, sample_rate, effects_config):
    """Wendet eine Liste von Effekten an"""
    output = audio.copy()
    
    for effect_name, params in effects_config:
        if effect_name == 'tremolo':
            output = apply_tremolo(output, sample_rate, params['rate'], params['depth'])
        elif effect_name == 'vibrato':
            output = apply_vibrato(output, sample_rate, params['rate'], params['depth'])
        elif effect_name == 'phaser':
            output = apply_phaser(output, sample_rate, params['rate'], params['depth'])
        elif effect_name == 'distortion':
            output = apply_distortion(output, params['amount'])
        elif effect_name == 'chorus':
            output = apply_chorus(output, sample_rate, params['voices'], params['depth'])
        elif effect_name == 'stutter':
            output = apply_stutter(output, sample_rate, params['rate'], params['probability'])
        elif effect_name == 'filter_sweep':
            output = apply_filter_sweep(output, sample_rate, params['rate'], params['range'])
        elif effect_name == 'bitcrush':
            output = apply_bitcrush(output, params['bits'])
    
    return output

def select_random_effects(part):
    """Wählt zufällig Effekte für einen Part aus"""
    if np.random.random() > part['effect_probability']:
        return []
    
    available = part['possible_effects']
    num_effects = np.random.randint(1, min(part['max_effects'], len(available)) + 1)
    
    selected_effects = []
    chosen = np.random.choice(available, size=num_effects, replace=False)
    
    for effect_name in chosen:
        effect_def = AVAILABLE_EFFECTS[effect_name]
        params = {}
        
        for param_name, (min_val, max_val) in effect_def['params'].items():
            params[param_name] = np.random.uniform(min_val, max_val)
        
        selected_effects.append((effect_name, params))
    
    return selected_effects

def get_random_value(param):
    """Holt einen zufälligen Wert aus Range oder gibt festen Wert zurück"""
    if isinstance(param, tuple) and len(param) == 2:
        return np.random.uniform(param[0], param[1])
    return param

def select_random_waveform(waveforms):
    """Wählt zufällig eine Waveform aus der Liste"""
    if isinstance(waveforms, list):
        return np.random.choice(waveforms)
    return waveforms

def generate_segment(part, duration, sample_rate, freq_min, freq_max):
    """Generiert ein Audio-Segment mit zufälligen Effekten und optionalem Frequency-Jumping"""
    freq_range = part['frequencies']
    freq_low = freq_min + (freq_max - freq_min) * freq_range[0]
    freq_high = freq_min + (freq_max - freq_min) * freq_range[1]
    
    amp_base, amp_sub, amp_add = part['amplitude']
    amp_min_db = amp_base - amp_sub
    amp_max_db = amp_base + amp_add
    amp_min = db_to_amplitude(amp_min_db)
    amp_max = db_to_amplitude(amp_max_db)
    
    n_samples = int(sample_rate * duration)
    audio = np.zeros(n_samples)
    
    # Randomisierte Parameter pro Segment
    continuity = get_random_value(part['continuity'])
    randomness = get_random_value(part['randomness'])
    waveform = select_random_waveform(part['waveform'])
    jumping = part['jumping'].copy()
    if jumping['enabled']:
        jumping['probability'] = get_random_value(jumping['probability'])
    
    # Zusätzliche Chunk-Duration-Variation
    base_chunk_duration = np.random.uniform(0.08, 0.12)  # Variiert zwischen 80-120ms
    chunk_duration = base_chunk_duration * (1.0 - randomness * 0.7)
    chunk_duration = max(0.02, chunk_duration)
    
    n_chunks = int(duration / chunk_duration)
    chunk_samples = int(sample_rate * chunk_duration)
    
    target_freq = np.random.uniform(freq_low, freq_high)
    target_amp = np.random.uniform(amp_min, amp_max)
    current_freq = target_freq
    current_amp = target_amp
    
    for i in range(n_chunks):
        if np.random.random() < continuity:
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples, n_samples)
            chunk_len = (end_idx - start_idx) / sample_rate
            
            # Frequency Jumping Logic
            should_jump = jumping['enabled'] and np.random.random() < jumping['probability']
            
            if should_jump:
                # Abrupter Sprung zu einer neuen Frequenz innerhalb der Range
                current_freq = np.random.uniform(freq_low, freq_high)
                current_amp = np.random.uniform(amp_min, amp_max)
            else:
                # Normale sanfte Änderungen
                if np.random.random() < randomness:
                    target_freq = np.random.uniform(freq_low, freq_high)
                    target_amp = np.random.uniform(amp_min, amp_max)
                
                smoothness = 1.0 - randomness * 0.7
                current_freq = current_freq * smoothness + target_freq * (1 - smoothness)
                current_amp = current_amp * smoothness + target_amp * (1 - smoothness)
            
            wave = generate_waveform(waveform, current_freq, chunk_len, sample_rate)
            
            # Sicherstellen, dass die Länge exakt passt
            expected_len = end_idx - start_idx
            if len(wave) > expected_len:
                wave = wave[:expected_len]
            elif len(wave) < expected_len:
                wave = np.pad(wave, (0, expected_len - len(wave)), 'constant')
            
            audio[start_idx:end_idx] = wave * current_amp
            
            # Kürzere Fades bei Jumps für abrupteren Charakter
            fade_samples = int(0.002 * sample_rate) if should_jump else int(0.005 * sample_rate)
            fade_samples = min(fade_samples, len(wave) // 4)
            
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                audio[start_idx:start_idx + fade_samples] *= fade_in
                audio[end_idx - fade_samples:end_idx] *= fade_out
    
    # Hardcut-Filter anwenden: Frequenzen außerhalb der globalen Range abschneiden
    audio = apply_hardcut_filter(audio, sample_rate, freq_min, freq_max)
    
    # Zufällige Effekte anwenden
    effects = select_random_effects(part)
    if effects:
        print(f"  Applying effects: {[e[0] for e in effects]}")
        audio = apply_effects(audio, sample_rate, effects)
    
    return audio

def crossfade(audio1, audio2, fade_duration, sample_rate):
    fade_samples = int(fade_duration * sample_rate)
    fade_samples = min(fade_samples, len(audio1), len(audio2))
    
    if fade_samples == 0:
        return np.concatenate([audio1, audio2])
    
    fade_out = np.linspace(1, 0, fade_samples)
    audio1[-fade_samples:] *= fade_out
    
    fade_in = np.linspace(0, 1, fade_samples)
    audio2[:fade_samples] *= fade_in
    
    audio1[-fade_samples:] += audio2[:fade_samples]
    
    return np.concatenate([audio1, audio2[fade_samples:]])

def generate_composition(pattern, total_duration, sample_rate, freq_min, freq_max, crossfade_duration=0.5, num_layers=1):
    parts = []
    for code in pattern.upper():
        if code not in PARTS_MAP:
            available = ', '.join(PARTS_MAP.keys())
            raise ValueError(f"Unknown part code: {code}. Available codes: {available}")
        parts.append(PARTS_MAP[code])
    
    segment_duration = total_duration / len(parts)
    
    print(f"Generating {num_layers} layer(s) with pattern '{pattern}'...")
    print(f"Each layer: {len(parts)} segments, each {segment_duration:.2f}s long")
    print(f"Applying hardcut filter: {freq_min}-{freq_max} Hz")
    
    # Liste für alle Layer
    layers = []
    
    for layer_num in range(num_layers):
        print(f"\n=== Generating Layer {layer_num + 1}/{num_layers} ===")
        
        # Erstes Segment
        audio = generate_segment(parts[0], segment_duration, sample_rate, freq_min, freq_max)
        jump_info = "with jumping" if parts[0]['jumping']['enabled'] else "no jumping"
        print(f"Segment 1/{len(parts)}: {parts[0]['name']} ({jump_info})")
        
        # Weitere Segmente
        for i, part in enumerate(parts[1:], start=2):
            segment = generate_segment(part, segment_duration, sample_rate, freq_min, freq_max)
            audio = crossfade(audio, segment, crossfade_duration, sample_rate)
            jump_info = f"jumping: {part['jumping']['probability']}" if part['jumping']['enabled'] else "no jumping"
            print(f"Segment {i}/{len(parts)}: {part['name']} ({jump_info})")
        
        layers.append(audio)
    
    # Layer zusammenführen
    if num_layers > 1:
        print(f"\n=== Merging {num_layers} layers ===")
        # Alle auf gleiche Länge bringen (falls minimal unterschiedlich)
        max_length = max(len(layer) for layer in layers)
        layers_padded = [np.pad(layer, (0, max_length - len(layer)), 'constant') for layer in layers]
        
        # Mit gleicher Gewichtung zusammenmischen
        weight = 1.0 / num_layers
        final_audio = np.sum([layer * weight for layer in layers_padded], axis=0)
        print(f"Each layer weighted at {weight:.1%}")
    else:
        final_audio = layers[0]
    
    # Normalisieren
    max_val = np.max(np.abs(final_audio))
    if max_val > 0:
        final_audio = final_audio / max_val
    
    return final_audio

def save_as_mp3(audio, sample_rate, output_file):
    try:
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            sf.write(temp_wav.name, audio, sample_rate, format='WAV')
            
            import subprocess
            try:
                subprocess.run(
                    ['ffmpeg', '-y', '-i', temp_wav.name, '-codec:a', 'libmp3lame', 
                     '-q:a', '2', output_file],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print(f"MP3 saved to: {output_file}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                output_wav = output_file.replace('.mp3', '.wav')
                sf.write(output_wav, audio, sample_rate)
                print(f"WAV saved to: {output_wav}")
    except Exception as e:
        print(f"Error saving audio: {e}")
    finally:
        if 'temp_wav' in locals() and os.path.exists(temp_wav.name):
            try:
                os.unlink(temp_wav.name)
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(
        description="Generate audio patterns with random effects and hardcut filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=get_parts_help_text()
    )
    
    parser.add_argument('pattern', type=str, help='Pattern string (e.g., TGHPO)')
    parser.add_argument('duration', type=int, help='Total duration in seconds')
    parser.add_argument('--freq-range', type=int, nargs=2, default=[500, 1500],
                        metavar=('MIN', 'MAX'), help='Frequency range in Hz (hardcut applied)')
    parser.add_argument('-o', '--output', type=str, default='output.mp3',
                        help='Output file')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Sample rate in Hz')
    parser.add_argument('--crossfade', type=float, default=1.0,
                        help='Crossfade duration')
    parser.add_argument('--layers', type=int, default=1,
                        help='Number of audio layers to generate and merge (default: 1)')
    
    args = parser.parse_args()
    
    try:
        print(f"Generating pattern '{args.pattern}' for {args.duration}s...")
        print(f"Frequency range: {args.freq_range[0]}-{args.freq_range[1]} Hz")
        print(f"Layers: {args.layers}")
        
        audio = generate_composition(
            args.pattern,
            args.duration,
            args.sample_rate,
            args.freq_range[0],
            args.freq_range[1],
            args.crossfade,
            args.layers
        )
        
        save_as_mp3(audio, args.sample_rate, "./stimfiles/" + args.output)
        print("Done!")
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())