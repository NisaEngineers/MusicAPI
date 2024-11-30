import os
from spleeter.separator import Separator
import autochord
import pretty_midi
import librosa
import matchering as mg
from pedalboard import Pedalboard, HighpassFilter, Compressor, Limiter, Reverb, Gain
from pedalboard.io import AudioFile
import numpy as np
from scipy.signal import butter, lfilter

class AudioSplitter:
    def __init__(self, stems=5):
        self.stems = stems

    def separate_audio(self, input_path, output_path):
        separator = Separator(f'spleeter:{self.stems}stems')
        os.makedirs(output_path, exist_ok=True)
        separator.separate_to_file(input_path, output_path)
        return {
            "vocals": os.path.join(output_path, 'vocals.wav'),
            "accompaniment": os.path.join(output_path, 'other.wav'),
            "bass": os.path.join(output_path, 'bass.wav'),
            "drums": os.path.join(output_path, 'drums.wav'),
            "piano": os.path.join(output_path, 'piano.wav')
        }

class ChordRecognizer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.chords = None
        self.midi_chords = pretty_midi.PrettyMIDI()
        self.instrument_chords = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    def recognize_chords(self):
        self.chords = autochord.recognize(self.audio_file, lab_fn='chords.lab')

    def chord_to_midi_notes(self, chord_name):
        note_mapping = {
            'C:maj': ['C4', 'E4', 'G4'],
            'C:min': ['C4', 'E-4', 'G4'],
            'D:maj': ['D4', 'F#4', 'A4'],
            'D:min': ['D4', 'F4', 'A4'],
            'E:maj': ['E4', 'G#4', 'B4'],
            'E:min': ['E4', 'G4', 'B4'],
            'F:maj': ['F4', 'A4', 'C5'],
            'F:min': ['F4', 'A-4', 'C5'],
            'G:maj': ['G4', 'B4', 'D5'],
            'G:min': ['G4', 'B-4', 'D5'],
            'A:maj': ['A4', 'C#5', 'E5'],
            'A:min': ['A4', 'C5', 'E5'],
            'B:maj': ['B4', 'D#5', 'F#5'],
            'B:min': ['B4', 'D5', 'F#5']
        }
        return note_mapping.get(chord_name, [])

    def generate_midi(self):
        for chord in self.chords:
            start_time = chord[0]
            end_time = chord[1]
            chord_name = chord[2]
            if chord_name != 'N':
                chord_notes = self.chord_to_midi_notes(chord_name)
                for note_name in chord_notes:
                    midi_note = pretty_midi.Note(
                        velocity=100,
                        pitch=librosa.note_to_midi(note_name),
                        start=start_time,
                        end=end_time
                    )
                    self.instrument_chords.notes.append(midi_note)
        self.midi_chords.instruments.append(self.instrument_chords)

    def save_midi(self, output_file):
        self.midi_chords.write(output_file)
        return output_file

class AudioMastering:
    def __init__(self):
        pass

    def master_audio(self, input_path, reference_path, output_path):
        mg.log(warning_handler=print)
        mg.process(
            target=input_path,
            reference=reference_path,
            results=[mg.pcm16(output_path)],
            preview_target=mg.pcm16("preview_target.flac"),
            preview_result=mg.pcm16("preview_result.flac"),
        )

    def process_audio_with_pedalboard(self, input_path, output_path):
        with AudioFile(input_path) as f:
            audio = f.read(f.frames)
            sample_rate = f.samplerate

        def stereo_widen(audio, width=1.2):
            left_channel = audio[0::2] * width
            right_channel = audio[1::2] * width
            widened_audio = np.empty_like(audio)
            widened_audio[0::2] = left_channel
            widened_audio[1::2] = right_channel
            return widened_audio

        def reduce_piano_volume(audio, sample_rate, freq_low=200, freq_high=2000, reduction_db=-18):
            nyquist = 0.5 * sample_rate
            low = freq_low / nyquist
            high = freq_high / nyquist
            b, a = butter(1, [low, high], btype='band')
            filtered_audio = lfilter(b, a, audio)
            gain_reduction = 10 ** (reduction_db / 20)
            reduced_audio = audio - (filtered_audio * gain_reduction)
            return reduced_audio

        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=100),
            Compressor(threshold_db=-20, ratio=4),
            Limiter(threshold_db=-0.1),
            Reverb(room_size=0.3, wet_level=0.2),
            Gain(gain_db=3),
        ])
        processed_audio = board(audio, sample_rate)
        processed_audio = stereo_widen(processed_audio)
        processed_audio = reduce_piano_volume(processed_audio, sample_rate)
        with AudioFile(output_path, 'w', sample_rate, processed_audio.shape[0]) as f:
            f.write(processed_audio)

# Example usage of the classes
if __name__ == "__main__":
    input_path = 'path/to/your/audio/file.mp3'
    output_base_path = 'output'
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_base_path, base_name)
    os.makedirs(output_path, exist_ok=True)

    # Split audio
    splitter = AudioSplitter()
    separated_files = splitter.separate_audio(input_path, output_path)

    # Recognize chords
    chord_recognizer = ChordRecognizer(separated_files['piano'])
    chord_recognizer.recognize_chords()
    midi_output_file = os.path.join(output_path, f'{base_name}_chords.mid')
    chord_recognizer.generate_midi()
    chord_recognizer.save_midi(midi_output_file)

    # Master audio
    mastering = AudioMastering()
    master_audio_path = os.path.join(output_path, f'{base_name}_master.wav')
    mastering.master_audio(separated_files['piano'], input_path, master_audio_path)

    # Apply pedalboard effects
    final_output_path = os.path.join(output_path, f'{base_name}_final.wav')
    mastering.process_audio_with_pedalboard(master_audio_path, final_output_path)

    print("Processing completed.")
