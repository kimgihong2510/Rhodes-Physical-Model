
import math
from collections import defaultdict

import mido
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from rhodes_synth import RhodesSingleKey

AUDIO_SAMPLE_RATE = 48000
SIM_SAMPLE_RATE = AUDIO_SAMPLE_RATE * 8

MIDI_FILE_NAME = "just_the_two_of_us"

midi = mido.MidiFile(MIDI_FILE_NAME+".mid")

# Group note events by pitch.
events_by_note = defaultdict(list)
current_seconds = 0.0
for msg in midi:
    current_seconds += msg.time
    if msg.type in ("note_on", "note_off"):
        sample_index = round(current_seconds * SIM_SAMPLE_RATE)
        events_by_note[msg.note].append((sample_index, msg))

# Render each pitch independently and add its signals to the shared buffers.
duration_seconds = midi.length
duration_samples = math.ceil(duration_seconds * SIM_SAMPLE_RATE)
pickup_audio = np.zeros(duration_samples)
tine_displacement = np.zeros(duration_samples)

for note, note_events in tqdm(
    sorted(events_by_note.items()), 
    total=len(events_by_note.keys())
):   
    synth = RhodesSingleKey(SIM_SAMPLE_RATE, note)

    def render_block(start, end):
        tine_block = np.zeros(end - start)
        pickup_block = np.zeros(end - start)
        for t in range(0, end-start):
            tine_sample, pickup_sample = synth.render_sample()
            tine_block[t] = tine_sample
            pickup_block[t] = pickup_sample
        return tine_block, pickup_block
    
    # Track the next sample to render for this pitch.
    current_sample = 0
    while len(note_events) > 0:
        assert note_events[0][0] >= 0
        if current_sample < note_events[0][0]:
            # Advance the model up to the next event sample.
            next_event_sample = note_events[0][0]

            tine_block, pickup_block = render_block(current_sample, next_event_sample)
            tine_displacement[current_sample:next_event_sample] += tine_block
            pickup_audio[current_sample:next_event_sample] += pickup_block
            current_sample = next_event_sample

        # Apply the midi event
        msg = note_events[0][1]
        note_events.pop(0)
        if msg.type == "note_on":
            velocity = msg.velocity
            assert velocity >= 0.0
            # MIDI note-on messages with zero velocity also release the key.
            if velocity == 0.0:
                synth.note_off()
            elif velocity > 0.0:
                synth.note_on(velocity)
        elif msg.type == "note_off":
            synth.note_off()
    # Continue after the last event until the MIDI file ends.
    tine_block, pickup_block = render_block(current_sample, duration_samples)
    tine_displacement[current_sample:duration_samples] += tine_block
    pickup_audio[current_sample:duration_samples] += pickup_block

def save_audio(audio, file_name):
    audio = librosa.resample(
        audio,
        orig_sr=SIM_SAMPLE_RATE,
        target_sr=AUDIO_SAMPLE_RATE,
    )

    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.8

    sf.write(f"{file_name}.wav", audio, AUDIO_SAMPLE_RATE, subtype="FLOAT")

save_audio(pickup_audio, MIDI_FILE_NAME+"_pickup")
