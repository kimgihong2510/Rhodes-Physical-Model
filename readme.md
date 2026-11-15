# Rhodes Piano Physical Model

A Python implementation of a Rhodes-style electric piano, including tine vibration, hammer contact, damping, and a magnetic pickup model.

## Demo
### 🎧 [Listen to Audio Demo](https://kimgihong2510.github.io/Rhodes-Physical-Model/demo/).

## References

- Tobias Sønderbo, *Real-time physical model of the Rhodes Electric Piano*, master's thesis, Aalborg University, 2024. [Thesis PDF](https://projekter.aau.dk/projekter/files/719175589/Master_Thesis_Real_time_physical_model_of_the_Rhodes_Electric_Piano_Tobias_Sonderbo.pdf). Basis for the finite-difference tine model, hammer contact, and damper implementation in `rhodes_synth.py`.
- I. Nishiguchi, H. Kaimori, A. Kameari, E. Takebuchi, and Y. Nojima, “Modeling of Magnetic Pickups for Electric Musical Instruments,” COMPUMAG, 2019. [DOI: 10.1109/COMPUMAG45669.2019.9032828](https://doi.org/10.1109/COMPUMAG45669.2019.9032828). Basis for the pickup lookup-table calculation in `generate_flux_table.py`.

## Implementation

- [`generate_flux_table.py`](generate_flux_table.py): Numerically integrates the pickup model to generate a lookup table mapping tine-tip displacement to unscaled magnetic flux. Saves the table to `flux_table.npz`.
- [`rhodes_synth.py`](rhodes_synth.py): Implements a single-key physical model of the tine, hammer, and damper. Uses the flux lookup table to calculate the pickup's induced EMF.
- [`render_midi.py`](render_midi.py): Reads a MIDI file, renders each pitch, and mixes the results. Saves the pickup output as a peak-normalized WAV file resampled to 48 kHz.

## Included Assets

- `flux_table.npz`: A precomputed flux lookup table for the configured pickup geometry.

Two demo MIDI files are included for testing:

- “Just the Two of Us” — sourced from [Online Sequencer](https://onlinesequencer.net/4897971).
- “What’d I Say” — sourced from [MIDI World](https://www.midiworld.com/download/792).

## TODO

- Extend the tine model from single-direction transverse vibration to planar vibration.
