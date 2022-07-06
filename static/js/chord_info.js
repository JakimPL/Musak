class ChordInfo {
    constructor() {
        this.reset();
    }

    reset() {
        this.type = null;
        this.type_name = null;
        this.inversion = -1;
        this.inversions_numbers = {};
        this.names = {
          '': 'major',
          'm': 'minor',
          'dim': 'diminished',
          'aug': 'augmented',
          'sus4': 'suspended (4)',
          '7': 'dominant',
          'maj7': 'major seventh',
          'm7': 'minor seventh',
          'm(maj7)': 'minor major seventh'
        };
    }
}
