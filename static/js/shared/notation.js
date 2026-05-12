import Vex from 'https://esm.sh/vexflow';

const STAVE_HEIGHT = 140;
const STAVE_PADDING = 40;
const REST_PLACEHOLDER_KEY = 'b/4';
const DEFAULT_WIDTH = 400;
const STAVE_X_OFFSET = 20;
const STAVE_Y_OFFSET = 20;
const DEFAULT_NUM_BEATS = 4;
const DEFAULT_BEAT_VALUE = 4;

/**
 * @param {Object} noteData - { keys: string[], duration: string, dots: number }
 * @param {string} clef
 * @returns {Vex.Flow.StaveNote}
 */
function buildStaveNote(noteData, clef) {
    const isRest = noteData.duration.endsWith('r');
    const keys = isRest ? [REST_PLACEHOLDER_KEY] : noteData.keys;
    const note = new Vex.Flow.StaveNote({ clef, keys, duration: noteData.duration });
    if (noteData.dots > 0) {
        Vex.Flow.Dot.buildAndAttach([note], { all: true });
    }
    return note;
}

/**
 * @param {Object} voiceData - { notes: Array }
 * @param {Vex.Flow.Stave} stave
 * @param {string} clef
 * @param {Vex.Flow.RenderContext} context
 */
function drawVoice(voiceData, stave, clef, context) {
    const staveWidth = stave.getWidth();
    const vfNotes = voiceData.notes.map(noteData => buildStaveNote(noteData, clef));
    const voice = new Vex.Flow.Voice({ num_beats: DEFAULT_NUM_BEATS, beat_value: DEFAULT_BEAT_VALUE }).setStrict(false);
    voice.addTickables(vfNotes);
    new Vex.Flow.Formatter().joinVoices([voice]).format([voice], staveWidth - STAVE_PADDING);
    voice.draw(context, stave);
}

/**
 * @param {Object} staveData - { clef: string, time_signature: [number, number]|null, voices: Array }
 * @param {Vex.Flow.RenderContext} context
 * @param {number} x
 * @param {number} y
 * @param {number} width
 */
function drawStave(staveData, context, x, y, width) {
    const stave = new Vex.Flow.Stave(x, y, width);
    stave.addClef(staveData.clef);
    if (staveData.time_signature) {
        stave.addTimeSignature(`${staveData.time_signature[0]}/${staveData.time_signature[1]}`);
    }
    stave.setContext(context).draw();
    for (const voice of staveData.voices) {
        drawVoice(voice, stave, staveData.clef, context);
    }
}

/**
 * Renders a ScoreData object as an SVG into containerElement.
 * @param {Object} scoreData - { staves: Array, tempo: number|null }
 * @param {HTMLElement} containerElement
 */
export function renderScore(scoreData, containerElement) {
    containerElement.innerHTML = '';
    const width = containerElement.clientWidth || DEFAULT_WIDTH;
    const totalHeight = scoreData.staves.length * STAVE_HEIGHT;
    const renderer = new Vex.Flow.Renderer(containerElement, Vex.Flow.Renderer.Backends.SVG);
    renderer.resize(width, totalHeight);
    const context = renderer.getContext();
    const staveWidth = width - STAVE_PADDING;
    scoreData.staves.forEach((staveData, index) => {
        drawStave(staveData, context, STAVE_X_OFFSET, index * STAVE_HEIGHT + STAVE_Y_OFFSET, staveWidth);
    });
}
