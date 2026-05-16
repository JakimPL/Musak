import { Renderer, Stave, StaveNote, Voice, Formatter, Dot, Beam, StaveTie } from 'https://esm.sh/vexflow@5.0.0';

const STAVE_HEIGHT = 140;
const STAVE_PADDING = 40;
const NOTE_WIDTH = 50;
const MAX_NOTES_PER_MEASURE = 16;
const NOTE_SPACING = 18;
const FIRST_STAVE_OVERHEAD = 70;
const REST_PLACEHOLDER_KEY = 'b/4';
const DEFAULT_WIDTH = 400;
const STAVE_X_OFFSET = 10;
const STAVE_Y_OFFSET = 20;
const DEFAULT_NUM_BEATS = 4;
const DEFAULT_BEAT_VALUE = 4;

function buildStaveNote(noteData, clef) {
    const isRest = noteData.duration.endsWith('r');
    const keys = isRest ? [REST_PLACEHOLDER_KEY] : noteData.keys;
    const note = new StaveNote({ clef, keys, duration: noteData.duration });
    if (noteData.dots > 0) {
        Dot.buildAndAttach([note], { all: true });
    }
    return note;
}

function drawVoice(voiceData, stave, staveData, context) {
    const vfNotes = voiceData.notes.map(noteData => buildStaveNote(noteData, staveData.clef));
    const numBeats = staveData.time_signature ? staveData.time_signature[0] : DEFAULT_NUM_BEATS;
    const beatValue = staveData.time_signature ? staveData.time_signature[1] : DEFAULT_BEAT_VALUE;
    const voice = new Voice({ num_beats: numBeats, beat_value: beatValue }).setStrict(false);
    voice.addTickables(vfNotes);
    const formatWidth = staveData.time_signature
        ? stave.getNoteEndX() - stave.getNoteStartX()
        : Math.min(stave.getWidth() - STAVE_PADDING, vfNotes.length * NOTE_WIDTH);
    new Formatter().joinVoices([voice]).format([voice], formatWidth);
    const nonRestNotes = vfNotes.filter((_, i) => !voiceData.notes[i].duration.endsWith('r'));
    const beams = Beam.generateBeams(nonRestNotes);
    voice.draw(context, stave);
    beams.forEach(beam => beam.setContext(context).draw());
    drawTies(voiceData, vfNotes, context);
    return { voiceData, vfNotes };
}

function drawTies(voiceData, vfNotes, context) {
    for (let index = 0; index < voiceData.notes.length - 1; index += 1) {
        const firstData = voiceData.notes[index];
        const secondData = voiceData.notes[index + 1];
        if (!firstData.tie_start || !secondData.tie_stop) {
            continue;
        }
        if (firstData.duration.endsWith('r') || secondData.duration.endsWith('r')) {
            continue;
        }
        const noteCount = Math.min(firstData.keys.length, secondData.keys.length);
        const indices = Array.from({ length: noteCount }, (_, noteIndex) => noteIndex);
        const tie = new StaveTie({
            first_note: vfNotes[index],
            last_note: vfNotes[index + 1],
            first_indices: indices,
            last_indices: indices,
        });
        tie.setContext(context).draw();
    }
}

function drawStave(staveData, context, x, y, width, showClef) {
    const stave = new Stave(x, y, width);
    if (staveData.clef === 'percussion') {
        stave.setConfigForLines([
            { visible: false },
            { visible: false },
            { visible: true },
            { visible: false },
            { visible: false },
        ]);
    }
    if (showClef) {
        stave.addClef(staveData.clef);
    }
    if (staveData.time_signature) {
        stave.addTimeSignature(`${staveData.time_signature[0]}/${staveData.time_signature[1]}`);
    }
    stave.setContext(context).draw();
    const voiceResults = [];
    for (const voice of staveData.voices) {
        voiceResults.push(drawVoice(voice, stave, staveData, context));
    }
    return { staveData, voiceResults };
}

function drawTiesAcrossStaves(leftResult, rightResult, context) {
    const voiceCount = Math.min(leftResult.voiceResults.length, rightResult.voiceResults.length);
    for (let voiceIndex = 0; voiceIndex < voiceCount; voiceIndex += 1) {
        const leftVoice = leftResult.voiceResults[voiceIndex];
        const rightVoice = rightResult.voiceResults[voiceIndex];
        const leftData = leftVoice.voiceData.notes[leftVoice.voiceData.notes.length - 1];
        const rightData = rightVoice.voiceData.notes[0];
        if (!leftData || !rightData || !leftData.tie_start || !rightData.tie_stop) {
            continue;
        }
        if (leftData.duration.endsWith('r') || rightData.duration.endsWith('r')) {
            continue;
        }
        const noteCount = Math.min(leftData.keys.length, rightData.keys.length);
        const indices = Array.from({ length: noteCount }, (_, noteIndex) => noteIndex);
        const tie = new StaveTie({
            first_note: leftVoice.vfNotes[leftVoice.vfNotes.length - 1],
            last_note: rightVoice.vfNotes[0],
            first_indices: indices,
            last_indices: indices,
        });
        tie.setContext(context).draw();
    }
}

/**
 * Renders a ScoreData object as an SVG into containerElement.
 * @param {Object} scoreData - { rows: Array<Array<StaveData>>, tempo: number|null }
 * @param {HTMLElement} containerElement
 */
export function renderScore(scoreData, containerElement) {
    containerElement.innerHTML = '';
    const containerWidth = containerElement.clientWidth || DEFAULT_WIDTH;
    const rows = scoreData.rows;

    const hasTimeSig = rows.some(row => row.some(stave => stave.time_signature));
    let naturalWidth = containerWidth;
    if (hasTimeSig) {
        const maxStaves = Math.max(...rows.map(r => r.length));
        const maxNotes = scoreData.max_notes_per_measure ?? MAX_NOTES_PER_MEASURE;
        const normalMeasureWidth = STAVE_PADDING + maxNotes * NOTE_SPACING;
        const firstMeasureWidth = normalMeasureWidth + FIRST_STAVE_OVERHEAD;
        const required = 2 * STAVE_X_OFFSET + firstMeasureWidth + (maxStaves - 1) * normalMeasureWidth;
        naturalWidth = Math.max(containerWidth, required);
    }

    const totalHeight = rows.length * STAVE_HEIGHT;
    const renderer = new Renderer(containerElement, Renderer.Backends.SVG);
    renderer.resize(naturalWidth, totalHeight);
    const context = renderer.getContext();
    rows.forEach((row, rowIndex) => {
        const y = rowIndex * STAVE_HEIGHT + STAVE_Y_OFFSET;
        let x = STAVE_X_OFFSET;
        const rowResults = [];
        row.forEach((staveData, colIndex) => {
            let width;
            if (!hasTimeSig) {
                width = (naturalWidth - 2 * STAVE_X_OFFSET) / row.length;
            } else {
                const normalWidth = (naturalWidth - 2 * STAVE_X_OFFSET - FIRST_STAVE_OVERHEAD) / row.length;
                width = colIndex === 0 ? normalWidth + FIRST_STAVE_OVERHEAD : normalWidth;
            }
            rowResults.push(drawStave(staveData, context, x, y, width, colIndex === 0));
            x += width;
        });
        for (let colIndex = 0; colIndex < rowResults.length - 1; colIndex += 1) {
            drawTiesAcrossStaves(rowResults[colIndex], rowResults[colIndex + 1], context);
        }
    });

    if (naturalWidth > containerWidth) {
        const svg = containerElement.querySelector('svg');
        if (svg) {
            const scale = containerWidth / naturalWidth;
            svg.setAttribute('viewBox', `0 0 ${naturalWidth} ${totalHeight}`);
            svg.setAttribute('width', containerWidth);
            svg.setAttribute('height', Math.round(totalHeight * scale));
        }
    }
}
