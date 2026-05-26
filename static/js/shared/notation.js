import {
    Accidental,
    Barline,
    Renderer,
    Stave,
    StaveNote,
    Voice,
    Formatter,
    Dot,
    Beam,
    StaveTie,
    StaveConnector,
} from 'https://esm.sh/vexflow@5.0.0';

const STAVE_HEIGHT = 140;
const GRAND_STAFF_HEIGHT = 220;
const GRAND_STAFF_GAP = 80;
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
    if (!isRest) {
        const accidentals = noteData.accidentals?.length
            ? noteData.accidentals
            : keys.map(key => accidentalFromKey(key));
        accidentals.forEach((accidental, index) => {
            if (accidental) {
                note.addModifier(new Accidental(accidental), index);
            }
        });
    }
    if (noteData.dots > 0) {
        Dot.buildAndAttach([note], { all: true });
    }
    return note;
}

function accidentalFromKey(key) {
    const pitch = key.split('/')[0] ?? '';
    if (pitch.includes('#')) {
        return '#';
    }
    if (pitch.includes('b')) {
        return 'b';
    }
    return null;
}

function buildVoiceResult(voiceData, stave, staveData) {
    const vfNotes = voiceData.notes.map(noteData => buildStaveNote(noteData, staveData.clef));
    vfNotes.forEach(note => note.setStave(stave));
    const numBeats = staveData.time_signature ? staveData.time_signature[0] : DEFAULT_NUM_BEATS;
    const beatValue = staveData.time_signature ? staveData.time_signature[1] : DEFAULT_BEAT_VALUE;
    const voice = new Voice({ num_beats: numBeats, beat_value: beatValue }).setStrict(false);
    voice.addTickables(vfNotes);
    return { voiceData, voice, stave, staveData, vfNotes };
}

function formatVoiceResults(voiceResults, formatWidth) {
    if (!voiceResults.length) {
        return;
    }
    const voices = voiceResults.map(result => result.voice);
    const formatter = new Formatter();
    voiceResults.forEach(result => formatter.joinVoices([result.voice]));
    formatter.format(voices, formatWidth);
}

function drawVoiceResult(voiceResult, context) {
    const { voiceData, voice, stave, vfNotes } = voiceResult;
    const nonRestNotes = vfNotes.filter((_, i) => !voiceData.notes[i].duration.endsWith('r'));
    const beams = Beam.generateBeams(nonRestNotes);
    voice.draw(context, stave);
    beams.forEach(beam => beam.setContext(context).draw());
    drawTies(voiceData, vfNotes, context);
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

function buildStaveResult(staveData, context, x, y, width, showClef) {
    const stave = new Stave(x, y, width);
    stave.setBegBarType(Barline.type.SINGLE);
    stave.setEndBarType(Barline.type.SINGLE);
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
    if (staveData.key_signature) {
        stave.addKeySignature(staveData.key_signature);
    }
    if (staveData.time_signature) {
        stave.addTimeSignature(`${staveData.time_signature[0]}/${staveData.time_signature[1]}`);
    }
    stave.setContext(context).draw();
    drawExplicitBarline(context, stave, x);
    drawExplicitBarline(context, stave, x + width);
<<<<<<< HEAD
    const voiceResults = [];
    for (const voice of staveData.voices) {
        voiceResults.push(drawVoice(voice, stave, staveData, context));
    }
    return { stave, staveData, voiceResults };
=======
    const voiceResults = staveData.voices.map(voiceData => buildVoiceResult(voiceData, stave, staveData));
    return { staveData, voiceResults };
>>>>>>> b154b11 (Synchronized: VexFlow staffs)
}

function formatStaveResult(staveResult) {
    const formatWidth = staveFormatWidth(staveResult, staveResult.voiceResults);
    formatVoiceResults(staveResult.voiceResults, formatWidth);
}

function staveFormatWidth(staveResult, voiceResults) {
    const { staveData } = staveResult;
    const stave = voiceResults[0]?.stave;
    if (!stave) {
        return 0;
    }
    if (hasMeasureHeader(staveData)) {
        return stave.getNoteEndX() - stave.getNoteStartX();
    }

    const maxNotes = Math.max(...voiceResults.map(result => result.vfNotes.length), 0);
    return Math.min(stave.getWidth() - STAVE_PADDING, maxNotes * NOTE_WIDTH);
}

function drawStaveResult(staveResult, context) {
    for (const voiceResult of staveResult.voiceResults) {
        drawVoiceResult(voiceResult, context);
    }
}

function hasMeasureHeader(staveData) {
    return Boolean(staveData.key_signature || staveData.time_signature);
}

function drawExplicitBarline(context, stave, x) {
    context.beginPath();
    context.moveTo(x, stave.getYForLine(0));
    context.lineTo(x, stave.getYForLine(4));
    context.stroke();
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

function synchronizedRowBlocks(rows) {
    const blocks = [];
    for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
        const row = rows[rowIndex];
        const nextRow = rows[rowIndex + 1];
        if (nextRow && isGrandStaffPair(row, nextRow)) {
            blocks.push([rowIndex, rowIndex + 1]);
            rowIndex += 1;
        } else {
            blocks.push([rowIndex]);
        }
    }
    return blocks;
}

function isGrandStaffPair(topRow, bottomRow) {
    return (
        topRow.length === bottomRow.length
        && topRow.some(staveData => staveData.clef === 'treble')
        && bottomRow.some(staveData => staveData.clef === 'bass')
    );
}

function formatSynchronizedColumns(rowResultsByIndex, rowBlock) {
    const columnCount = Math.max(...rowBlock.map(rowIndex => rowResultsByIndex[rowIndex].length));
    for (let colIndex = 0; colIndex < columnCount; colIndex += 1) {
        const columnStaves = rowBlock
            .map(rowIndex => rowResultsByIndex[rowIndex][colIndex])
            .filter(Boolean);
        const columnVoices = columnStaves.flatMap(staveResult => staveResult.voiceResults);
        if (!columnVoices.length) {
            continue;
        }
        const formatWidth = staveFormatWidth(columnStaves[0], columnVoices);
        formatVoiceResults(columnVoices, formatWidth);
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
    const isGrandStaff = scoreData.layout === 'grand_staff';

    const hasMeasureHeaders = rows.some(row => row.some(stave => hasMeasureHeader(stave)));
    let naturalWidth = containerWidth;
    if (hasMeasureHeaders) {
        const maxStaves = Math.max(...rows.map(r => isGrandStaff ? Math.ceil(r.length / 2) : r.length));
        const maxNotes = scoreData.max_notes_per_measure ?? MAX_NOTES_PER_MEASURE;
        const normalMeasureWidth = STAVE_PADDING + maxNotes * NOTE_SPACING;
        const firstMeasureWidth = normalMeasureWidth + FIRST_STAVE_OVERHEAD;
        const required = 2 * STAVE_X_OFFSET + firstMeasureWidth + (maxStaves - 1) * normalMeasureWidth;
        naturalWidth = Math.max(containerWidth, required);
    }

    const totalHeight = rows.length * (isGrandStaff ? GRAND_STAFF_HEIGHT : STAVE_HEIGHT);
    const renderer = new Renderer(containerElement, Renderer.Backends.SVG);
    renderer.resize(naturalWidth, totalHeight);
    const context = renderer.getContext();
<<<<<<< HEAD
    rows.forEach((row, rowIndex) => {
        if (isGrandStaff) {
            drawGrandStaffRow(row, rowIndex, context, naturalWidth, hasMeasureHeaders);
        } else {
            drawSeparateRow(row, rowIndex, context, naturalWidth, hasMeasureHeaders);
=======
    const rowResultsByIndex = rows.map((row, rowIndex) => {
        const y = rowIndex * STAVE_HEIGHT + STAVE_Y_OFFSET;
        let x = STAVE_X_OFFSET;
        return row.map((staveData, colIndex) => {
            const width = staveWidth(row.length, colIndex, naturalWidth, hasMeasureHeaders);
            const staveResult = buildStaveResult(staveData, context, x, y, width, colIndex === 0);
            x += width;
            return staveResult;
        });
    });

    for (const rowBlock of synchronizedRowBlocks(rows)) {
        if (rowBlock.length > 1) {
            formatSynchronizedColumns(rowResultsByIndex, rowBlock);
        } else {
            for (const staveResult of rowResultsByIndex[rowBlock[0]]) {
                formatStaveResult(staveResult);
            }
        }
    }

    rowResultsByIndex.forEach(rowResults => {
        rowResults.forEach(staveResult => drawStaveResult(staveResult, context));
        for (let colIndex = 0; colIndex < rowResults.length - 1; colIndex += 1) {
            drawTiesAcrossStaves(rowResults[colIndex], rowResults[colIndex + 1], context);
>>>>>>> b154b11 (Synchronized: VexFlow staffs)
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

<<<<<<< HEAD
function drawSeparateRow(row, rowIndex, context, naturalWidth, hasMeasureHeaders) {
    const y = rowIndex * STAVE_HEIGHT + STAVE_Y_OFFSET;
    let x = STAVE_X_OFFSET;
    const rowResults = [];
    row.forEach((staveData, colIndex) => {
        const width = staveWidth(naturalWidth, row.length, colIndex, hasMeasureHeaders);
        rowResults.push(drawStave(staveData, context, x, y, width, colIndex === 0));
        x += width;
    });
    for (let colIndex = 0; colIndex < rowResults.length - 1; colIndex += 1) {
        drawTiesAcrossStaves(rowResults[colIndex], rowResults[colIndex + 1], context);
    }
}

function drawGrandStaffRow(row, rowIndex, context, naturalWidth, hasMeasureHeaders) {
    const y = rowIndex * GRAND_STAFF_HEIGHT + STAVE_Y_OFFSET;
    let x = STAVE_X_OFFSET;
    const measureCount = Math.ceil(row.length / 2);
    const rightResults = [];
    const leftResults = [];
    for (let measureIndex = 0; measureIndex < measureCount; measureIndex += 1) {
        const rightData = row[2 * measureIndex];
        const leftData = row[2 * measureIndex + 1];
        if (!rightData || !leftData) {
            continue;
        }
        const width = staveWidth(naturalWidth, measureCount, measureIndex, hasMeasureHeaders);
        const showClef = measureIndex === 0;
        const rightResult = drawStave(rightData, context, x, y, width, showClef);
        const leftResult = drawStave(leftData, context, x, y + GRAND_STAFF_GAP, width, showClef);
        drawGrandStaffConnectors(rightResult.stave, leftResult.stave, context, { showBrace: measureIndex === 0 });
        rightResults.push(rightResult);
        leftResults.push(leftResult);
        x += width;
    }
    for (let measureIndex = 0; measureIndex < rightResults.length - 1; measureIndex += 1) {
        drawTiesAcrossStaves(rightResults[measureIndex], rightResults[measureIndex + 1], context);
        drawTiesAcrossStaves(leftResults[measureIndex], leftResults[measureIndex + 1], context);
    }
}

function staveWidth(naturalWidth, staveCount, colIndex, hasMeasureHeaders) {
    if (!hasMeasureHeaders) {
        return (naturalWidth - 2 * STAVE_X_OFFSET) / staveCount;
    }
    const normalWidth = (naturalWidth - 2 * STAVE_X_OFFSET - FIRST_STAVE_OVERHEAD) / staveCount;
    return colIndex === 0 ? normalWidth + FIRST_STAVE_OVERHEAD : normalWidth;
}

function drawGrandStaffConnectors(rightStave, leftStave, context, { showBrace }) {
    if (showBrace) {
        new StaveConnector(rightStave, leftStave)
            .setType(StaveConnector.type.BRACE)
            .setContext(context)
            .draw();
    }
    new StaveConnector(rightStave, leftStave)
        .setType(StaveConnector.type.SINGLE_LEFT)
        .setContext(context)
        .draw();
}
=======
function staveWidth(rowLength, colIndex, naturalWidth, hasMeasureHeaders) {
    if (!hasMeasureHeaders) {
        return (naturalWidth - 2 * STAVE_X_OFFSET) / rowLength;
    }

    const normalWidth = (naturalWidth - 2 * STAVE_X_OFFSET - FIRST_STAVE_OVERHEAD) / rowLength;
    return colIndex === 0 ? normalWidth + FIRST_STAVE_OVERHEAD : normalWidth;
}
>>>>>>> b154b11 (Synchronized: VexFlow staffs)
