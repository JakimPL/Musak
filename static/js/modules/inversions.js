import { Score } from '../score.js';
import { ChordInfo } from '../chord_info.js';
import { getPath } from '../path.js';
import { renderScore } from '../shared/notation.js';
import { playSound, playAgain } from '../play.js';
import { postForm, loadJSON } from '../shared/api.js';
import { renderForm } from './form.js';

let score;
let chordData;
let submitLock = false;
let audioPath;

function updateScore(point) {
    score.update(point);
    document.getElementById('points').textContent = score.points;
    document.getElementById('total').textContent = score.total;
}

const ANSWER_BTN_CLASS = 'btn-answer';

function lockSubmitButton() {
    submitLock = true;
    const btn = document.getElementById('submit');
    btn.disabled = true;
    btn.textContent = 'Generating…';
}

function unlockSubmitButton() {
    submitLock = false;
    const btn = document.getElementById('submit');
    btn.disabled = false;
    btn.textContent = 'Generate inversion';
}

function showError(message) {
    const banner = document.getElementById('error-banner');
    if (banner) {
        banner.textContent = message;
        banner.classList.remove('hidden');
    }
}

function clearError() {
    const banner = document.getElementById('error-banner');
    if (banner) banner.classList.add('hidden');
}

function hideChordInfo() {
    document.getElementById('score_info').style.display = 'none';
    document.getElementById('chord_info').style.display = 'none';
    document.getElementById('play_again').style.display = 'none';
    chordData.type = null;
}

function showChordInfo() {
    document.getElementById('score_info').style.visibility = 'visible';
    document.getElementById('chord_info').style.visibility = 'visible';
    document.getElementById('score_container').style.visibility = 'visible';
}

function setChordInfo(data) {
    const chordType = data.base_note + data.chord_type;
    const chordInversion = inversionText(data.inversion_index);

    document.getElementById('chord_type').textContent = chordType;
    document.getElementById('chord_inversion').textContent = chordInversion;

    chordData.type = data.chord_type;
    chordData.type_name = chordData.names[data.chord_type];
    chordData.inversion = chordInversion;
    chordData.inversions_numbers = data.inversions_numbers;
}

function inversionText(index) {
    return index === 0 ? 'Root position' : 'Inversion no. ' + index;
}

function resetInversionButtons() {
    const inversionButtons = document.getElementById('inversion_buttons');
    while (inversionButtons.firstChild) {
        inversionButtons.removeChild(inversionButtons.lastChild);
    }
}

function addChordTypeButtons(chordTypes) {
    resetInversionButtons();
    const inversionButtons = document.getElementById('inversion_buttons');
    const chordInfo = document.getElementById('chord_info');

    for (const type of chordTypes) {
        const button = document.createElement('input');
        button.type = 'button';
        button.className = ANSWER_BTN_CLASS;
        button.value = chordData.names[type];

        button.addEventListener('click', function () {
            if (this.value === chordData.type_name) {
                chordInfo.style.borderColor = '#248a6d';
                this.style.backgroundColor = '#16a34a';
                this.style.borderColor = '#16a34a';
                this.style.color = 'white';
                const numberOfButtons = chordData.inversions_numbers[type];
                if (numberOfButtons <= 1) {
                    showChordInfo();
                    updateScore(1);
                } else {
                    setTimeout(() => addInversionButtons(numberOfButtons), 500);
                }
            } else {
                chordInfo.style.borderColor = '#dc2626';
                this.style.backgroundColor = '#dc2626';
                this.style.borderColor = '#dc2626';
                this.style.color = 'white';
                updateScore(0);
                document.getElementById('score_info').style.visibility = 'visible';
            }
        });

        inversionButtons.appendChild(button);
    }
}

function addInversionButtons(numberOfButtons) {
    resetInversionButtons();
    const inversionButtons = document.getElementById('inversion_buttons');
    const chordInfo = document.getElementById('chord_info');

    for (let index = 0; index < numberOfButtons; index++) {
        const button = document.createElement('input');
        button.type = 'button';
        button.className = ANSWER_BTN_CLASS;
        button.value = inversionText(index);

        button.addEventListener('click', function () {
            showChordInfo();
            if (this.value === chordData.inversion) {
                chordInfo.style.borderColor = '#248a6d';
                this.style.backgroundColor = '#16a34a';
                this.style.borderColor = '#16a34a';
                this.style.color = 'white';
                updateScore(1);
            } else {
                chordInfo.style.borderColor = '#dc2626';
                this.style.backgroundColor = '#dc2626';
                this.style.borderColor = '#dc2626';
                this.style.color = 'white';
                updateScore(0);
            }
        });

        inversionButtons.appendChild(button);
    }
}

async function onSubmit(event) {
    event.preventDefault();

    if (!submitLock) {
        clearError();
        lockSubmitButton();
        const form = document.getElementById('settings_form');
        const apiUrl = form.dataset.apiUrl;

        try {
            const response = await postForm(apiUrl, form);
            unlockSubmitButton();

            if ('directory' in response) {
                audioPath = getPath(response.directory, response.audio_source);
                playSound(audioPath);

                const jsonPath = getPath(response.directory, response.chord_info);
                const infoData = await loadJSON(jsonPath);
                setChordInfo(infoData);

                renderScore(response.score_data, document.getElementById('score_container'));
                document.getElementById('score_container').style.visibility = 'hidden';

                const scoreEl = document.getElementById('score_info');
                scoreEl.style.display = '';
                scoreEl.style.visibility = 'hidden';
                const infoEl = document.getElementById('chord_info');
                infoEl.style.display = '';
                infoEl.style.visibility = 'hidden';

                document.getElementById('play_again').style.display = '';
                document.getElementById('play_again').style.visibility = 'visible';

                score.unlock();
            }

            if ('inversions_numbers' in response) {
                chordData.inversions_numbers = response.inversions_numbers;
            }

            if ('chord_types' in response) {
                addChordTypeButtons(response.chord_types);
            }
        } catch (err) {
            unlockSubmitButton();
            showError(window.DEBUG ? `An error occurred: ${err.message}` : 'An error occurred. Please try again.');
        }
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    score = new Score();
    chordData = new ChordInfo();
    submitLock = false;
    hideChordInfo();

    const form = document.getElementById('settings_form');
    const config = await loadJSON(form.dataset.configUrl);
    renderForm(document.getElementById('form_fields'), config);

    form.addEventListener('submit', onSubmit);
    document.getElementById('play_again').addEventListener('click', () => playAgain(audioPath));
});
