import { Score } from '../score.js';
import { playSound, playAgain, stopSound } from '../play.js';
import { renderScore } from '../shared/notation.js';
import { postForm, loadJSON } from '../shared/api.js';
import { renderForm } from './form.js';

let score;
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
    btn.textContent = 'Generate interval';
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

function hideIntervalInfo() {
    document.getElementById('score_info').style.display = 'none';
    document.getElementById('interval_info').style.display = 'none';
    document.getElementById('play_again').style.display = 'none';
}

function setIntervalInfo(data) {
    const intervalType = data.base_note + ' (' + data.interval + ')';
    document.getElementById('interval_type').textContent = intervalType;
    document.getElementById('interval').textContent = data.name.replaceAll('_', ' ');
}

function addButtons(intervals) {
    const intervalButtons = document.getElementById('interval_buttons');
    const intervalInfo = document.getElementById('interval_info');
    while (intervalButtons.firstChild) {
        intervalButtons.removeChild(intervalButtons.lastChild);
    }

    for (const [name] of Object.entries(intervals)) {
        const button = document.createElement('input');
        button.type = 'button';
        button.className = ANSWER_BTN_CLASS;
        button.value = name.replaceAll('_', ' ');

        button.addEventListener('click', function () {
            document.getElementById('score_container').style.visibility = 'visible';
            document.getElementById('score_info').style.visibility = 'visible';
            document.getElementById('interval_info').style.visibility = 'visible';

            if (this.value === document.getElementById('interval').textContent) {
                intervalInfo.style.borderColor = '#248a6d';
                this.style.backgroundColor = '#16a34a';
                this.style.borderColor = '#16a34a';
                this.style.color = 'white';
                updateScore(1);
            } else {
                intervalInfo.style.borderColor = '#dc2626';
                this.style.backgroundColor = '#dc2626';
                this.style.borderColor = '#dc2626';
                this.style.color = 'white';
                updateScore(0);
            }
        });

        intervalButtons.appendChild(button);
    }
}

async function onSubmit(event) {
    event.preventDefault();

    if (!submitLock) {
        clearError();
        stopSound();
        lockSubmitButton();
        const form = document.getElementById('settings_form');
        const apiUrl = form.dataset.apiUrl;

        try {
            const response = await postForm(apiUrl, form);
            unlockSubmitButton();

            if (response.audio_data) {
                audioPath = response.audio_data;
                playSound(audioPath);

                setIntervalInfo(response.interval_info);

                renderScore(response.score_data, document.getElementById('score_container'));
                document.getElementById('score_container').style.visibility = 'hidden';

                const scoreEl = document.getElementById('score_info');
                scoreEl.style.display = '';
                scoreEl.style.visibility = 'hidden';
                const infoEl = document.getElementById('interval_info');
                infoEl.style.display = '';
                infoEl.style.visibility = 'hidden';

                document.getElementById('play_again').style.display = '';
                document.getElementById('play_again').style.visibility = 'visible';

                score.unlock();
            }

            if ('intervals' in response) {
                addButtons(response.intervals);
            }
        } catch (err) {
            unlockSubmitButton();
            showError(window.DEBUG ? `An error occurred: ${err.message}` : 'An error occurred. Please try again.');
        }
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    score = new Score();
    submitLock = false;
    hideIntervalInfo();

    const form = document.getElementById('settings_form');
    const config = await loadJSON(form.dataset.configUrl);
    renderForm(document.getElementById('form_fields'), config);

    form.addEventListener('submit', onSubmit);
    document.getElementById('play_again').addEventListener('click', () => playAgain(audioPath));
});
