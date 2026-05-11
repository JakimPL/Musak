import { Score } from '../score.js';
import { getPath } from '../path.js';
import { playSound, playAgain } from '../play.js';
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

function lockSubmitButton() {
    submitLock = true;
    document.getElementById('submit').style.opacity = '0.6';
}

function unlockSubmitButton() {
    submitLock = false;
    document.getElementById('submit').style.opacity = '1.0';
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
        button.className = 'input';
        button.value = name.replaceAll('_', ' ');

        button.addEventListener('click', function () {
            document.getElementById('score_info').style.display = '';
            document.getElementById('interval_info').style.display = '';
            document.getElementById('interval_info').style.visibility = 'visible';
            document.getElementById('interval_image').style.visibility = 'visible';

            if (this.value === document.getElementById('interval').textContent) {
                intervalInfo.style.borderColor = '#248a6d';
                this.style.background = 'green';
                updateScore(1);
            } else {
                intervalInfo.style.borderColor = 'red';
                this.style.background = 'red';
                updateScore(0);
            }
        });

        intervalButtons.appendChild(button);
    }
}

async function onSubmit(event) {
    event.preventDefault();

    if (!submitLock) {
        lockSubmitButton();
        const form = document.getElementById('settings_form');
        const apiUrl = form.dataset.apiUrl;

        try {
            const response = await postForm(apiUrl, form);
            unlockSubmitButton();

            if ('directory' in response) {
                audioPath = getPath(response.directory, response.audio_source);
                playSound(audioPath);

                const jsonPath = getPath(response.directory, response.interval_info);
                const infoData = await loadJSON(jsonPath);
                setIntervalInfo(infoData);

                const imagePath = getPath(response.directory, response.image_source);
                const img = document.getElementById('interval_image');
                img.style.visibility = 'hidden';
                img.setAttribute('src', imagePath);

                document.getElementById('play_again').style.display = '';
                document.getElementById('play_again').style.visibility = 'visible';
                document.getElementById('interval_info').style.visibility = 'hidden';

                score.unlock();
            }

            if ('intervals' in response) {
                addButtons(response.intervals);
            }
        } catch (err) {
            unlockSubmitButton();
            alert(window.DEBUG ? `An error occurred: ${err.message}` : 'An error occurred');
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
