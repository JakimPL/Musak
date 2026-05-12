import { playAgain, stopSound } from '../play.js';
import { postForm, loadJSON } from '../shared/api.js';
import { renderForm } from './form.js';
import { renderScore } from '../shared/notation.js';

let submitLock = false;
let audioPath;

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
    btn.textContent = 'Generate rhythm';
}

function showError(message) {
    const banner = document.getElementById('error_banner');
    if (banner) {
        banner.textContent = message;
        banner.classList.remove('hidden');
    }
}

function clearError() {
    const banner = document.getElementById('error_banner');
    if (banner) banner.classList.add('hidden');
}

function hideScore() {
    const play = document.getElementById('play');
    play.style.display = 'none';
    play.style.visibility = 'hidden';

    const container = document.getElementById('score_container');
    container.innerHTML = '';
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

            if ('score_data' in response && response.score_data) {
                audioPath = response.audio_data;

                const container = document.getElementById('score_container');
                renderScore(response.score_data, container);

                const play = document.getElementById('play');
                play.style.display = '';
                play.style.visibility = 'visible';
            }

            if (response.exception) {
                showError(`An error during generating the rhythm: ${response.exception}`);
            } else if (response.time_signature_error) {
                showError('The denominator must be a power of two.');
            }

        } catch (err) {
            unlockSubmitButton();
            showError(window.DEBUG ? `An error occurred: ${err.message}` : 'An error occurred. Please try again.');
        }
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    submitLock = false;
    hideScore();

    const form = document.getElementById('settings_form');
    const config = await loadJSON(form.dataset.configUrl);
    renderForm(document.getElementById('form_fields'), config);

    form.addEventListener('submit', onSubmit);
    document.getElementById('play').addEventListener('click', () => playAgain(audioPath));
});
