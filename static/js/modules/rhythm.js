import { getPath } from '../path.js';
import { playAgain } from '../play.js';
import { postForm, loadJSON } from '../shared/api.js';
import { renderForm } from './form.js';

let submitLock = false;
let audioPath;

function lockSubmitButton() {
    submitLock = true;
    document.getElementById('submit').style.opacity = '0.6';
}

function unlockSubmitButton() {
    submitLock = false;
    document.getElementById('submit').style.opacity = '1.0';
}

function hideScore() {
    const play = document.getElementById('play');
    play.style.display = 'none';
    play.style.visibility = 'hidden';

    const img = document.getElementById('rhythm_image');
    img.style.display = 'none';
    img.style.visibility = 'hidden';
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

            if ('image_source' in response) {
                audioPath = getPath(response.directory, response.audio_source);

                const img = document.getElementById('rhythm_image');
                img.setAttribute('src', response.image_source);
                img.setAttribute('alt', response.score || '');

                const play = document.getElementById('play');
                play.style.display = '';
                play.style.visibility = 'visible';

                img.style.display = '';
                img.style.visibility = 'visible';
            }

            document.getElementById('error').textContent =
                response.exception ? 'An error during generating the image:' : '';
            document.getElementById('error_message').textContent =
                response.exception || '';
            document.getElementById('time_signature_error').textContent =
                response.time_signature_error ? 'the denominator has to be a power of two!' : '';

        } catch (err) {
            unlockSubmitButton();
            document.getElementById('error').textContent = 'An error during generating the image:';
            document.getElementById('error_message').textContent = window.DEBUG ? (err.message || '') : '';
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
