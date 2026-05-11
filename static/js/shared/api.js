/**
 * Posts a form to the given URL using the format expected by the Django backend:
 * submit=<url-encoded-form-data>&csrfmiddlewaretoken=<token>
 * Returns parsed JSON response.
 */
export async function postForm(url, formElement) {
    const formData = new FormData(formElement);
    const csrfToken = formData.get('csrfmiddlewaretoken') || '';
    const serialized = new URLSearchParams(formData).toString();

    const body = new URLSearchParams({
        submit: serialized,
        csrfmiddlewaretoken: csrfToken,
    });

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: body.toString(),
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error_message || response.statusText);
    }
    return response.json();
}

/**
 * Loads JSON from the given URL. Returns parsed data.
 */
export async function loadJSON(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to load JSON from ${url}: ${response.statusText}`);
    }
    return response.json();
}
