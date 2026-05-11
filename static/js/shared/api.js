/**
 * Posts a form to the given URL as application/x-www-form-urlencoded.
 * Returns parsed JSON response.
 */
export async function postForm(url, formElement) {
    const body = new URLSearchParams(new FormData(formElement));

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: body.toString(),
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || response.statusText);
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
