/**
 * Posts a form to the given URL as application/x-www-form-urlencoded.
 * Returns parsed JSON response.
 * @param {string} url
 * @param {HTMLFormElement} formElement
 * @param {((loading: boolean) => void) | null} onLoading - optional callback fired with true before request, false after
 */
export async function postForm(url, formElement, onLoading = null) {
    onLoading?.(true);
    const body = new URLSearchParams(new FormData(formElement));
    try {
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
    } finally {
        onLoading?.(false);
    }
}

/**
 * Loads JSON from the given URL. Returns parsed data.
 * @param {string} url
 * @param {((loading: boolean) => void) | null} onLoading - optional callback fired with true before request, false after
 */
export async function loadJSON(url, onLoading = null) {
    onLoading?.(true);
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load JSON from ${url}: ${response.statusText}`);
        }
        return response.json();
    } finally {
        onLoading?.(false);
    }
}

