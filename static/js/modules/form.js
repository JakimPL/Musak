/**
 * Renders a form from a config schema JSON object.
 * Schema shape: { groups: [{ label, fields: [{ name, type, label, default, min, max }] }] }
 *
 * @param {HTMLElement} container - The element to render fields into
 * @param {Object} schema - Config schema from GET /api/.../config
 */
export function renderForm(container, schema) {
    container.innerHTML = '';

    for (const group of schema.groups) {
        const groupDiv = document.createElement('div');
        groupDiv.className = 'mb-3';

        const label = document.createElement('p');
        label.innerHTML = `<span class="text-xs font-bold uppercase tracking-wide text-gray-500 dark:text-gray-400">${group.label}</span>`;
        groupDiv.appendChild(label);

        for (const field of group.fields) {
            const fieldDiv = document.createElement('div');
            fieldDiv.className = 'mt-1';
            fieldDiv.appendChild(createField(field));
            groupDiv.appendChild(fieldDiv);
        }

        container.appendChild(groupDiv);
    }
}

const INPUT_CLASSES = 'form-input';
const LABEL_CLASSES = 'form-label';

function createField(field) {
    const wrapper = document.createElement('div');

    if (field.type === 'boolean') {
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.name = field.name;
        input.id = field.name;
        if (field.default) input.checked = true;
        input.value = 'on';
        input.className = 'mr-2 accent-primary';
        input.setAttribute('aria-label', field.label);

        const lbl = document.createElement('label');
        lbl.htmlFor = field.name;
        lbl.textContent = field.label;
        lbl.className = 'text-sm text-gray-700 dark:text-gray-300 cursor-pointer';

        wrapper.appendChild(input);
        wrapper.appendChild(lbl);
    } else if (field.type === 'slider') {
        const lbl = document.createElement('label');
        lbl.htmlFor = field.name;
        lbl.className = LABEL_CLASSES;

        const valueSpan = document.createElement('span');
        valueSpan.className = 'float-right font-bold tabular-nums';
        valueSpan.textContent = field.default;
        lbl.textContent = field.label + ' ';
        lbl.appendChild(valueSpan);

        const input = document.createElement('input');
        input.type = 'range';
        input.name = field.name;
        input.id = field.name;
        input.value = field.default;
        input.step = '1';
        if (field.min !== null && field.min !== undefined) input.min = field.min;
        if (field.max !== null && field.max !== undefined) input.max = field.max;
        input.className = 'form-slider';
        input.setAttribute('aria-label', field.label);
        input.addEventListener('input', () => { valueSpan.textContent = input.value; });

        wrapper.appendChild(lbl);
        wrapper.appendChild(input);
    } else if (field.type === 'integer') {
        const lbl = document.createElement('label');
        lbl.htmlFor = field.name;
        lbl.textContent = field.label;
        lbl.className = LABEL_CLASSES;

        const input = document.createElement('input');
        input.type = 'number';
        input.name = field.name;
        input.id = field.name;
        input.value = field.default;
        if (field.min !== null && field.min !== undefined) input.min = field.min;
        if (field.max !== null && field.max !== undefined) input.max = field.max;
        input.className = INPUT_CLASSES;
        input.setAttribute('aria-label', field.label);

        wrapper.appendChild(lbl);
        wrapper.appendChild(input);
    } else if (field.type === 'text') {
        const lbl = document.createElement('label');
        lbl.htmlFor = field.name;
        lbl.textContent = field.label;
        lbl.className = LABEL_CLASSES;

        const input = document.createElement('input');
        input.type = 'text';
        input.name = field.name;
        input.id = field.name;
        input.value = field.default || '';
        input.className = INPUT_CLASSES;
        input.setAttribute('aria-label', field.label);

        wrapper.appendChild(lbl);
        wrapper.appendChild(input);
    }

    return wrapper;
}

