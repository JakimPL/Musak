/**
 * Renders a form from a config schema JSON object.
 * Schema shape: { groups: [{ label, fields: [{ name, type, label, default, min, max }] }] }
 *
 * @param {HTMLElement} container - The element to render fields into
 * @param {Object} schema - Config schema from GET /api/*/config
    * @returns { HTMLFormElement } The rendered form
        */
export function renderForm(container, schema) {
    container.innerHTML = '';

    for (const group of schema.groups) {
        const groupDiv = document.createElement('div');
        groupDiv.className = 'table_row';

        const label = document.createElement('p');
        label.innerHTML = `<label><strong>${group.label}</strong></label>`;
        groupDiv.appendChild(label);

        for (const field of group.fields) {
            const fieldDiv = document.createElement('div');
            fieldDiv.appendChild(createField(field));
            groupDiv.appendChild(fieldDiv);
        }

        container.appendChild(groupDiv);
    }
}

function createField(field) {
    const wrapper = document.createElement('div');

    if (field.type === 'boolean') {
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.name = field.name;
        input.id = field.name;
        if (field.default) input.checked = true;
        input.value = 'on';

        const lbl = document.createElement('label');
        lbl.htmlFor = field.name;
        lbl.textContent = field.label;

        wrapper.appendChild(input);
        wrapper.appendChild(lbl);
    } else if (field.type === 'integer') {
        const lbl = document.createElement('label');
        lbl.htmlFor = field.name;
        lbl.innerHTML = `<strong>${field.label}:</strong>`;

        const input = document.createElement('input');
        input.type = 'number';
        input.name = field.name;
        input.id = field.name;
        input.value = field.default;
        if (field.min !== null && field.min !== undefined) input.min = field.min;
        if (field.max !== null && field.max !== undefined) input.max = field.max;

        wrapper.appendChild(lbl);
        wrapper.appendChild(input);
    } else if (field.type === 'text') {
        const lbl = document.createElement('label');
        lbl.htmlFor = field.name;
        lbl.textContent = field.label;

        const input = document.createElement('input');
        input.type = 'text';
        input.name = field.name;
        input.id = field.name;
        input.value = field.default || '';

        wrapper.appendChild(lbl);
        wrapper.appendChild(input);
    }

    return wrapper;
}
