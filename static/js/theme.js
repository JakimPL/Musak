const toggle = document.getElementById('theme-toggle');
if (toggle) {
    toggle.addEventListener('click', () => {
        const isDark = document.documentElement.classList.toggle('dark');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    });
}
