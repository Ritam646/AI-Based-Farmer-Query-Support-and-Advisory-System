// Initialize i18next
i18next.init({
    lng: 'en',
    resources: {
        en: { translation: window.translations.en },
        hi: { translation: window.translations.hi },
        ta: { translation: window.translations.ta }
    }
}, function(err, t) {
    updateContent();
});

// Language switcher
document.getElementById('language-select').addEventListener('change', function() {
    i18next.changeLanguage(this.value, function() {
        updateContent();
    });
});

// Update content with translations
function updateContent() {
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        if (key.startsWith('[placeholder]')) {
            element.placeholder = i18next.t(key.replace('[placeholder]', ''));
        } else {
            element.innerHTML = i18next.t(key);
        }
    });
}

// Disease prediction
document.getElementById('predict-disease').addEventListener('click', async () => {
    const fileInput = document.getElementById('disease-image');
    const resultDiv = document.getElementById('disease-result');
    if (!fileInput.files[0]) {
        resultDiv.innerHTML = '<p class="text-red-500">Please upload an image.</p>';
        return;
    }
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    resultDiv.innerHTML = '<p>Loading...</p>';
    try {
        const response = await fetch('http://localhost:8000/predict_disease', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (response.ok) {
            resultDiv.innerHTML = `
                <p class="text-green-600 font-semibold">${i18next.t('results.disease')} ${data.disease}</p>
                <p>${i18next.t('results.confidence')} ${(data.confidence * 100).toFixed(2)}%</p>
            `;
        } else {
            resultDiv.innerHTML = `<p class="text-red-500">${data.detail}</p>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<p class="text-red-500">Error: ${error.message}</p>`;
    }
});

// Knowledge base query
document.getElementById('query-knowledge').addEventListener('click', async () => {
    const queryInput = document.getElementById('knowledge-query');
    const resultDiv = document.getElementById('knowledge-result');
    const query = queryInput.value.trim();
    if (!query) {
        resultDiv.innerHTML = '<p class="text-red-500">Please enter a query.</p>';
        return;
    }
    resultDiv.innerHTML = '<p>Loading...</p>';
    try {
        const response = await fetch('http://localhost:8000/rag', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        const data = await response.json();
        if (response.ok) {
            resultDiv.innerHTML = `
                <p class="text-green-600">${data.answer}</p>
                ${data.sources.length ? '<details><summary>Sources</summary><ul>' + data.sources.map(src => `<li>${src}</li>`).join('') + '</ul></details>' : ''}
            `;
        } else {
            resultDiv.innerHTML = `<p class="text-red-500">${data.detail}</p>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<p class="text-red-500">Error: ${error.message}</p>`;
    }
});

// Market analysis (placeholder)
document.getElementById('query-market').addEventListener('click', () => {
    const resultDiv = document.getElementById('market-result');
    resultDiv.innerHTML = '<p class="text-gray-600">Market analysis coming soon!</p>';
});