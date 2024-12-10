fetch('http://localhost:8765/translate', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        text: "Hello, how are you?"
    })
})
    .then(response => response.json())
    .then(data =>
    {
        console.log('Translated Text:', data.translated_text);
    })
    .catch((error) =>
    {
        console.error('Error:', error);
    });
