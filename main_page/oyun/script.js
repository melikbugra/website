const hayirBtn = document.getElementById('hayir-btn');
const evetBtn = document.getElementById('evet-btn');
const question = document.getElementById('question');
const buttonsDiv = document.querySelector('.buttons');
const result = document.getElementById('result');
const container = document.getElementById('main-container');

// Hayır butonunu kaçırma fonksiyonu
hayirBtn.addEventListener('mouseover', () => {
    const i = Math.floor(Math.random() * (window.innerWidth - hayirBtn.clientWidth));
    const j = Math.floor(Math.random() * (window.innerHeight - hayirBtn.clientHeight));

    hayirBtn.style.position = 'fixed';
    hayirBtn.style.left = i + 'px';
    hayirBtn.style.top = j + 'px';
});

// Evet butonuna tıklama
evetBtn.addEventListener('click', () => {
    question.classList.add('hidden');
    buttonsDiv.classList.add('hidden');
    result.classList.remove('hidden');
    hayirBtn.style.display = 'none';

    // Havai fişekler (Confetti)
    fireworks();

    // Balonlar
    createBalloons();
});

function fireworks() {
    const duration = 5 * 1000;
    const animationEnd = Date.now() + duration;
    const defaults = { startVelocity: 30, spread: 360, ticks: 60, zIndex: 0 };

    function randomInRange(min, max) {
        return Math.random() * (max - min) + min;
    }

    const interval = setInterval(function () {
        const timeLeft = animationEnd - Date.now();

        if (timeLeft <= 0) {
            return clearInterval(interval);
        }

        const particleCount = 50 * (timeLeft / duration);
        confetti(Object.assign({}, defaults, { particleCount, origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 } }));
        confetti(Object.assign({}, defaults, { particleCount, origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 } }));
    }, 250);
}

function createBalloons() {
    const container = document.body;
    for (let i = 0; i < 20; i++) {
        const balloon = document.createElement('div');
        balloon.className = 'balloon';
        balloon.style.left = Math.random() * 100 + 'vw';
        balloon.style.backgroundColor = `hsl(${Math.random() * 360}, 70%, 70%)`;
        balloon.style.animationDelay = Math.random() * 2 + 's';
        balloon.style.animationDuration = (Math.random() * 3 + 3) + 's';
        container.appendChild(balloon);
    }
}
