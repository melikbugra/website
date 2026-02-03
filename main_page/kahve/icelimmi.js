const hayirBtn = document.getElementById('hayir-btn');
const evetBtn = document.getElementById('evet-btn');
const question = document.getElementById('question');
const buttonsDiv = document.querySelector('.buttons');
const result = document.getElementById('result');
const container = document.getElementById('main-container');

function moveButton() {
    // Ekran sınırlarını al (butonun dışarı taşmaması için)
    const maxX = window.innerWidth - hayirBtn.clientWidth - 20;
    const maxY = window.innerHeight - hayirBtn.clientHeight - 20;

    // Rastgele yeni pozisyon (en az 10px kenarlardan pay bırak)
    const randomX = Math.max(10, Math.floor(Math.random() * maxX));
    const randomY = Math.max(10, Math.floor(Math.random() * maxY));

    hayirBtn.style.position = 'fixed';
    hayirBtn.style.left = randomX + 'px';
    hayirBtn.style.top = randomY + 'px';
    hayirBtn.style.right = 'auto';
    hayirBtn.style.bottom = 'auto';
    hayirBtn.style.transform = 'none'; // Merkeze sabitleyen transformu kaldır
}

// Masaüstü için
hayirBtn.addEventListener('mouseover', moveButton);

// Mobil için (Dokunduğu anda kaçsın)
hayirBtn.addEventListener('touchstart', (e) => {
    e.preventDefault(); // Tıklama olayını engelle
    moveButton();
});

// Evet butonuna tıklama
evetBtn.addEventListener('click', () => {
    document.querySelector('.emoji-header').style.display = 'none';
    question.classList.add('hidden');
    buttonsDiv.classList.add('hidden');
    result.classList.remove('hidden');
    hayirBtn.style.display = 'none';

    fireworks();
    createBalloons();
});

function fireworks() {
    const duration = 5 * 1000;
    const animationEnd = Date.now() + duration;
    const defaults = { startVelocity: 30, spread: 360, ticks: 60, zIndex: 1000 };

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
    for (let i = 0; i < 25; i++) {
        const balloon = document.createElement('div');
        balloon.className = 'balloon';
        balloon.style.left = Math.random() * 100 + 'vw';
        balloon.style.backgroundColor = `hsl(${Math.random() * 360}, 70%, 70%)`;
        balloon.style.animationDelay = Math.random() * 2 + 's';
        balloon.style.animationDuration = (Math.random() * 3 + 4) + 's';
        document.body.appendChild(balloon);
    }
}
