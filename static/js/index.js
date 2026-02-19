document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('overlayCanvas');
    const ctx = canvas.getContext('2d');
    const videoElement = document.getElementById('videoStream');
    
    const btnFreeze = document.getElementById('btnFreeze');
    const btnSave = document.getElementById('btnSave');
    const statusBadge = document.getElementById('statusBadge');
    
    let startX, startY, currentX, currentY;
    let isDrawing = false;
    let isFrozen = false;
    let rect = null; 

    // Atualiza o status visual
    function setStatus(text, color = "#333", textColor = "#fff") {
        statusBadge.innerText = text;
        statusBadge.style.backgroundColor = color;
        statusBadge.style.color = textColor;
    }

    // Sincroniza a resolução interna do Canvas com a resolução real da câmera (1280x960)
    videoElement.onload = function() {
        // Ignora o tamanho visual da tela e foca na matriz de pixels real
        canvas.width = 1280;
        canvas.height = 960;
    };

    // Função auxiliar para calcular a posição exata do mouse (com correção de escala CSS)
    function getMousePos(e) {
        const rectBounds = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rectBounds.width;
        const scaleY = canvas.height / rectBounds.height;
        
        return {
            x: (e.clientX - rectBounds.left) * scaleX,
            y: (e.clientY - rectBounds.top) * scaleY
        };
    }

    canvas.addEventListener('mousedown', (e) => {
        if (!isFrozen) {
            alert("⚠️ Congele a tela primeiro para selecionar o objeto com precisão!");
            return;
        }
        const pos = getMousePos(e);
        startX = pos.x;
        startY = pos.y;
        isDrawing = true;
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;
        const pos = getMousePos(e);
        currentX = pos.x;
        currentY = pos.y;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawRect(startX, startY, currentX - startX, currentY - startY);
    });

    canvas.addEventListener('mouseup', (e) => {
        if (!isDrawing) return;
        isDrawing = false;
        
        let w = currentX - startX;
        let h = currentY - startY;
        
        if (w === 0 || h === 0) return;

        rect = {
            x: Math.round(w < 0 ? currentX : startX),
            y: Math.round(h < 0 ? currentY : startY),
            w: Math.round(Math.abs(w)),
            h: Math.round(Math.abs(h))
        };

        // Envia para o backend instantaneamente
        sendCrop();
    });

    function drawRect(x, y, w, h) {
        ctx.strokeStyle = '#dc3545'; // Vermelho
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 6]); // Tracejado maneiro
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = 'rgba(220, 53, 69, 0.15)'; // Fundo vermelho transparente
        ctx.fillRect(x, y, w, h);
    }

    async function sendCrop() {
        ctx.clearRect(0, 0, canvas.width, canvas.height); 
        ctx.setLineDash([]); 

        const formData = new URLSearchParams();
        formData.append('box', JSON.stringify(rect));
        
        setStatus("Processando IA...", "#0d6efd"); // Azul

        try {
            await fetch('/set_target', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: formData.toString()
            });
            setStatus("Alvo segmentado! Revise e salve.", "#198754"); // Verde
        } catch (error) {
            setStatus("Erro na IA", "#dc3545"); // Vermelho
        }
    }

    // Funções anexadas ao objeto window para serem acessíveis no onclick (ou adicione listeners)
    window.toggleFreeze = async function() {
        try {
            const response = await fetch('/toggle_freeze', { method: 'POST' });
            const data = await response.json();
            
            isFrozen = data.is_frozen;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (isFrozen) {
                btnFreeze.innerHTML = '<span class="icon">▶️</span> Descongelar Tela';
                btnFreeze.className = "btn btn-unfreeze";
                btnSave.style.display = "flex"; 
                setStatus("Tela Congelada. Desenhe a Bounding Box!", "#ffc107", "#000"); // Amarelo
            } else {
                btnFreeze.innerHTML = '<span class="icon">❄️</span> Congelar Tela';
                btnFreeze.className = "btn btn-freeze";
                btnSave.style.display = "none"; 
                setStatus("Vídeo rodando normalmente...", "#333");
            }
        } catch (error) {
            setStatus("Erro ao congelar", "#dc3545");
        }
    };

    window.saveToDataset = async function() {
        try {
            const response = await fetch('/save_dataset', { method: 'POST' });
            const data = await response.json();
            
            if(data.status === "success") {
                setStatus(`✅ Salvo: ${data.message}`, "#198754");
                
                // Descongela automaticamente após 1.5s para fluxo de trabalho rápido
                setTimeout(() => { toggleFreeze(); }, 1500);
            } else {
                setStatus(`❌ Erro: ${data.message}`, "#dc3545");
            }
        } catch (error) {
            setStatus("Erro ao salvar", "#dc3545");
        }
    };

    // Associa os botões às funções
    btnFreeze.addEventListener('click', toggleFreeze);
    btnSave.addEventListener('click', saveToDataset);
});